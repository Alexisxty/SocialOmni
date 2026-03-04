import os
import torch
import time
import numpy as np
from bisect import bisect_left

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
import warnings
import argparse

warnings.filterwarnings("ignore")

speechqa_prompt = '''
You are a spoken-question answering and voice-command execution assistant.
I will provide an audio clip containing the user’s spoken question or instruction. Understand the audio and output only the final text answer or the execution result.
Rules:
1. Do not repeat the audio verbatim. Do not describe transcription. Do not explain your reasoning.
2. Be detailed and informative. Do NOT answer with a single word/phrase if the user request implies “some/examples/list/overview”. Provide concrete content.
    a. For “what is/define/explain/why/how/what if”: start the first sentence by naming the topic using the key noun from the question, then give a concise answer plus key points/reasons.
    b. For “list/give me some”: give multiple items (default 5–10) with brief descriptors when helpful, BUT do not invent items to pad the list—prefer fewer correct items over many wrong ones. Treat “give me some X” as a request for multiple examples; provide multiple items, not a single description.
    c. For factual questions: for “when/where/who/how many/how large/what year” questions, default to the shortest canonical answer. Add extra details only if explicitly asked. Otherwise, give the direct answer plus essential supporting facts (names, dates, locations, units) when appropriate; do not add uncertain specifics.
    d. For transformations/commands: output the completed final result.
    e. If the request is ambiguous, choose the most likely interpretation and provide useful content. If multiple common interpretations exist, provide 2–3 short options in the answer (do not ask a question) unless the user explicitly asks you to ask a clarifying question.
3. Output plain text only. No labels like “Answer:”, no greetings, no filler. Never output the token “Answer:”.
Outside MCQ mode, NEVER output the phrase “The answer is” and NEVER output a bare option letter (A/B/C/D) as the whole answer.
4. STRICT INSTRUCTION FOLLOWING (HIGHEST PRIORITY):
If the user specifies ANY hard constraint, you must follow it exactly even if that makes the answer shorter or less detailed. Hard constraints include:
    a. exact output format or wrapper (e.g., must start with “P.S.”, wrap entire response in quotes/<< >>, section headers like “SECTION X”, “output only …”)
    b. casing constraints (ALL CAPS / only lowercase)
    c. banned words/letters, required words/letters
    d. exact counts (number of sections/sentences/items, minimum/maximum occurrences of a letter/word)
    e. “repeat the exact request word for word first” / “do not say anything before repeating”
    When hard constraints exist, do not add anything extra outside the required format.
5. CONSTRAINT SELF-CHECK (do silently, output only final result):
Before finalizing, quickly verify:
    a. required wrapper/headers are present
    b. banned words/letters do not appear (including different casing)
    c. required counts are satisfied (sections/items; min/max letter/word occurrences when specified)
    d. the response is complete (no cut-off sentences)
6. NO CHAIN-OF-THOUGHT DISCLOSURE:
If the user explicitly asks to reveal chain-of-thought / “think out loud” / step-by-step reasoning, do not comply. Provide only the final answer/result in the required format.'''

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 32
    layer_per_gpu = num_layers // world_size
    layer_per_gpu = [i * layer_per_gpu for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)

    device_map['vision'] = 0
    device_map['audio'] = 0
    device_map['linear_proj'] = 0
    device_map['linear_proj_audio'] = 0
    device_map['model.model.word_embeddings.weight'] = 0
    device_map['model.model.norm.weight'] = 0
    device_map['model.lm_head.weight'] = 0
    device_map['model.model.norm'] = 0
    device_map[f'model.model.layers.{num_layers - 1}'] = 0
    device_map['talker'] = 0
    return device_map


class BailingMMInfer:
    def __init__(self,
                 model_name_or_path,
                 generation_config=None,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer, self.processor = self.load_model_processor()

        if generation_config is None:
            generation_config = {"num_beams": 1}
        self.generation_config = GenerationConfig.from_dict(generation_config)

    def load_model_processor(self):
        tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
        processor = AutoProcessor.from_pretrained('.', trust_remote_code=True)

        model = BailingMM2NativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=split_model(),
            load_talker=False,
        ).to(dtype=torch.bfloat16)
        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512, sys_prompt_exp=None, use_cot_system_prompt=False, lang=None):
        text = self.processor.apply_chat_template(
            messages,
            sys_prompt_exp=sys_prompt_exp,
            use_cot_system_prompt=use_cot_system_prompt
        )

        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            audio_kwargs={"use_whisper_encoder": True},
            return_tensors="pt",
        ).to(self.model.device)
        
        if lang is not None:
            language = torch.tensor([self.tokenizer.encode(f'{lang}\t')]).to(inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], language], dim=1)
            attention_mask = inputs['attention_mask']
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=attention_mask.dtype)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        srt_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.processor.gen_terminator,
                generation_config=self.generation_config,
                num_logits_to_keep=1,
            )

        end_time = time.time()
        # print(self.tokenizer.decode(generated_ids[0]))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # tps = generated_ids.shape[1] / (end_time - srt_time)
        # print(f"generated {generated_ids.shape[1]} tokens in {end_time - srt_time:.2f} seconds, tokens per second: {tps:.2f} tokens/s")

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    
    
if __name__ == '__main__':
    model_name_or_path = '/input/sunyunxiao.syx/checkpoints/bailing_native_moe_ming_flash_v2.0_xpo_final_20260205' # aistudio://12872297/Ming-Flash-2.0-20251005-HF
    model = BailingMMInfer(
        model_name_or_path,
    )

    audio_path = "data/wavs/"
    
    # ASR
    print("Testing ASR...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                },
                {"type": "audio", "audio": os.path.join(audio_path, "BAC009S0915W0292.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages, lang="Chinese")
    print(f"debug asr output:{output}")
    print(f"Generate time asr: {(time.time() - srt_time):.2f}s")

    # Dialect ASR
    print("Testing Dialect ASR...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                },
                {"type": "audio", "audio": os.path.join(audio_path, "shanghai.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages, lang="上海")
    print(f"debug dialect asr output:{output}")
    print(f"Generate time asr: {(time.time() - srt_time):.2f}s")

    # Speech QA
    print("Testing Speech QA...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": speechqa_prompt},
                {"type": "audio", "audio": os.path.join(audio_path, "speechQA_sample.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages)
    print(f"debug speechqa output:{output}")
    print(f"Generate time speechqa: {(time.time() - srt_time):.2f}s")

    # AAC
    print("Testing AAC...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "请写一句话描述这段音频。",
                },
                {"type": "audio", "audio": "data/wavs/glass-breaking-151256.mp3"},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages)
    print(f"debug aac output:{output}")
    print(f"Generate time aac: {(time.time() - srt_time):.2f}s")

    # ContextASR
    print("Testing ContextASR...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.This is an audio about Culinary Traditions.This audio may contains the following words or phrases:Gansu Province,Uyghur,Xinjiang,clay sealing method,Umsh stew,copper cauldrons.",
                },
                {"type": "audio", "audio": os.path.join(audio_path, "DLNER-013420_EN.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages)
    print(f"debug context asr output:{output}")
    print(f"Generate time asr: {(time.time() - srt_time):.2f}s")
