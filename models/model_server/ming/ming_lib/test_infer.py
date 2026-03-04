import os
import torch
import time
import numpy as np
from bisect import bisect_left

from transformers import (
    AutoProcessor,
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

import warnings

warnings.filterwarnings("ignore")

def generate(messages, processor, model, sys_prompt_exp=None, use_cot_system_prompt=False, max_new_tokens=512):
    text = processor.apply_chat_template(
        messages, 
        sys_prompt_exp=sys_prompt_exp,
        use_cot_system_prompt=use_cot_system_prompt
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    srt_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )

    end_time = time.time()

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # tps = generated_ids.shape[1] / (end_time - srt_time)
    # print(f"generated {generated_ids.shape[1]} tokens in {end_time - srt_time:.2f} seconds, tokens per second: {tps:.2f} tokens/s")

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

if __name__ == '__main__':
    model_name_or_path = "."
    code_path = "."
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        load_image_gen=False,
    ).to(dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(code_path, trust_remote_code=True)
    vision_path = "/input/sunyunxiao.syx/assets/"

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
                {"type": "text", "text": "What kind of flower is this?"},
            ],
        }
    ]

    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=512)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请介绍下你自己"}
            ],
        }
    ]

    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=512)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": os.path.join(vision_path, "yoga.mp4")},
                {"type": "text", "text": "What is the woman doing?"},
            ],
        }
    ]
    
    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=512)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "中国的首都是哪里？"},
            ],
        },
        {
            "role": "ASSISTANT",
            "content": [
                {"type": "text", "text": "北京"},
            ],
        },
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "它的占地面积是多少？有多少常住人口？"},
            ],
        },
    ]

    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=512)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
            ],
        }
    ]

    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=8192, use_cot_system_prompt=True)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")


    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": os.path.join(vision_path, "yoga.mp4"), "max_frames": 40, "sample": "uniform"},
                {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
                {"type": "text", "text": "What is the woman doing in the video and what kind of flower is in the image?"},
            ],
        }
    ]

    srt_time = time.time()
    output_text = generate(
        messages, processor=processor, model=model, max_new_tokens=512
    )
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "video",
                    "video": os.path.join(vision_path, "video_demo294_0.mp4"),
                },
                {
                    "type": "text",
                    "text": "我们好像快到了吧？前面那个写着沃尔玛的就是我们要去的地方吗？",
                },
            ],
        },
        {
            "role": "ASSISTANT",
            "content": [
                {
                    "type": "text",
                    "text": "是的，根据路边的指示牌，我们正在接近沃尔玛超市的停车场。",
                },
            ],
        },
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "video",
                    "video": os.path.join(vision_path, "video_demo294_1.mp4"),
                },
                {
                    "type": "audio",
                    "audio": os.path.join(vision_path, "video_demo_query.wav"),
                },
            ],
        },
    ]

    srt_time = time.time()
    output_text = generate(messages, processor=processor, model=model, max_new_tokens=512)
    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
