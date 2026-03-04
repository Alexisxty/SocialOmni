import copy
import json
import time
import warnings
from peft import PeftModel
import torch
import torchaudio
from transformers import AutoProcessor
import os
import sys
import re
import yaml
import random
import numpy as np
from loguru import logger

from AudioVAE.modeling_audio_vae import AudioVAE
from modeling_bailing_talker import BailingTalker2


def seed_everything(seed=1895):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
warnings.filterwarnings("ignore")

BASE_CAPTION_TEMPLATE = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,

            },
            "IP": None,
        }
    ]
}


class Talker:
    def __init__(self, model_path, device='cuda:0', stream=True):
        self.device = device
        self.model = BailingTalker2.from_pretrained(
            f'{model_path}/talker').eval().to(dtype=torch.bfloat16, device=device)
        self.model.use_vllm = False
        self.vae = AudioVAE.from_pretrained(
            f'{model_path}/talker/vae').eval().to(dtype=torch.bfloat16, device=device)
        self.stream = stream

    def create_instruction(self, user_input: dict):
        new_caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
        target_item_dict = new_caption["audio_sequence"][0]

        for key, value in user_input.items():
            if key in target_item_dict:
                target_item_dict[key] = value

        if target_item_dict["BGM"].get("SNR", None) is not None:
            new_order = ["序号", "说话人", "BGM", "情感", "方言", "风格", "语速", "基频", "音量", "IP"]
            target_item_dict = {k: target_item_dict[k] for k in new_order if k in target_item_dict}
            new_caption["audio_sequence"][0] = target_item_dict

        return new_caption

    def speech_generation(
        self,
        prompt,
        text,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        instruction=None,
        prompt_wav_path=None,
        prompt_text=None,
        voice_name=None,
        max_decode_steps=200,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        output_wav_path='./out.wav',
        taskname='TTS'
    ):
        if instruction is not None:
            instruction = self.create_instruction(instruction)
            instruction = json.dumps(instruction, ensure_ascii=False)

        start_time = time.perf_counter()
        all_wavs = []

        for tts_speech, _, _, _ in self.model.instruct_audio_generation(
            prompt=prompt,
            text=text,
            use_spk_emb=use_spk_emb,
            use_zero_spk_emb=use_zero_spk_emb,
            instruction=instruction,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            voice_name=voice_name,
            max_length=50,
            audio_detokenizer=self.vae,
            stream=self.stream,
            taskname=taskname
        ):
            all_wavs.append(tts_speech)

        waveform = torch.cat(all_wavs, dim=-1)
        end_time = time.perf_counter()
        sample_points = waveform.size(1)
        sample_rate = self.vae.config.sample_rate

        audio_duration = sample_points / sample_rate
        logger.info(f"inference time cost: {end_time - start_time:.3f}s, duration: {audio_duration:.3f}s, rtf: {(end_time - start_time) / audio_duration:.3f}")

        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_wav_path, waveform, sample_rate=sample_rate)

        return waveform

    def omni_audio_generation(
        self,
        tts_text,
        voice_name='DB30',
        prompt_text=None,
        prompt_wav_path=None,
        output_wav_path='./out.wav'
    ):
        start_time = time.perf_counter()
        all_wavs = []

        idx = 0
        for tts_speech, text_list, _, _ in self.model.omni_audio_generation(
            prompt='Please generate speech based on the following description.\n',
            tts_text=tts_text,
            voice_name=voice_name,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_length=50,
            audio_detokenizer=self.vae, stream=self.stream
        ):
            all_wavs.append(tts_speech)
            logger.info(f"Current {idx} text: {text_list}")
            idx += 1
        waveform = torch.cat(all_wavs, dim=-1)
        end_time = time.perf_counter()
        sample_points = waveform.size(1)
        sample_rate = self.vae.config.sample_rate

        audio_duration = sample_points / sample_rate
        logger.info(f"inference time cost: {end_time - start_time:.3f}s, duration: {audio_duration:.3f}s, rtf: {(end_time - start_time) / audio_duration:.3f}")

        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_wav_path, waveform, sample_rate=sample_rate)

        return waveform


if __name__ == '__main__':
    model = Talker('/input/lyuyongjie.lyj/ckpts/flash2.0_dpo_hf')

    # Online TTS
    response = model.omni_audio_generation(
        tts_text='这是一条测试语句。欢迎使用百灵。你可以问我一些问题。',
        voice_name='DB30',
        output_wav_path='output/online_tts.wav'
    )
    logger.info(f"Generated Response: {response}")

    # TTA
    decode_args = {
        "max_decode_steps": 200,
        "cfg": 4.5,
        "sigma": 0.3,
        "temperature": 2.5,
        "taskname": "TTA"
    }
    messages = {
        "prompt": "Please generate audio events based on given text.\n",
        "text": "A person is snoring",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/tta.wav')
    logger.info(f"Generated Response: {response}")

    # Zero-shot TTS
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "TTS"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
        "use_spk_emb": True,
        "prompt_wav_path": "data/wavs/10002287-00000094.wav",
        "prompt_text": "在此奉劝大家别乱打美白针。"
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/tts.wav')
    logger.info(f"Generated Response: {response}")

    # BGM
    decode_args = {
        "max_decode_steps": 400,
        "taskname": "BGM"
    }
    attr = {
        "Genre": "凯尔特民间音乐.",
        "Mood": "兴奋.",
        "Instrument": "手风琴.",
        "Theme": "旅行.",
        "Duration": "60s."
    }
    text = " " + " ".join([f"{key}: {value}" for key, value in attr.items()])
    messages = {
        "prompt": "Please generate music based on the following description.\n",
        "text": text,
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/bgm.wav')
    logger.info(f"Generated Response: {response}")

    # Emotion
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "EMOTION"
    }
    instruction = {
        "情感": "高兴"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "等到七月底项目结束,我就可以申请休年假了,好期待哦!",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/0006_000038.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/emotion.wav')
    logger.info(f"Generated Response: {response}")

    # Podcast
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "PODCAST"
    }
    dialog = [
        {"speaker_1": "你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。"},
        {"speaker_2": "就是那个叫什么，变相一节课的嘛。"},
        {"speaker_1": "嗯。"},
        {"speaker_2": "一部搞笑的电影。"},
        {"speaker_1": "一部搞笑的。"}
    ]
    text = " " + "\n ".join([f"{k}:{v}" for item in dialog for k, v in item.items()]) + "\n"
    prompt_diag = [
        {"speaker_1": "并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了"},
        {"speaker_2": "对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求"},
    ]
    prompt_text = " " + "\n ".join([f"{k}:{v}" for item in prompt_diag for k, v in item.items()]) + "\n"

    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": text,
        "use_spk_emb": True,
        "prompt_wav_path": [
            "data/wavs/CTS-CN-F2F-2019-11-11-423-012-A.wav",
            "data/wavs/CTS-CN-F2F-2019-11-11-423-012-B.wav"
        ],
        "prompt_text": prompt_text
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/podcast.wav')
    logger.info(f"Generated Response: {response}")

    # Basic
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "BASIC"
    }
    instruction = {
        "语速": "快速",
        "基频": "中",
        "音量": "中",
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/10002287-00000094.wav",
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/basic.wav')
    logger.info(f"Generated Response: {response}")

    # Dialect
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "DIALECT"
    }
    instruction = {
        "方言": "广粤话"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "肯定系太想玩棍兴奋得滞",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/00000309-00000300.wav",
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/dialect.wav')
    logger.info(f"Generated Response: {response}")

    # Style
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "STYLE"
    }
    instruction = {
        "风格": "性别: 男性嗓音.\n\n音高: 男性低沉音域，语末音高略降.\n\n语速: 整体语速偏快，结尾趋缓.\n\n音量: 音量洪亮，力度感强.\n\n年龄: 中年男性.\n\n清晰度: 吐字清晰，字正腔圆.\n\n流畅度: 表达流畅，一气呵成.\n\n口音: 标准普通话，略带北方腔调.\n\n音色质感: 音色浑厚坚实，略带粗砺.\n\n情绪: 严肃果决，不容置疑.\n\n语调: 肯定式降调，命令意味浓厚.\n\n性格: 自信坚定，具有威严感.",
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我们全船全军都该换换装备了。",
        "instruction": instruction,
        "use_zero_spk_emb": True
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/style.wav')
    logger.info(f"Generated Response: {response}")

    # IP
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "IP"
    }
    instruction = {
        "IP": "水浒传_武松"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "他不行了,都怪我害了他,他就相信您叶老师,您救救他吧。",
        "instruction": instruction,
        "use_zero_spk_emb": True
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/ip.wav')
    logger.info(f"Generated Response: {response}")

    # Speech + bgm
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "SPEECH_BGM"
    }
    instruction = {
        "BGM": {"Genre": "新灵魂乐.", "Mood": "多愁善感/忧郁/孤独.", "Instrument": "原声钢琴.", "Theme": "分手.", "SNR": 10.0,
                "ENV": None}
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/00000309-00000300.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/speech_bgm.wav')
    logger.info(f"Generated Response: {response}")

    # speech+sound
    decode_args = {
        "max_decode_steps": 200,
        "taskname": "SPEECH_SOUND"
    }
    instruction = {
        "BGM": {"ENV": "Birds chirping", "SNR": 10.0, "Genre": None, "Mood": None, "Instrument": None, "Theme": None}
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/00000309-00000300.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/speech_sound.wav')
    logger.info(f"Generated Response: {response}")
