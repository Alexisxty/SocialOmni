import os
import torch
import numpy as np
import soundfile
import random

from modeling_bailing_talker import BailingTalker2
from AudioVAE.modeling_audio_vae import AudioVAE


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def input_wrapper(tts_text):
    for i in tts_text:
        yield i
    # return tts_text

if __name__ == '__main__':
    device = 'cuda:0'
    # mdl_path = '/heyuan2_12/workspace/wanren.pj/resource/inclusionAI/Ming-Lite-Omni-1.5'
    mdl_path = "/input/lyuyongjie.lyj/ckpts/flash2.0_dpo_hf"
    talker = BailingTalker2.from_pretrained(
        f'{mdl_path}/talker', torch_dtype=torch.bfloat16).eval().to(dtype=torch.bfloat16, device=device)
    talker.use_vllm = False
    vae = AudioVAE.from_pretrained(
        f'{mdl_path}/talker/vae', torch_dtype=torch.bfloat16).eval().to(dtype=torch.bfloat16, device=device)
    
    # tts_text = '这是一条测试语句。欢迎使用百灵。你可以问我一些问题。' 
    tts_texts = [
        "GitHub Education opens doors to new skills, tools, and a collaborative community eager to drive innovation. Join us and build a foundation for your future in technology.",
        "Once there was a Little Girl with long yellow hair — no bigger'n me — who had a pony, black as ink, with white stars all around his head and both cheeks.",
        "The development of the Ant Group, formerly known as Alipay and founded in 2014 by Jack Ma's Alibaba group.",
        # "顺风时提高警惕，逆风时笃定前行。",
        # "可是我不能停下来与你多聊会，再见。",
        # "嗯，可以的，那您告诉我一下什么时候出发呀，我现在来查一下呢？",
        # "蚂蚁集团 ant group was founded in 2014 by jack ma of alibaba group，总部在杭州。今天天气不错，温度是25度 Celsius，非常适合户外活动 like hiking or biking。",
        # "朋友圈photo帮你tag了, check下privacy setting。",
        # "GitHub 表示，Copilot Chat 测试版将通过微软的 Visual Studio 和 Visual Studio Code 应用程序向“所有企业用户”开放。",
        # "因为在Transformer中最多的multi head attention和Mask multi head attention来自Scaled dot product attention，而scaled dot product attention来自self attention，而self attention是attention的一种，",
        # "他刚搬到这座城市，He loves the weather here.",
        "空调remote没电, 用手机app调temperature吧。",
        "垃圾周四morning收, 今晚记得take out。",
        "Please remember to bring your ID card, 你必须出示证件才能进入。",
        "我正准备开启一个人的gap year。这对我而言是全新的体验，说不忐忑是假的。",
        # "在过去的几年里，Prompt engineering在自然语言处理领域得到了广泛研究。",
        # "下午的时间会更加适合办理Business Visa。",
        # "用了一段时间BB霜之后skin变得dark yellow了是什么原因呢？",
        # "当然可以！以下是一段中文和英文的笑话：中文笑话： 为什么熊猫总是抱着竹子？ 因为它们怕被“熊抱”！ 英文笑话： Why did the tomato turn red? Because it saw the salad dressing! 希望你喜欢这两个笑话！",
        ]
    
    dir_out = 'wavs_ce_tmp'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # data_metas = []
    for i, ori_text in enumerate(tts_texts):
        # for spk in ['DB30', 'lingguang']:  
        for spk in ['lingguang']:  
            this_outfile = os.path.join(dir_out, f'out{i}_ce_v1_{spk}.wav')
            all_wavs = []
            set_all_random_seed(1024)    ### 

            ## 第一次合成有一些是编译初始化，dropout
            for tts_speech, this_text, text_position, duration in talker.omni_audio_generation(
                    tts_text=input_wrapper(ori_text),
                    voice_name=spk,
                    audio_detokenizer=vae, stream=True
                ):
                ...
            print("\n=======================================================================")    
            for tts_speech, this_text, text_position, duration in talker.omni_audio_generation(
                    tts_text=input_wrapper(ori_text),
                    voice_name=spk,
                    audio_detokenizer=vae, stream=True
                ):
                all_wavs.append(tts_speech)
                # if this_text != "":
                print(this_text, '\t', len(this_text), tts_speech.shape, tts_speech.shape[1]/vae.config.sample_rate, text_position, duration)
                    
            if all_wavs:
                waveform = torch.cat(all_wavs, dim=-1)
                soundfile.write(this_outfile, waveform.T.numpy(), vae.config.sample_rate)
                print(f'save audio:\t{this_outfile}\n{ori_text}\n{len(ori_text)}\t{vae.config.sample_rate}', waveform.shape[1]/vae.config.sample_rate)
            else:
                print(f'{ori_text}\thas no valid segments')
