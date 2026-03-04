import argparse
import torch
import time
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--config_dir', type=str, default=".")
    args = parser.parse_args()

    srt_time = time.time()
    config = AutoConfig.from_pretrained(args.config_dir)
    print(f"Load config time: {(time.time() - srt_time):.2f}s")

    srt_time = time.time()
    with no_init_weights():
        model = BailingMM2NativeForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    print(f"Init model time: {(time.time() - srt_time):.2f}s")

    srt_time = time.time()
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    state_dict.pop('audio.positional_embedding')
    print(f"Load state_dict time: {(time.time() - srt_time):.2f}s")

    srt_time = time.time()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing: {missing}")
    print(f"Unexpected: {unexpected}")
    print(f"Load weights time: {(time.time() - srt_time):.2f}s")

    srt_time = time.time()
    model.save_pretrained(args.dest_dir, safe_serialization=True)
    print(f"Save pretrained time: {(time.time() - srt_time):.2f}s")
    # python tests/models/bailingmm_moe_v2_lite/tests/convert_2_hf.py --ckpt_path /input/sunyunxiao.syx/checkpoints/Ming_Flash_2.0_test/temporary_step_interval-500-2000-0.pth --dest_dir /input/sunyunxiao.syx/checkpoints/Ming_Flash_2.0_test/