import os.path as osp

import torch.nn as nn
import torch
from transformers import AutoTokenizer
from ..custom_diffusers.modeling_t5 import T5ForConditionalGeneration
from peft import LoraConfig
from diffusers.utils import logging

from .constants import huggingface_cache_dir

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def add_special_token(tokenizer, text_encoder, add_color, add_font, font_ann_path, color_ann_path, add_align=False):
    import json
    with open(font_ann_path, 'r') as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, 'r') as f:
        idx_color_dict = json.load(f)

    font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]
    color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
    align_token = [f'<align-{i}>' for i in range(3)]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token
    if add_align:
        additional_special_tokens += align_token
    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer))

def add_special_token_multilingual(tokenizer, text_encoder, add_color, add_font, font_ann_path, color_ann_path, add_align=False):
    import json

    with open(font_ann_path, 'r') as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, 'r') as f:
        idx_color_dict = json.load(f)

    font_token = []
    for font_code in idx_font_dict:
        prefix = font_code[:3]
        if prefix == 'cn-' or prefix == 'en-' or prefix == 'jp-' or prefix == 'kr-':
            font_token.append(f'<{prefix}font-{idx_font_dict[font_code]}>')
        else:
            font_token.append(f'<font-{idx_font_dict[font_code]}>')
    color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
    align_token = [f'<align-{i}>' for i in range(3)]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token
    if add_align:
        additional_special_tokens += align_token
    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer))


def load_byt5_and_byt5_tokenizer(
    byt5_ckpt_path, 
    byt5_name='google/byt5-small', 
    special_token=False, 
    color_special_token=False,
    font_special_token=False,
    align_special_token=False,
    train_text_encoder_lora=False,
    text_encoder_lora_rank=32,
    font_ann_path='glyph/font_uni_10-lang_idx.json',
    color_ann_path='glyph/color_idx.json',
    multilingual=True
):
    print(byt5_name, byt5_ckpt_path, font_ann_path, color_ann_path)
    # byt5_tokenizer = AutoTokenizer.from_pretrained(
    #     byt5_name, cache_dir=huggingface_cache_dir, # args.byt5_model_name_or_path, cache_dir=huggingface_cache_dir,
    # )
    # byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
    #     byt5_name, cache_dir=huggingface_cache_dir,
    # ).get_encoder()

    byt5_tokenizer = AutoTokenizer.from_pretrained(
        byt5_ckpt_path, cache_dir=None, # args.byt5_model_name_or_path, cache_dir=huggingface_cache_dir,
    )
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_ckpt_path, cache_dir=None,
    ).get_encoder()

    if special_token:
        if not multilingual:
            add_special_token(byt5_tokenizer, byt5_text_encoder, add_color=color_special_token, add_font=font_special_token,
            font_ann_path=font_ann_path, color_ann_path=color_ann_path)
        else:
            add_special_token_multilingual(byt5_tokenizer, byt5_text_encoder, add_color=color_special_token, add_font=font_special_token,
            font_ann_path=font_ann_path, color_ann_path=color_ann_path)

    if train_text_encoder_lora:
        try:
            text_lora_config = LoraConfig(
                r=text_encoder_lora_rank,
                lora_alpha=text_encoder_lora_rank,
                init_lora_weights="gaussian",
                target_modules=["q", "k", "v", "o"],
            )
            byt5_text_encoder.add_adapter(text_lora_config)
        except:
            raise ValueError
    
    # if byt5_ckpt_path is not None:
    #     trainable_module_dict = dict()
    #     trainable_module_dict['text_encoder'] = byt5_text_encoder
        
    #     class TrainableModuleWrapper(nn.Module):
    #         def __init__(self, trainable_modules):
    #             super().__init__()
    #             self.trainable_modules = nn.ModuleDict()
    #             self.trainable_modules.update(trainable_modules)        

    #     trainable_unet = TrainableModuleWrapper(trainable_module_dict)
    #     state_dict = {k: v for k, v in torch.load(byt5_ckpt_path, map_location='cpu').items() if k.startswith('trainable_modules.text_encoder.')}
    #     missing_keys, unexpected_keys = trainable_unet.load_state_dict(state_dict)
    #     assert missing_keys == [] and unexpected_keys == []
    #     logger.info(f'Loaded pretrained byt5 from {byt5_ckpt_path}')
    # else:
    #     logger.info(f'Loaded original byt5 weight')
    
    return byt5_text_encoder, byt5_tokenizer