from typing import Optional, Dict, Any
import math
import numpy as np

import torch
from torch.nn import functional as nnf
from diffusers.models.attention import (
    _chunked_feed_forward,
)
T = torch.Tensor

def monkey_patch_region_attention_glyph_attention_select(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
):
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    training_forward = cross_attention_kwargs.pop("training_forward", True)
    if training_forward:
        return monkey_patch_region_attention_glyph_attention_training_forward(
            self,
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            cross_attention_kwargs,
            class_labels,
            added_cond_kwargs,
        )
    else:
        return monkey_patch_region_attention_glyph_attention_inference_cfg_forward(
            self,
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            cross_attention_kwargs,
            class_labels,
            added_cond_kwargs,
        )
    
    
def monkey_patch_region_attention_glyph_attention_training_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.FloatTensor:
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Retrieve lora scale.
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    # prepare Layout Guided CA
    glyph_mask = cross_attention_kwargs.pop('glyph_mask_dict')
    if glyph_mask is not None:
        # b,h,w,1
        glyph_mask = glyph_mask[hidden_states.shape[1]].unsqueeze(-1)
    # b, interleave_repeat_times[i] = num(element[i]) + 1
    interleave_repeat_times_tensor = cross_attention_kwargs.pop('interleave_repeat_times_tensor')
    interleave_repeat_times_list = cross_attention_kwargs.pop('interleave_repeat_times_list')
    interleave_start_indexes = cross_attention_kwargs.pop('interleave_start_indexes') 
    global_ratio = cross_attention_kwargs.pop('global_ratio')
    aspect_ratio = cross_attention_kwargs.pop('aspect_ratio')

    glyph_encoder_hidden_states = cross_attention_kwargs.pop("glyph_encoder_hidden_states", None)
    # a dict. visual_feat_len: tensor(b, visual_feat_len，text—_feat_len)
    glyph_attn_mask = cross_attention_kwargs.pop("glyph_attn_masks_dict", None)
    bg_attn_mask = cross_attention_kwargs.pop("bg_attn_masks_dict", None)
    feat_fetch_idx = cross_attention_kwargs.pop("feat_fetch_idx_dict", None)
    if glyph_attn_mask is not None:
        glyph_attn_mask = glyph_attn_mask[hidden_states.shape[1]]
    if bg_attn_mask is not None:
        bg_attn_mask = bg_attn_mask[hidden_states.shape[1]]
    if feat_fetch_idx is not None:
        # b, h, w
        feat_fetch_idx = feat_fetch_idx[hidden_states.shape[1]]
        # b, h, w, c
        feat_fetch_idx = feat_fetch_idx[:, :, :, None].repeat(1, 1, 1, hidden_states.shape[-1])
    assert encoder_attention_mask is None, "encoder_attention_mask is not supported in this block."

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # use two attention to alleviate gpu memory peak
        # fetch bg prompt
        bg_encoder_hidden_states = encoder_hidden_states[interleave_start_indexes + 1]
        glyph_attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=torch.cat([bg_encoder_hidden_states, glyph_encoder_hidden_states], dim=1),
            attention_mask=torch.cat([bg_attn_mask, glyph_attn_mask], dim=-1),
            **cross_attention_kwargs,
        )
        '''
            Layout Guided CA core code:
            1. batch forward, repeat interleave expand norm hidden states, 
            2. batch the forward to get raw attn_output
            3. merge according to bbox
        '''
        # no cfg considered here!
        # sum(interleave_repeat_times), hw, c
        norm_hidden_states = norm_hidden_states.repeat_interleave(repeats=interleave_repeat_times_tensor, dim=0)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )

        # attn_output_shape: b, resolution * resolution, c
        batch, num_tokens, c = attn_output.shape
        resolution_w= int(math.sqrt(num_tokens/aspect_ratio))
        resolution_h= int(math.sqrt(num_tokens*aspect_ratio))
        assert resolution_w*resolution_h==num_tokens
        attn_output = attn_output.view(batch, resolution_h, resolution_w, c)
        glyph_attn_output = glyph_attn_output.view(-1, resolution_h, resolution_w, c)
        # b, h, w, c
        attn_output_base = attn_output[interleave_start_indexes]
        attn_output = torch.split(
            attn_output, split_size_or_sections=interleave_repeat_times_list, dim=0
        )
        glyph_attn_output = torch.chunk(
            glyph_attn_output, chunks=glyph_attn_output.shape[0], dim=0
        )
        attn_output = sum([[i, j] for i, j in zip(attn_output, glyph_attn_output)], [])
        attn_output = torch.cat(attn_output, dim=0)
        # b, h, w, c
        attn_output = torch.gather(attn_output, dim=0, index=feat_fetch_idx)
        element_mask = 1- glyph_mask
        # TODO: Ablate whether glyph layers need global_ratio.
        attn_output = glyph_mask * attn_output + (
            element_mask * attn_output_base * global_ratio +
            element_mask * attn_output * (1 - global_ratio)
        )
        # b,num_tokens,c
        attn_output = attn_output.view(-1, num_tokens, c)

        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(
            self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
        )
    else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states

def monkey_patch_region_attention_glyph_attention_inference_cfg_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.FloatTensor:
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Retrieve lora scale.
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    # prepare Layout Guided CA
    glyph_mask = cross_attention_kwargs.pop('glyph_mask_dict')
    if glyph_mask is not None:
        # b,h,w,1
        glyph_mask = glyph_mask[hidden_states.shape[1]].unsqueeze(-1)
    # b, interleave_repeat_times[i] = num(element[i]) + 1
    interleave_repeat_times_tensor = cross_attention_kwargs.pop('interleave_repeat_times_tensor')
    interleave_repeat_times_list = cross_attention_kwargs.pop('interleave_repeat_times_list')
    interleave_start_indexes = cross_attention_kwargs.pop('interleave_start_indexes') 
    global_ratio = cross_attention_kwargs.pop('global_ratio')
    aspect_ratio = cross_attention_kwargs.pop('aspect_ratio')

    glyph_encoder_hidden_states = cross_attention_kwargs.pop("glyph_encoder_hidden_states", None)
    try:
        no_cfg_bs = glyph_encoder_hidden_states.shape[0]
    except:
        no_cfg_bs=1
    # a dict. visual_feat_len: tensor(b, visual_feat_len，text—_feat_len)
    glyph_attn_mask = cross_attention_kwargs.pop("glyph_attn_masks_dict", None)
    bg_attn_mask = cross_attention_kwargs.pop("bg_attn_masks_dict", None)
    feat_fetch_idx = cross_attention_kwargs.pop("feat_fetch_idx_dict", None)
    t_step=cross_attention_kwargs.pop("timestep", None)

    glyph_index=cross_attention_kwargs.pop("glyph_index_dict", None)

    if glyph_attn_mask is not None:
        glyph_attn_mask = glyph_attn_mask[hidden_states.shape[1]]
        NO_GLYPH=False
    else:
        NO_GLYPH=True
    if glyph_index is not None:
        glyph_index = glyph_index[hidden_states.shape[1]]
    if bg_attn_mask is not None:
        bg_attn_mask = bg_attn_mask[hidden_states.shape[1]]
    if feat_fetch_idx is not None:
        # b, h, w
        feat_fetch_idx = feat_fetch_idx[hidden_states.shape[1]]
        # b, h, w, c
        feat_fetch_idx = feat_fetch_idx[:, :, :, None].repeat(1, 1, 1, hidden_states.shape[-1])
    assert encoder_attention_mask is None, "encoder_attention_mask is not supported in this block."

    

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # use two attention to alleviate gpu memory peak
        if not NO_GLYPH:

            bg_encoder_hidden_states = encoder_hidden_states[glyph_encoder_hidden_states.shape[0]:][interleave_start_indexes + 1]
            # no need to cfg here
            glyph_attn_output = self.attn2(
                norm_hidden_states[no_cfg_bs:],
                encoder_hidden_states=torch.cat([bg_encoder_hidden_states, glyph_encoder_hidden_states], dim=1),
                attention_mask=torch.cat([bg_attn_mask, glyph_attn_mask], dim=-1),
                **cross_attention_kwargs,
            )
        '''
            Layout Guided CA core code:
            1. batch forward, repeat interleave expand norm hidden states, 
            2. batch the forward to get raw attn_output
            3. merge according to bbox
        '''
        uncond_norm_hidden_states, cond_norm_hidden_states = torch.chunk(norm_hidden_states, chunks=2, dim=0)
        cond_norm_hidden_states = cond_norm_hidden_states.repeat_interleave(repeats=interleave_repeat_times_tensor, dim=0)
        norm_hidden_states = torch.cat((uncond_norm_hidden_states, cond_norm_hidden_states), dim=0)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        ucond_attn_output, attn_output = attn_output[:no_cfg_bs], attn_output[no_cfg_bs:]
        
        # conditional part
        # attn_output_shape: b, resolution * resolution, c
        batch, num_tokens, c = attn_output.shape
        resolution_w= int(math.sqrt(num_tokens/aspect_ratio))
        resolution_h= int(math.sqrt(num_tokens*aspect_ratio))
        assert resolution_w*resolution_h==num_tokens
        attn_output = attn_output.view(batch, resolution_h, resolution_w, c)
        
        # b, h, w, c
        attn_output_base = attn_output[interleave_start_indexes]
        if not NO_GLYPH:
            attn_output = torch.split(
                attn_output, split_size_or_sections=interleave_repeat_times_list, dim=0
            )
            
            glyph_attn_output = glyph_attn_output.view(-1, resolution_h, resolution_w, c)
            glyph_attn_output = torch.chunk(
                glyph_attn_output, chunks=glyph_attn_output.shape[0], dim=0
            )
            attn_output = sum([[i, j] for i, j in zip(attn_output, glyph_attn_output)], [])
            attn_output = torch.cat(attn_output, dim=0)
            # b, h, w, c
            attn_output = torch.gather(attn_output, dim=0, index=feat_fetch_idx)
            element_mask = 1- glyph_mask
            attn_output = glyph_mask * attn_output + (
                element_mask * attn_output_base * global_ratio +
                element_mask * attn_output * (1 - global_ratio)
            )
        else:
            attn_output = torch.gather(attn_output, dim=0, index=feat_fetch_idx)
            attn_output=attn_output_base * global_ratio + attn_output * (1 - global_ratio)
        # b,num_tokens,c
        attn_output = attn_output.view(-1, num_tokens, c)
        attn_output = torch.cat((ucond_attn_output, attn_output), dim=0)

        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(
            self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
        )
    else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states
