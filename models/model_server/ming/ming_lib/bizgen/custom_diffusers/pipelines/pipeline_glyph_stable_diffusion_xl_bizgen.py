from typing import Optional, List, Union, Dict, Tuple, Callable, Any
import torch
import tqdm
from torchvision import utils as vutils
from transformers import T5Tokenizer
from ..modeling_t5 import T5EncoderModel
import torch.nn.functional as F
import numpy as np
import re

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    UNet2DConditionModel,
    KarrasDiffusionSchedulers,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    VaeImageProcessor,
    is_invisible_watermark_available,
    StableDiffusionXLLoraLoaderMixin,
    PipelineImageInput,
    adjust_lora_scale_text_encoder,
    scale_lora_layers,
    unscale_lora_layers,
    USE_PEFT_BACKEND,
    TextualInversionLoaderMixin,
    StableDiffusionXLPipelineOutput,
    ImageProjection,
    logging,
    replace_example_docstring,
    EXAMPLE_DOC_STRING,
    is_torch_xla_available,
    rescale_noise_cfg,
    retrieve_timesteps,
    XLA_AVAILABLE,
    deprecate,
)
from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class GlyphStableDiffusionXLBizGenPipeline(StableDiffusionXLPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "byt5_tokenizer",
        "text_encoder",
        "text_encoder_2",
        "byt5_text_encoder",
        "byt5_mapper",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        byt5_text_encoder: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        byt5_tokenizer: T5Tokenizer,
        byt5_mapper,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        text_feat_length: int = 512,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super(StableDiffusionXLPipeline, self).__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            byt5_text_encoder=byt5_text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            byt5_tokenizer=byt5_tokenizer,
            byt5_mapper=byt5_mapper,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(text_feat_length=text_feat_length)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_feat_length = text_feat_length

        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        text_prompt = None,
        interleave_start_indexes = np.array([0]),
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_attn_mask: Optional[torch.LongTensor] = None,
        byt5_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_prompt = [text_prompt] if isinstance(text_prompt, str) else text_prompt

        batch_size = len(text_prompt)
        assert batch_size == 1

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        # As prompts can be far longer than clip's max length, we need to split them into chunks
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            img_prompt_id_batchs = []
            for img_prompts, tokenizer in zip(prompts, tokenizers):
                pad_token = tokenizer.pad_token_id
                total_tokens = tokenizer(img_prompts, truncation=False)['input_ids']
                bos = total_tokens[0][0]
                eos = total_tokens[0][-1]
                total_tokens = [i[1:-1] for i in total_tokens]
                new_total_tokens = []
                last_sep = -1
                comma_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(',')[0])
                period_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('.')[0])
                sep_id_set = set([comma_id, period_id])
                sep_padding_backtrack = 20
                for token_ids in total_tokens:
                    new_total_tokens.append([])
                    empty_flag = True
                    head_75_tokens = []
                    while len(token_ids) + len(head_75_tokens) >= 75:
                        while len(head_75_tokens) < 75:
                            token_id = token_ids.pop(0)
                            if token_id in sep_id_set:
                                last_sep = len(head_75_tokens)
                            head_75_tokens.append(token_id)
                        if (
                            sep_padding_backtrack !=0 and 
                            last_sep != -1 and 
                            len(head_75_tokens) - last_sep <= sep_padding_backtrack
                        ):
                            break_location = last_sep + 1
                            next_segment = head_75_tokens[break_location:]
                            head_75_tokens = head_75_tokens[:break_location]
                            padding_len = 75 - len(head_75_tokens)
                            temp_77_token_ids = [bos] + head_75_tokens + [eos] + [pad_token] * padding_len
                        else:
                            next_segment = []
                            temp_77_token_ids = [bos] + head_75_tokens + [eos]
                        new_total_tokens[-1].append(temp_77_token_ids)
                        last_sep = -1
                        head_75_tokens = next_segment
                        empty_flag = False
                    if len(token_ids) > 0 or empty_flag:
                        padding_len = 75 - len(token_ids)
                        temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
                        new_total_tokens[-1].append(temp_77_token_ids)
                max_77_len = len(max(new_total_tokens, key=len))
                for new_tokens in new_total_tokens:
                    if len(new_tokens) < max_77_len:
                        padding_len = max_77_len - len(new_tokens)
                        new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
                # b,segment_len,77
                new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
                img_prompt_id_batchs.append(new_total_tokens)
            if img_prompt_id_batchs[0].shape[1] > img_prompt_id_batchs[1].shape[1]:
                tokenizer = tokenizers[1]
                pad_token = tokenizer.pad_token_id
                bos = tokenizer.bos_token_id
                eos = tokenizer.eos_token_id
                padding_len = img_prompt_id_batchs[0].shape[1] - img_prompt_id_batchs[1].shape[1]
                # padding_len, 77
                padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
                # b, padding_len, 77
                padding_part = padding_part.unsqueeze(0).repeat(img_prompt_id_batchs[1].shape[0], 1, 1)
                img_prompt_id_batchs[1] = torch.cat((img_prompt_id_batchs[1], padding_part), dim=1)
            elif img_prompt_id_batchs[0].shape[1] < img_prompt_id_batchs[1].shape[1]:
                tokenizer = tokenizers[0]
                pad_token = tokenizer.pad_token_id
                bos = tokenizer.bos_token_id
                eos = tokenizer.eos_token_id
                padding_len = img_prompt_id_batchs[1].shape[1] - img_prompt_id_batchs[0].shape[1]
                # padding_len, 77
                padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
                # b, padding_len, 77
                padding_part = padding_part.unsqueeze(0).repeat(img_prompt_id_batchs[0].shape[0], 1, 1)
                img_prompt_id_batchs[0] = torch.cat((img_prompt_id_batchs[0], padding_part), dim=1)        
            
            embeddings = []
            # print("checkpoint 4")
            for segment_idx in range(img_prompt_id_batchs[0].shape[1]):
                prompt_embeds_list = []
                for i, text_encoder in enumerate(text_encoders):
                    # b, segment_len, sequence_len
                    text_input_ids = img_prompt_id_batchs[i].to(text_encoder.device)
                    # b, sequence_len, dim
                    prompt_embeds = text_encoder(
                        text_input_ids[:, segment_idx],
                        output_hidden_states=True,
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    temp_pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)
                # b, sequence_len, dim
                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                embeddings.append(prompt_embeds)
                if segment_idx == 0:
                    # use the first segment's pooled prompt embeddings as 
                    # the pooled prompt embeddings
                    # b, dim->b, dim
                    pooled_prompt_embeds = temp_pooled_prompt_embeds.view(bs_embed, -1)
            # interleaved_b, segment_len * sequence_len, dim
            prompt_embeds = torch.cat(embeddings, dim=1)
            # b, dim
            pooled_prompt_embeds = pooled_prompt_embeds[interleave_start_indexes]
            
            if byt5_prompt_embeds is None:
                byt5_text_inputs = self.byt5_tokenizer(
                    text_prompt,
                    padding="max_length",
                    max_length=self.text_feat_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                byt5_text_input_ids = byt5_text_inputs.input_ids
                byt5_attention_mask = (
                    byt5_text_inputs.attention_mask.to(self.byt5_text_encoder.device) 
                    if text_attn_mask is None else 
                    text_attn_mask.to(
                        self.byt5_text_encoder.device, 
                        dtype=byt5_text_inputs.attention_mask.dtype
                    )
                )
                with torch.cuda.amp.autocast(enabled=False):
                    byt5_prompt_embeds = self.byt5_text_encoder(
                        byt5_text_input_ids.to(self.byt5_text_encoder.device),
                        attention_mask=byt5_attention_mask.float(),
                    )
                    byt5_prompt_embeds = byt5_prompt_embeds[0]
                    byt5_prompt_embeds = self.byt5_mapper(byt5_prompt_embeds, byt5_attention_mask)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)[:batch_size]
            negative_byt5_prompt_embeds = torch.zeros_like(byt5_prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            raise NotImplementedError

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            byt5_seq_len = negative_byt5_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.to(dtype=self.byt5_text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.view(batch_size * num_images_per_prompt, byt5_seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            batch_size * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                batch_size * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            byt5_prompt_embeds, 
            negative_byt5_prompt_embeds,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        full_img_caption=None,
        region_meta=None,
        multilingual=False,
        global_ratio: float = 0.2,
        box_bias=0,
        align_token=False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        layer_cfg_ratio: float = 0.5,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            "",
            None,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False


        # 2. Define call parameters

        batch_size = 1
        assert num_images_per_prompt == 1
        
        device = self._execution_device

        self.index_resolution_list = [[height//16,width//16],[height//32,width//32]]
        # xyxy
        base_img_size = [*region_meta[0]["bottom_right"], *region_meta[0]["bottom_right"]]
        region_meta[0] = region_meta[0].copy()
        # bg->element
        region_meta[0]['category'] = 'element'
        element_region_meta = [i for i in region_meta if i["category"] == "element"]
        NO_GLYPH=False
        glyph_region_meta = [i for i in region_meta if i["category"] == "text"]
        element_num = len(element_region_meta)
        glyph_num = len(glyph_region_meta)
        if glyph_num == 0:
            NO_GLYPH=True
        # reorder layers, remove bg
        reordered_region_meta = (
            element_region_meta + 
            glyph_region_meta
        )

        # glyph+num_element
        region_mask = torch.zeros(
            glyph_num + element_num, self.index_resolution_list[0][0], self.index_resolution_list[0][1], dtype=torch.int,
            device=device,
        )
        
        interleave_repeat_times = [element_num + 1]
        interleave_start_indexes = np.array([0])
        if not NO_GLYPH:
            text_idx_list, fake_text_prompt = self.get_text_start_pos(
                [i['text'] for i in glyph_region_meta], align_token=align_token,
                multilingual=multilingual,
            )
            glyph_mask = torch.zeros(
                self.index_resolution_list[0][0], self.index_resolution_list[0][1], self.text_feat_length,
                device=device,
            )
            glyph_idx = element_num

        element_idx = 0
        
        min_box_edge_len = min(self.index_resolution_list[0][0] // self.index_resolution_list[-1][0], self.index_resolution_list[0][1] // self.index_resolution_list[-1][1])
        resolution_size=[self.index_resolution_list[0][1],self.index_resolution_list[0][0],self.index_resolution_list[0][1],self.index_resolution_list[0][0]]

        #layer_cfg
        do_layer_cfg=False
        layer_cfg_mask = torch.zeros(
            1, height//8, width//8, dtype=torch.int,
            device=device,
        )
        for region_item in reordered_region_meta:
            
            region_box = [*region_item["top_left"], *region_item["bottom_right"]]
            region_box[2] = region_box[2] - region_box[0]
            region_box[3] = region_box[3] - region_box[1]
            region_box = [int(i / 8 + 0.5) for i in region_box]
            region_box[0] = np.clip(region_box[0], 0, width//8 - 1)
            region_box[1] = np.clip(region_box[1], 0, height//8 - 1)
            region_box[2] = np.clip(region_box[2], 1, width//8 - region_box[0])
            region_box[3] = np.clip(region_box[3], 1, height//8 - region_box[1])
            if "cfg" in region_item:
                do_layer_cfg=True
                layer_cfg_mask[0, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = region_item["cfg"]
            else:
                layer_cfg_mask[0, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = self.guidance_scale


        for region_item in reordered_region_meta:
            region_box = [*region_item["top_left"], *region_item["bottom_right"]]
            # TODO: prone to bug, double check it!
            region_box[2] = region_box[2] - region_box[0]+box_bias
            region_box[3] = region_box[3] - region_box[1]+box_bias
            region_box = [int(i / base_i * res_i + 0.5) for i, base_i, res_i in zip(region_box, base_img_size, resolution_size)]
            region_box[0] = np.clip(region_box[0], 0, resolution_size[0] - min_box_edge_len)
            region_box[1] = np.clip(region_box[1], 0, resolution_size[1] - min_box_edge_len)
            region_box[2] = np.clip(region_box[2], min_box_edge_len, resolution_size[0])
            region_box[3] = np.clip(region_box[3], min_box_edge_len, resolution_size[1])
            
            if region_item['category'] == 'element':
                region_mask[element_idx, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = 1
                # remove overlapping
                region_mask[:element_idx, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = 0
                element_idx += 1
            else:
                # category == 'text'
                region_mask[glyph_idx, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = 1
                # Note that we don't handle the overlap of texts
                region_mask[:element_num, region_box[1]: region_box[1]+region_box[3], region_box[0]: region_box[0]+region_box[2]] = 0
                glyph_mask[
                    region_box[1]: region_box[1]+region_box[3], 
                    region_box[0]: region_box[0]+region_box[2],
                    text_idx_list[glyph_idx - element_num]: text_idx_list[glyph_idx + 1 - element_num],
                ] = 1
                glyph_idx += 1
        # 1, h, w, feat_len
        if not NO_GLYPH:
            glyph_attn_masks = glyph_mask.unsqueeze(0)
            # 1, h, w
            bg_attn_masks = (torch.sum(glyph_attn_masks, dim=-1) == 0).to(dtype=glyph_attn_masks.dtype)
        else:
            bg_attn_masks = torch.ones_like(region_mask[0])

        if not NO_GLYPH:
            region_mask = torch.cat(
                (
                    region_mask[:-glyph_num], 
                    (region_mask[-glyph_num:].sum(dim=0, keepdim=True) > 0).to(dtype=torch.int),
                ),
                dim=0
            )


        feat_fetch_idx_dict = dict()
        glyph_mask_dict = dict()
        # get feat fetch index
        for idx_res in self.index_resolution_list:
            # num_element + 1, idx_res, idx_res
            region_mask_res = F.interpolate(
                region_mask[None].float(),
                size=(idx_res[0], idx_res[1]),
                mode='nearest',
            )[0].int() if idx_res != self.index_resolution_list[0] else region_mask
            # h, w
            #  +1 to skip base layer
            feat_fetch_idx = torch.argmax(region_mask_res, dim=0) + 1
            # 1,h,w
            feat_fetch_idx_dict[idx_res[0] * idx_res[1]] = feat_fetch_idx.unsqueeze(0).to(device)
            # 1, idx_res, idx_res
            if not NO_GLYPH:
                glyph_mask_dict[idx_res[0] * idx_res[1]] = region_mask_res[-1].unsqueeze(0).to(device)
        
        img_prompts = [full_img_caption] + [i['caption'] for i in element_region_meta]
        # check text layer prompt format
        if not multilingual:
            pattern = r'(<(?:color|font|align)-\d+>)'
        else:
            pattern = r'(<(?:color|en-font|cn-font|jp-font|kr-font|align)-\d+>)'
        total_text_prompt = ''
        total_sanity_text_prompt = ''
        if not NO_GLYPH:
            for text_i in glyph_region_meta:
                str_matches = re.findall(pattern, text_i['caption'])
                if not align_token:
                    if len(str_matches) == 2:
                        color_attri, font_attri = sorted(str_matches)
                    elif len(str_matches) == 3:
                        color_attri, font_attri = sorted(str_matches)[1:]
                    else:
                        raise ValueError("Invalid text prompt")
                    if "cn-font" in color_attri:
                        color_attri, font_attri = font_attri, color_attri
                    assert 'color' in color_attri and 'font' in font_attri
                    text_prompt = f'Text "{text_i["text"]}"'
                    sanity_text_prompt = text_prompt
                    
                    attr_suffix = ", ".join([color_attri, font_attri])
                    sanity_attr_suffix = ", ".join(['0', '1'])
                else:
                    algin_attri, color_attri, font_attri = sorted(str_matches)[:3]
                    assert 'color' in color_attri and 'font' in font_attri and 'align' in algin_attri
                    text_prompt = f'Text "{text_i["text"]}"'
                    sanity_text_prompt = text_prompt
                    
                    attr_suffix = ", ".join([color_attri, font_attri, algin_attri])
                    sanity_attr_suffix = ", ".join(['0', '1', '2'])
                
                text_prompt += " in " + attr_suffix
                text_prompt += ". "
                total_text_prompt = total_text_prompt + text_prompt
                
                sanity_text_prompt += " in " + sanity_attr_suffix
                sanity_text_prompt += ". "
                total_sanity_text_prompt = total_sanity_text_prompt + sanity_text_prompt
            if multilingual:
                total_sanity_text_prompt=total_sanity_text_prompt.encode('utf-8')
            # sanity check
            assert total_sanity_text_prompt == fake_text_prompt

        self.attn_mask_resolution_list = [[height//16,width//16],[height//32,width//32]]
        
        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            byt5_prompt_embeds,
            negative_byt5_prompt_embeds,
        ) = self.encode_prompt(
            prompt=img_prompts,
            prompt_2=img_prompts,
            text_prompt=total_text_prompt,
            interleave_start_indexes=interleave_start_indexes,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        
        # b, h, w
        bg_attn_masks = (1 - bg_attn_masks) * -10000.0
        bg_attn_masks_dict = dict()
        bg_attn_masks = bg_attn_masks.unsqueeze(1)
        if not NO_GLYPH:
            # b, h, w, text_feat_len
            glyph_attn_masks = (1 - glyph_attn_masks) * -10000.0
            glyph_attn_masks_dict = dict()
            
            glyph_index_dict=dict()
            # b, text_fet_len, h, w
            glyph_attn_masks = glyph_attn_masks.permute(0, 3, 1, 2)
            # b, 1, h, w
        
            for mask_resolution in self.attn_mask_resolution_list:
                if mask_resolution != glyph_attn_masks.shape[-2:]:
                    rescaled_glyph_attn_masks = F.interpolate(
                        glyph_attn_masks, size=(mask_resolution[0], mask_resolution[1]), mode='nearest',
                    )
                    rescaled_bg_attn_masks = F.interpolate(
                        bg_attn_masks, size=(mask_resolution[0], mask_resolution[1]), mode='nearest',
                    )
                else:
                    rescaled_glyph_attn_masks = glyph_attn_masks
                    rescaled_bg_attn_masks = bg_attn_masks
                
                # b, text_fet_len, h, w->b, h, w, text_fet_len->b, h*w, text_fet_len
                rescaled_glyph_attn_masks = rescaled_glyph_attn_masks.permute(0, 2, 3, 1).flatten(1, 2)

                # b,1,h,w->b,h,w->b,h,w,1->b,h*w,1->b,h*w,clip_feat_len
                rescaled_glyph_index=rescaled_bg_attn_masks.flatten().nonzero().squeeze(-1).detach().cpu().tolist()
                glyph_index_dict[mask_resolution[0] * mask_resolution[1]] = rescaled_glyph_index
                rescaled_bg_attn_masks = rescaled_bg_attn_masks.squeeze(1).unsqueeze(-1)
                rescaled_bg_attn_masks = rescaled_bg_attn_masks.flatten(1, 2)
                rescaled_bg_attn_masks = rescaled_bg_attn_masks.expand(-1, -1, prompt_embeds.shape[1])

                glyph_attn_masks_dict[mask_resolution[0] * mask_resolution[1]] = rescaled_glyph_attn_masks.to(device, dtype=prompt_embeds.dtype)
                bg_attn_masks_dict[mask_resolution[0] * mask_resolution[1]] = rescaled_bg_attn_masks.to(device, dtype=prompt_embeds.dtype)

            del glyph_attn_masks
            del bg_attn_masks
        


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            # byt5_prompt_embeds = torch.cat([negative_byt5_prompt_embeds, byt5_prompt_embeds], dim=0)

            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        byt5_prompt_embeds = byt5_prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        if NO_GLYPH:
            added_cross_attention_kwargs = dict(
                glyph_encoder_hidden_states=None,
                glyph_attn_masks_dict=None,
                bg_attn_masks_dict=None,
                feat_fetch_idx_dict=feat_fetch_idx_dict,
                global_ratio=global_ratio,
                interleave_start_indexes=interleave_start_indexes,
                interleave_repeat_times_tensor=torch.tensor(interleave_repeat_times, device=device),
                interleave_repeat_times_list=interleave_repeat_times,
                glyph_mask_dict=None,
                training_forward=False,
                aspect_ratio=height/width,
                glyph_index_dict=None,
            )
        else:
            added_cross_attention_kwargs = dict(
                glyph_encoder_hidden_states=byt5_prompt_embeds,
                glyph_attn_masks_dict=glyph_attn_masks_dict,
                bg_attn_masks_dict=bg_attn_masks_dict,
                feat_fetch_idx_dict=feat_fetch_idx_dict,
                global_ratio=global_ratio,
                interleave_start_indexes=interleave_start_indexes,
                interleave_repeat_times_tensor=torch.tensor(interleave_repeat_times, device=device),
                interleave_repeat_times_list=interleave_repeat_times,
                glyph_mask_dict=glyph_mask_dict,
                training_forward=False,
                aspect_ratio=height/width,
                glyph_index_dict=glyph_index_dict,
            )

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm.tqdm(timesteps)):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                if self.cross_attention_kwargs is None:
                    cross_attention_kwargs = {}
                else:
                    cross_attention_kwargs = self.cross_attention_kwargs
                added_cross_attention_kwargs.update(dict(timestep=t))
                cross_attention_kwargs.update(added_cross_attention_kwargs)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if do_layer_cfg and int(t)/1000<layer_cfg_ratio:
                        noise_pred = noise_pred_uncond +  torch.mul(layer_cfg_mask, (noise_pred_text - noise_pred_uncond))
                    else:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


    def get_text_start_pos(self, texts, align_token=False, multilingual=False):
        prompt = ""
        if multilingual:
            prompt = "".encode('utf-8')
        '''
        Text "{text}" in {color}, {type}, {weight}.
        '''
        pos_list = []
        for text in texts:
            pos_list.append(len(prompt))
            text_prompt = f'Text "{text}"'
            if align_token:
                attr_list = ['0', '1', '2']
            else:
                attr_list = ['0', '1']

            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
            text_prompt += ". "
            if multilingual:
                text_prompt = text_prompt.encode('utf-8')

            prompt = prompt + text_prompt
        pos_list.append(len(prompt))
        return pos_list, prompt
