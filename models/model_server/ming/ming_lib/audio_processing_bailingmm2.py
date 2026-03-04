from typing import List, Tuple, Dict, Optional, Any, Union
import os
import copy
import importlib

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import whisper
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import TensorType
from transformers.feature_extraction_utils import FeatureExtractionMixin, BatchFeature

NORM_FACTOR_FOR_DTYPE = {
    torch.int8: 2**7,
    torch.int16: 2**15,
    torch.int32: 2**31,
    torch.int64: 2**63,
    torch.float32: 1,
    torch.float64: 1,
}

# special tokens
DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
DEFAULT_TEXT_TOKEN = '<text>'
DEFAULT_ASR_TOKEN = '<asr>'
DEFAULT_TTS_TOKEN = '<tts>'

def _load_torchaudio():
    try:
        return importlib.import_module("torchaudio")
    except Exception:  # noqa: BLE001
        return None


def _load_kaldi():
    try:
        return importlib.import_module("torchaudio.compliance.kaldi")
    except Exception:  # noqa: BLE001
        return None


class BailingMM2AudioProcessor(FeatureExtractionMixin):
    def __init__(self, wav_frontend_args: Dict[str, Any]=None, whisper_frontend_args: Dict[str, Any]=None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 16000
        if wav_frontend_args is not None:
            self.wav_frontend = WavFrontend(**wav_frontend_args)
        if whisper_frontend_args is not None:
            self.whisper_frontend = WhisperFrontend(**whisper_frontend_args)

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["wav_frontend"] = output["wav_frontend"].__dict__
        output["wav_frontend"]["cmvn"] = output["wav_frontend"]["cmvn"].tolist()
        output["wav_frontend"]["_non_persistent_buffers_set"] = list(output["wav_frontend"]["_non_persistent_buffers_set"])
        output["audio_processor_type"] = self.__class__.__name__
        if 'whisper_frontend' in output:
            output["whisper_frontend"] = output["whisper_frontend"].__dict__
        return output

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Auto-fill the cmvn file path.
        """
        result, kwargs = super().get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        if not result["wav_frontend_args"]["cmvn_file"].startswith("/"):
            # Convert to an absolute path.
            if os.path.isdir(pretrained_model_name_or_path):
                pretrained_model_dir = pretrained_model_name_or_path
            else:
                pretrained_model_dir = os.path.dirname(pretrained_model_name_or_path)
            result["wav_frontend_args"]["cmvn_file"] = os.path.join(
                pretrained_model_dir, result["wav_frontend_args"]["cmvn_file"]
            )
        return result, kwargs

    def __call__(self, audios, **kwargs) -> BatchFeature:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)

    def _preprocess_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        use_whisper_encoder: bool = False,
        maximum_audio_duration: float = -1
    ) -> torch.Tensor:
        waveform = normalize_audio_tensor(waveform, sample_rate, target_sample_rate=self.sample_rate)
        if maximum_audio_duration > 0:
            waveform = waveform[:int(maximum_audio_duration * self.sample_rate)]
        if not use_whisper_encoder:
            audio_feat = self.wav_frontend(waveform.unsqueeze(0), [len(waveform)])[0].squeeze(0)
        else:
            audio_feat = self.whisper_frontend(waveform.unsqueeze(0), [len(waveform)])[0].squeeze(0)
        return audio_feat

    def _make_batched_audios(self, audio_feat_list: List[torch.Tensor], use_whisper_encoder=False) -> Dict[str, Any]:
        audio_feats_lengths = torch.tensor([[audio_feat.shape[0] for audio_feat in audio_feat_list]], dtype=torch.long)
        if not use_whisper_encoder:
            encoder_feats_lengths = audio_feats_lengths
        else:
            # whisper + project layer has two conv
            encoder_feats_lengths = ((audio_feats_lengths-3+2*1)//2+1-3+2*1)//2+1

        audio_feats = torch.cat(audio_feat_list, dim=0).unsqueeze(0)
        return {"audio_feats": audio_feats.numpy(), "audio_feats_lengths": audio_feats_lengths.numpy(), "encoder_feats_lengths": encoder_feats_lengths}

    def preprocess(
        self,
        audios: Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(audios, List):
            audio_inputs = self._make_batched_audios([self._preprocess_audio(waveform, sr, use_whisper_encoder=kwargs.get('use_whisper_encoder', False)) for waveform, sr in audios], use_whisper_encoder=kwargs.get('use_whisper_encoder', False))
        else:
            waveform, sr = audios
            audio_inputs = self._make_batched_audios([self._preprocess_audio(waveform, sr, use_whisper_encoder=kwargs.get('use_whisper_encoder', False))])
        return BatchFeature(data=audio_inputs, tensor_type=return_tensors)


class WavFrontend(torch.nn.Module):
    """Conventional frontend structure for ASR.
    """

    def __init__(
            self,
            cmvn_file: Optional[str] = None,
            fs: int = 16000,
            window: str = 'hamming',
            n_mels: int = 80,
            frame_length: int = 25,
            frame_shift: int = 10,
            filter_length_min: int = -1,
            filter_length_max: int = -1,
            lfr_m: int = 1,
            lfr_n: int = 1,
            dither: float = 1.0,
            snip_edges: bool = True,
            upsacle_samples: bool = True,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        self.cmvn = None if self.cmvn_file is None else load_cmvn(self.cmvn_file)

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kaldi = _load_kaldi()
        if kaldi is None:
            raise RuntimeError("torchaudio/kaldi 不可用，无法执行 fbank 前端。")
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=0.0, #self.dither
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs,
                              snip_edges=self.snip_edges)

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats,
                                     batch_first=True,
                                     padding_value=0.0)
        # import ipdb;ipdb.set_trace()
        return feats_pad, feats_lens

    def forward_fbank(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kaldi = _load_kaldi()
        if kaldi is None:
            raise RuntimeError("torchaudio/kaldi 不可用，无法执行 fbank 前端。")
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=self.dither,
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs)

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, :input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens


class WhisperFrontend:
    def __init__(self, n_mels: int=128):
        self.n_mels = n_mels

    def __call__(self, input: torch.Tensor, input_lengths: List[int]):
        """
        input: [B, T]
        input_lengths: [B]
        """

        assert input.size(0) == 1

        mel = whisper.log_mel_spectrogram(input.squeeze(0), n_mels=self.n_mels).to(input.device)   # [n_mels, T]
        feats_pad = mel.transpose(0, 1).unsqueeze(0)  # [B=1, T, n_mels]
        feats_lens = torch.tensor([mel.size(1)], dtype=torch.long)  # [B=1]
        return feats_pad, feats_lens

def load_cmvn(cmvn_file):
    # src_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # with open(os.path.join(src_dir, "am.mvn"), 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    lines = ['<Nnet> \n', '<Splice> 560 560\n', '[ 0 ]\n', '<AddShift> 560 560 \n', '<LearnRateCoef> 0 [ -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 -8.311879 -8.600912 -9.615928 -10.43595 -11.21292 -11.88333 -12.36243 -12.63706 -12.8818 -12.83066 -12.89103 -12.95666 -13.19763 -13.40598 -13.49113 -13.5546 -13.55639 -13.51915 -13.68284 -13.53289 -13.42107 -13.65519 -13.50713 -13.75251 -13.76715 -13.87408 -13.73109 -13.70412 -13.56073 -13.53488 -13.54895 -13.56228 -13.59408 -13.62047 -13.64198 -13.66109 -13.62669 -13.58297 -13.57387 -13.4739 -13.53063 -13.48348 -13.61047 -13.64716 -13.71546 -13.79184 -13.90614 -14.03098 -14.18205 -14.35881 -14.48419 -14.60172 -14.70591 -14.83362 -14.92122 -15.00622 -15.05122 -15.03119 -14.99028 -14.92302 -14.86927 -14.82691 -14.7972 -14.76909 -14.71356 -14.61277 -14.51696 -14.42252 -14.36405 -14.30451 -14.23161 -14.19851 -14.16633 -14.15649 -14.10504 -13.99518 -13.79562 -13.3996 -12.7767 -11.71208 ]\n', '<Rescale> 560 560\n', '<LearnRateCoef> 0 [ 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 0.155775 0.154484 0.1527379 0.1518718 0.1506028 0.1489256 0.147067 0.1447061 0.1436307 0.1443568 0.1451849 0.1455157 0.1452821 0.1445717 0.1439195 0.1435867 0.1436018 0.1438781 0.1442086 0.1448844 0.1454756 0.145663 0.146268 0.1467386 0.1472724 0.147664 0.1480913 0.1483739 0.1488841 0.1493636 0.1497088 0.1500379 0.1502916 0.1505389 0.1506787 0.1507102 0.1505992 0.1505445 0.1505938 0.1508133 0.1509569 0.1512396 0.1514625 0.1516195 0.1516156 0.1515561 0.1514966 0.1513976 0.1512612 0.151076 0.1510596 0.1510431 0.151077 0.1511168 0.1511917 0.151023 0.1508045 0.1505885 0.1503493 0.1502373 0.1501726 0.1500762 0.1500065 0.1499782 0.150057 0.1502658 0.150469 0.1505335 0.1505505 0.1505328 0.1504275 0.1502438 0.1499674 0.1497118 0.1494661 0.1493102 0.1493681 0.1495501 0.1499738 0.1509654 ]\n', '</Nnet> \n']
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n:]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


def normalize_audio_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
    device=None,
    target_sample_rate: Optional[int] = None,
):
    # Ensure dtype == float32.
    assert waveform.dtype in NORM_FACTOR_FOR_DTYPE, f"Unsupported waveform dtype: {waveform.dtype}"
    norm_factor = NORM_FACTOR_FOR_DTYPE[waveform.dtype]
    waveform = waveform.to(torch.float32) / norm_factor

    # Remove the channel dimension.
    while len(waveform.shape) > 1:
        waveform = waveform[0]

    # Move to device.
    if device is not None:
        waveform = waveform.to(device)

    # Resample.
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        torchaudio = _load_torchaudio()
        if torchaudio is not None:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            if device is not None:
                resampler = resampler.to(device)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        else:
            # 纯 torch 回退：线性插值重采样，避免 torchaudio 二进制依赖导致导入失败。
            old_len = waveform.shape[0]
            new_len = max(1, int(round(old_len * float(target_sample_rate) / float(sample_rate))))
            waveform = F.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    return waveform
