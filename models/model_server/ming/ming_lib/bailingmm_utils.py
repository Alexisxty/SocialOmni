import base64
import logging
import math
import os
from io import BytesIO
from tqdm.contrib.concurrent import thread_map

import numpy as np

import requests
import torch

from PIL import Image
try:
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None
from typing import Union, Tuple, List

VIDEO_FETCH_VERSION = os.environ.get("VIDEO_FETCH_VERSION", "v1")
if VIDEO_FETCH_VERSION == "v1":
    from bailingmm_utils_video import v1_fetch_video as fetch_video
else:
    from bailingmm_utils_video import v2_fetch_video as fetch_video
from bailingmm_utils_video import VideoInput

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 16384 * 32 * 32
MAX_RATIO = 200

VideoInput = Union[
    List["Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["torch.Tensor"]],
]


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def is_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("base64,") or image_file.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    elif isinstance(image_file, Image.Image):
        return True
    else:
        return False

def is_video(video_file):
    if isinstance(video_file, str) and video_file.lower().endswith(
            ('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
        return True
    else:
        return False

def is_audio(audio_file):
    if isinstance(audio_file, str) and audio_file.lower().endswith(
            (".wav", ".mp3", ".aac", ".flac", ".alac", ".m4a", ".ogg", ".wma", ".aiff", ".amr", ".au")):
        return True
    else:
        return False

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def fetch_image_wo_resize(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")

    return image

def fetch_audio(ele: dict[str, str | torch.Tensor], return_tensor="pt") -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio 不可用，当前环境不支持音频解码。")

    if "audio" in ele:
        audio = ele["audio"]
    else:
        audio = ele["audio_url"]

    if isinstance(audio, torch.Tensor):
        waveform = audio
        sample_rate: int = ele.get("sample_rate", 16000)
    elif audio.startswith("http://") or audio.startswith("https://"):
        audio_file = BytesIO(requests.get(audio, stream=True).content)
        waveform, sample_rate = torchaudio.load(audio_file)
    elif audio.startswith("file://"):
        waveform, sample_rate = torchaudio.load(audio[7:])
    else:
        waveform, sample_rate = torchaudio.load(audio)
    if return_tensor == "pt":
        return waveform, sample_rate
    else:
        return waveform.numpy(), sample_rate

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or "video_url" in ele
                            or "audio" in ele
                            or "audio_url" in ele
                            or ele["type"] in ["image", "image_url", "video", "video_url", "audio", "audio_url"]
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_reference_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> list[Image.Image] | None:
    vision_infos = extract_vision_info(conversations)
    ## Read images
    image_inputs = []

    def inner_process_func(vision_info):
        if "image" in vision_info or "image_url" in vision_info:
            res_list = []
            if "image" in vision_info and isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    res_list.append(fetch_image_wo_resize({"type": "image", "image": vision_info["image"][i]}))
            elif "image_url" in vision_info and vision_info["image_url"].get("url", None) is not None:
                vision_info["image_url"] = vision_info["image_url"].get("url")
                res_list.extend([fetch_image_wo_resize(vision_info)])
            else:
                res_list.extend([fetch_image_wo_resize(vision_info)])
            return {'image_inputs':res_list}
        else:
            return None

    vision_infos_reslist = thread_map(inner_process_func, vision_infos, disable=True)
    for res in vision_infos_reslist:
        if res is None:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
        elif 'image_inputs' in res:
            image_inputs.extend(res['image_inputs'])

    if len(image_inputs) > 1: # 当前的多图输入逻辑，只保留第一张作为vae参考
        image_inputs = [image_inputs[0]]

    if len(image_inputs) == 0:
        image_inputs = None

    return image_inputs


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, list[
    torch.Tensor | list[np.ndarray]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images, videos or audios
    image_inputs = []
    video_inputs = []
    audio_inputs = []

    def inner_process_func(vision_info):
        if "image" in vision_info or "image_url" in vision_info:
            res_list = []
            if "image" in vision_info and isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    res_list.append(fetch_image({"type": "image", "image": vision_info["image"][i]}))
            elif "image_url" in vision_info and vision_info["image_url"].get("url", None) is not None:
                vision_info["image_url"] = vision_info["image_url"].get("url")
                res_list.extend([fetch_image(vision_info)])
            else:
                res_list.extend([fetch_image(vision_info)])
            return {'image_inputs':res_list}

        elif "video" in vision_info or "video_url" in vision_info:
            if "video_url" in vision_info and vision_info["video_url"].get("url", None) is not None:
                data_value = vision_info["video_url"].get("url")
            elif "video" in vision_info and not os.path.isdir(vision_info['video']):
                data_value = vision_info['video']
            else:
                data_value = [os.path.join(vision_info['video'], frame) for frame in sorted(os.listdir(vision_info['video']))]
            vision_info['video']=data_value
            return {"video_inputs": [fetch_video(vision_info, return_metadata=True)]}

        elif "audio" in vision_info or "audio_url" in vision_info:
            if "audio" in vision_info and isinstance(vision_info["audio"], (tuple, list)):
                return {"audio_inputs":[fetch_audio(info) for info in vision_info["audio"]]}
            elif "audio_url" in vision_info and vision_info["audio_url"].get("url", None) is not None:
                vision_info["audio_url"] = vision_info["audio_url"].get("url")
                return {"audio_inputs":[fetch_audio(vision_info)]}
            else:
                return {"audio_inputs":[fetch_audio(vision_info)]}
        else:
            return None

    vision_infos_reslist = thread_map(inner_process_func, vision_infos, disable=True)
    for res in vision_infos_reslist:
        if res is None:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
        elif 'image_inputs' in res:
            image_inputs.extend(res['image_inputs'])
        elif 'video_inputs' in res:
            video_inputs.extend(res['video_inputs'])
        elif 'audio_inputs' in res:
            audio_inputs.extend(res['audio_inputs'])

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(audio_inputs) == 0:
        audio_inputs = None
    return image_inputs, video_inputs, audio_inputs


def get_closest_ratio(height: float, width: float, aspect_ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(aspect_ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return aspect_ratios[closest_ratio], float(closest_ratio)

def process_ratio(ori_h, ori_w, highres=512):
    ASPECT_RATIO_512 = {
        "0.25": [256, 1024],
        "0.26": [256, 992],
        "0.27": [256, 960],
        "0.28": [256, 928],
        "0.32": [288, 896],
        "0.33": [288, 864],
        "0.35": [288, 832],
        "0.4": [320, 800],
        "0.42": [320, 768],
        "0.48": [352, 736],
        "0.5": [352, 704],
        "0.52": [352, 672],
        "0.57": [384, 672],
        "0.6": [384, 640],
        "0.68": [416, 608],
        "0.72": [416, 576],
        "0.78": [448, 576],
        "0.82": [448, 544],
        "0.88": [480, 544],
        "0.94": [480, 512],
        "1.0": [512, 512],
        "1.07": [512, 480],
        "1.13": [544, 480],
        "1.21": [544, 448],
        "1.29": [576, 448],
        "1.38": [576, 416],
        "1.46": [608, 416],
        "1.67": [640, 384],
        "1.75": [672, 384],
        "2.0": [704, 352],
        "2.09": [736, 352],
        "2.4": [768, 320],
        "2.5": [800, 320],
        "2.89": [832, 288],
        "3.0": [864, 288],
        "3.11": [896, 288],
        "3.62": [928, 256],
        "3.75": [960, 256],
        "3.88": [992, 256],
        "4.0": [1024, 256],
    }
    ASPECT_RATIO_1024 = {
        '0.25': [512, 2048], '0.26': [512, 1984], '0.27': [512, 1920], '0.28': [512, 1856],
        '0.32': [576, 1792], '0.33': [576, 1728], '0.35': [576, 1664], '0.4': [640, 1600],
        '0.42':  [640, 1536], '0.48': [704, 1472], '0.5': [704, 1408], '0.52': [704, 1344],
        '0.57': [768, 1344], '0.6': [768, 1280], '0.68': [832, 1216], '0.72': [832, 1152],
        '0.78': [896, 1152], '0.82': [896, 1088], '0.88': [960, 1088], '0.94': [960, 1024],
        '1.0':  [1024, 1024], '1.07': [1024,  960], '1.13': [1088,  960], '1.21': [1088,  896],
        '1.29': [1152,  896], '1.38': [1152,  832], '1.46': [1216,  832], '1.67': [1280,  768],
        '1.75': [1344,  768], '2.0': [1408,  704], '2.09': [1472,  704], '2.4': [1536,  640],
        '2.5': [1600,  640], '2.89': [1664,  576], '3.0': [1728,  576], '3.11': [1792,  576],
        '3.62': [1856,  512], '3.75': [1920,  512], '3.88': [1984,  512], '4.0': [2048,  512],
    }

    ASPECT_RATIO_672 = {
        '0.28': [352, 1280],
        '0.32': [384, 1184],
        '0.38': [416, 1088],
        '0.44': [448, 1024],
        '0.52': [480, 928],
        '0.57': [512, 896],
        '0.65': [544, 832],
        '0.75': [576, 768],
        '0.83': [608, 736],
        '0.91': [640, 704],
        '1.00': [672, 672],
        '1.10': [704, 640],
        '1.21': [736, 608],
        '1.33': [768, 576],
        '1.39': [800, 576],
        '1.53': [832, 544],
        '1.69': [864, 512],
        '1.75': [896, 512],
        '1.93': [928, 480],
        '2.00': [960, 480],
        '2.21': [992, 448],
        '2.29': [1024, 448],
        '2.54': [1056, 416],
        '2.62': [1088, 416],
        '2.69': [1120, 416],
        '3.00': [1152, 384],
        '3.08': [1184, 384],
        '3.17': [1216, 384],
        '3.55': [1248, 352],
        '3.64': [1280, 352],
    }

    assert len(ASPECT_RATIO_512) == len(ASPECT_RATIO_1024)

    aspect_ratio_dict = {
        512 : ASPECT_RATIO_512,
        672 : ASPECT_RATIO_672,
        1024 : ASPECT_RATIO_1024,
    }

    if highres is None or highres is False:
        highres = 512
    elif highres is True:
        highres = 1024

    aspect_ratio = aspect_ratio_dict[min([i for i in aspect_ratio_dict], key=lambda x: abs(x - highres))]

    closest_size, _ = get_closest_ratio(ori_h, ori_w, aspect_ratios=aspect_ratio)
    closest_size = list(map(lambda x: int(x), closest_size))
    if closest_size[0] / ori_h > closest_size[1] / ori_w:
        resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
    else:
        resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
    return closest_size, resize_size


def find_first_index_of_consecutive_ones(lst):
    """
    输入一个由0和1组成的列表，返回每个连续1片段的第一个1的索引。
    
    参数:
        lst (list): 元素为0或1的列表
    
    返回:
        list: 每个连续1片段的首个1的索引列表
    """
    result = []
    i = 0
    n = len(lst)
    
    while i < n:
        if lst[i] == 1:
            # 找到一个连续1片段的开始
            result.append(i)
            # 跳过整个连续的1片段
            while i < n and lst[i] == 1:
                i += 1
        else:
            i += 1
    
    return result

def merge_consecutive_ones(lst, n):
    """
    输入一个由0和1组成的列表，将每个连续的1片段（长度 >= 1）中每n个1合并为一个1，
    要求每个连续1片段的长度必须能被n整除。
    保持0和1的相对顺序。
    
    参数:
        lst: list, 元素为0或1
        n: int, 合并的单位大小（正整数）
    
    返回:
        list: 合并后的列表
    """
    assert isinstance(lst, list), "输入必须是列表"
    assert isinstance(n, int) and n > 0, "n必须是正整数"
    
    # 遍历列表，提取连续1的段，检查每段长度是否能被n整除
    i = 0
    while i < len(lst):
        if lst[i] == 1:
            count = 0
            start = i
            # 统计连续1的个数
            while i < len(lst) and lst[i] == 1:
                count += 1
                i += 1
            # 断言：连续1的个数必须能被n整除
            assert count % n == 0, f"连续1的片段从索引{start}开始，长度为{count}，不能被n={n}整除"
        else:
            i += 1

    # 通过分组合并生成新列表
    result = []
    i = 0
    while i < len(lst):
        if lst[i] == 0:
            result.append(0)
            i += 1
        else:
            # 处理连续的1
            count = 0
            while i < len(lst) and lst[i] == 1:
                count += 1
                i += 1
            # 每n个1合并为一个1
            result.extend([1] * (count // n))
    
    return result

def get_default_image_gen_hw(image_gen_highres, image_gen_aspect_ratio):
    if image_gen_aspect_ratio is None:
        image_gen_aspect_ratio = 1.0

    closest_size, _ = process_ratio(ori_h=512, ori_w=int(512.0 * image_gen_aspect_ratio), highres=image_gen_highres)
    h, w = closest_size
    return h, w
