
from __future__ import annotations

from decord import VideoReader, cpu
import numpy as np
import os

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
from qwen_vl_utils.vision_process import smart_nframes

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 196 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768



def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor



# def extract_k_frames_decord_cpu(video_path: str, k: int = 30):
#     vr = VideoReader(video_path, ctx=cpu(0))  # explicitly use CPU
#     total_frames = len(vr)
#     print(f"Total frames in video: {total_frames}")
#     indices = np.linspace(0, total_frames - 1, k, dtype=int)
#     frames = vr.get_batch(indices).asnumpy()  # shape: (k, H, W, 3)
#     fps           = vr.get_avg_fps()
#     video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
#     return video, indices, total_frames, fps, vr

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




# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))


def extract_k_frames_decord_cpu(video_path, image_factor: int = IMAGE_FACTOR, k=None, min_frames=None, save_image=False) -> torch.Tensor | list[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0))  # explicitly use CPU
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Calculate video duration in seconds
    duration_seconds = total_frames / fps
    
    # Calculate k based on video duration if not provided
    base_frames = 8
    
    if duration_seconds <= 60:
        k = base_frames
    else:
        # Calculate additional frames for time beyond 60 seconds
        excess_time = duration_seconds - 60
        additional_60s_blocks = int(excess_time // 60)  # Complete 100-second blocks
        additional_frames = 0
        for block in range(additional_60s_blocks):
            additional_frames += base_frames * (2 ** block)

        k = base_frames + additional_frames
    
    # if max_frames is set, ensure k is at least max_frames
    if min_frames is not None:
        k = max(k, min_frames)
        
    indices = np.linspace(0, total_frames - 1, k, dtype=int)
    video = vr.get_batch(indices).asnumpy()
    
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    frame_indices = torch.linspace(0, nframes-1, k).long()
    video = video[frame_indices]
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Convert to list of PIL images
    video = [transforms.functional.to_pil_image(frame) for frame in video]
    return video, indices, total_frames, fps, vr



def extract_k_frames_decord_cpu_duration_based(video_path) -> torch.Tensor | list[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0))  # explicitly use CPU
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    k = smart_nframes({}, total_frames, fps)

    indices = np.linspace(0, total_frames - 1, k, dtype=int)
    video = vr.get_batch(indices).asnumpy()  # shape: (k, H, W, 3)
    
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    frame_indices = torch.linspace(0, nframes-1, k).long()
    video = video[frame_indices]
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Convert to list of PIL images
    video = [transforms.functional.to_pil_image(frame) for frame in video]
    return video, indices, total_frames, fps, vr