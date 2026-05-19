
from PIL import Image

import torch
import numpy as np


from typing import List, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

def adaptive_frame_sampling(
    scores: torch.Tensor,
    max_frames: int = 32,
    vr=None,
    temperature: float = 1.0,
    pick_mode: str = "linspace",
) -> Tuple[List[Image.Image], List[int]]:
    """
    Sample frames from a video in a differentiable way based on surprise scores.

    Args:
        scores (Tensor): 1D tensor of length N containing surprise scores for each interval.
        max_frames (int): Total number of frames to sample.
        vr: Video reader object supporting __len__, get_avg_fps() and get_batch(list[int]).
        temperature (float): Temperature parameter for the Gumbel-Softmax.
        pick_mode (str): "linspace" or "random" for choosing timestamps within each interval.

    Returns:
        sampled_frames (List[PIL.Image]): The sampled frames.
        frame_indices (List[int]): The selected frame indices in the video.
    """
    # Ensure scores is a 1D tensor
    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.to(dtype=torch.float32)
    n_intervals = scores.shape[0]

    # Baseline: allocate one frame per interval
    base_allocation = torch.ones(n_intervals, dtype=torch.long)
    remaining_frames = max_frames - n_intervals
    if remaining_frames < 0:
        raise ValueError("max_frames must be >= number of intervals")

    # Allocate extra frames using Gumbel-Softmax sampling with replacement
    extra_allocation = torch.zeros(n_intervals, dtype=torch.long)
    if remaining_frames > 0:
        # Use gumbel softmax repeatedly to allocate the remaining frames.
        # Each iteration samples one interval (with straight-through).
        logits = scores.unsqueeze(0)  # shape (1, n_intervals)
        for _ in range(remaining_frames):
            # gumbel_softmax returns a one-hot vector if hard=True
            sample = F.gumbel_softmax(logits, tau=temperature, hard=True)
            # sample shape is (1, n_intervals); accumulate counts
            extra_allocation += sample.squeeze(0).to(dtype=torch.long)
    # Total allocation: at least one per interval plus the extras
    allocation = base_allocation + extra_allocation  # shape (n_intervals,)

    # Determine total frames and duration from the video reader
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps

    # Generate time bounds for each interval
    # interval i corresponds to times [intervals[i], intervals[i+1])
    intervals = torch.linspace(0.0, duration, n_intervals + 1)

    # Collect timestamps for each interval
    time_stamps = []
    for idx, num in enumerate(allocation):
        num_frames_i = num.item()
        if num_frames_i > 0:
            start_time = intervals[idx].item()
            end_time = intervals[idx + 1].item()
            print(f"Interval {idx}: {start_time:.2f} to {end_time:.2f}, sampling {num_frames_i} frames")
            # Choose timestamps either uniformly spaced or uniformly random
            if pick_mode == "linspace":
                # endpoint=False to exclude the end time so that frames don't repeat across intervals
                ts = np.linspace(start_time, end_time, num_frames_i, endpoint=False)
            elif pick_mode == "random":
                ts = start_time + (end_time - start_time) * torch.rand(num_frames_i)
            else:
                raise ValueError(f"Unknown pick_mode: {pick_mode}")
            time_stamps.extend(ts.tolist())

    # Clip timestamps to [0, duration] and convert to frame indices
    time_stamps = torch.tensor(time_stamps).clamp(0.0, duration)
    frame_indices = (time_stamps * fps).long().unique().tolist()

    # Retrieve frames from the video reader
    # vr.get_batch expects a list of indices and returns an object with .asnumpy()
    frames_nd = vr.get_batch(frame_indices).asnumpy()
    sampled_frames = [Image.fromarray(f) for f in frames_nd]

    return sampled_frames, frame_indices


def caption_frames(model, processor, frames, scores):
    """
    Generate captions for a video using the weighted captioning method.
    
    Args:
        model: The Qwen2.5 VL model.
        processor: The processor for the model.
        video_frames: List of PIL images representing video frames.
        scores: List of scores corresponding to each frame.
        max_length: Maximum length of the generated caption.
        num_beams: Number of beams for beam search.
        device: Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        List of generated captions.
    """
    cap_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Summarize the key events in the video in 100 words."},
                {"type": "video"},
            ],
        }
    ]
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)
    cap_inputs = processor(
        text=cap_prompt,
        videos=frames,
        return_tensors="pt"
    )
    cap_ids = model.generate(**cap_inputs.to(model.device), max_new_tokens=200)
    caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].lower().split("assistant\n")[1].replace(":", "")
    return caption

def caption_by_weight(model, processor, frames, scores, vr):
    """
    Generate captions for a video using the weighted captioning method.
    
    Args:
        model: The Qwen2.5 VL model.
        processor: The processor for the model.
        video_frames: List of PIL images representing video frames.
        scores: List of scores corresponding to each frame.
        
    Returns:
        List of generated captions.
    """
    sampled_frames, frame_indices = adaptive_frame_sampling(scores=scores, vr=vr)
    caption = caption_frames(model, processor, sampled_frames, scores)
    return caption, str(frame_indices)

