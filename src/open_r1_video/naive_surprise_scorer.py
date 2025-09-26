import torch

from torch.nn.functional import softmax
from typing import List, Dict
from PIL import Image

from tqdm import tqdm

import os
import math

import gc
import numpy as np

from .weighted_captioning_grad import caption_by_weight

import torch
from contextlib import nullcontext

import torch
import secrets

import random


import re
import ast
from typing import List, Optional
from PIL import Image
import torch

def score_frames(context_frames: List[Image.Image], model=None, processor=None) -> List[float]:
    """
    Score video frames for surprisingness.
    
    Args:
        context_frames: List of PIL Images representing video frames
        model: The vision-language model
        processor: The model processor
        
    Returns:
        List of float scores where 0.0 = surprising, 1.0 = not surprising
    """
    # Updated conversation with clearer instructions
    cap_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": """You are analyzing video frames for surprisingness. For each frame, assign a label of 1 if it is surprising and 0 if it is not.
                  1 = surprising/unexpected content
                  0 = expected content
                  Return ONLY a Python list of numbers with one entry for each frame.
                   Do not include any explanation or other text.
                  """},
                {"type": "video"},
            ],
        }
    ]
    
    # Apply chat template
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)
    
    # Process inputs (removed the undefined 'observed_frame' variable)
    cap_inputs = processor(
        text=cap_prompt,
        videos=context_frames,
        return_tensors="pt"
    )
    
    # Generate response
    with torch.no_grad():
        cap_ids = model.generate(**cap_inputs.to(model.device), max_new_tokens=50)
        raw_output = processor.batch_decode(cap_ids, skip_special_tokens=True)[0]
    
    # Process the output to extract scores
    scores = parse_scores_from_output(raw_output)
    print(f"Raw model output: {raw_output}")
    return scores

def score_frames_batch(context_frames: List[Image.Image], model=None, processor=None) -> List[float]:
    """
    Alternative batch approach - may be more efficient but depends on model capabilities.
    """
    # Create numbered image content
    image_content = []
    for i in range(len(context_frames)):
        image_content.append({"type": "image"})
    
    conv = [
        {
            "role": "user", 
            "content": [
                {"type": "text",
                 "text": f"""Analyze these {len(context_frames)} video frames for surprisingness.
                 
                 For each frame, assign:
                 1 = surprising/unexpected content
                 0 = normal/expected content
                 
                 Return ONLY a Python list like [0, 1, 0, 1] with {len(context_frames)} numbers.
                 No explanation needed."""}
            ] + image_content
        }
    ]
    
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    
    inputs = processor(
        text=prompt,
        images=context_frames,  # All frames at once
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output_ids = model.generate(**inputs.to(model.device), max_new_tokens=100)
        raw_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    scores = parse_scores_from_output(raw_output)
    # print(f"Batch output: {raw_output}")
    return scores

def parse_scores_from_output(output: str) -> List[float]:
    """
    Parse the model output to extract a list of float scores.
    
    Args:
        output: Raw string output from the model
        
    Returns:
        List of float scores
    """
    try:
        # Clean the output - remove common prefixes and find the list
        cleaned = output.lower()
        
        # Try to find content after "assistant" marker
        if "assistant" in cleaned:
            cleaned = cleaned.split("assistant")[-1]
        
        # Remove common prefixes and clean up
        cleaned = cleaned.replace(":", "").strip()
        
        # Use regex to find list-like patterns
        list_pattern = r'\[([\d\s,\.]+)\]'
        matches = re.findall(list_pattern, cleaned)
        
        if matches:
            # Take the first match and parse it
            list_content = matches[0]
            # Split by comma and convert to floats
            scores = [float(x.strip()) for x in list_content.split(',') if x.strip()]
            return scores
        
        # Alternative: try to parse the entire cleaned string as a list
        # Look for bracket patterns
        bracket_start = cleaned.find('[')
        bracket_end = cleaned.find(']')
        
        if bracket_start != -1 and bracket_end != -1:
            list_str = cleaned[bracket_start:bracket_end + 1]
            try:
                # Use ast.literal_eval for safe parsing
                scores = ast.literal_eval(list_str)
                # Convert to floats and ensure it's a list
                if isinstance(scores, (list, tuple)):
                    return [float(score) for score in scores]
            except (ValueError, SyntaxError):
                pass
        
        # Fallback: look for individual numbers
        numbers = re.findall(r'\d+\.?\d*', cleaned)
        if numbers:
            return [float(num) for num in numbers]
            
    except Exception as e:
        print(f"Error parsing scores: {e}")
        print(f"Raw output: {output}")
    
    # Return empty list if parsing fails
    return []

# Example usage with error handling
def score_frames_with_validation(context_frames: List[Image.Image], model=None, processor=None) -> List[float]:
    """
    Wrapper function that includes validation and fallback handling.
    """

    scores = score_frames_batch(context_frames, model, processor)
        
    # Validate that we got the expected number of scores
    if len(scores) != len(context_frames):
        print(f"Warning: Expected {len(context_frames)} scores, got {len(scores)}")
        
        # Pad with neutral scores if too few
        while len(scores) < len(context_frames):
            scores.append(0.5)  # neutral score
        
        # Truncate if too many
        scores = scores[:len(context_frames)]
    
    # Ensure all scores are in valid range [0.0, 1.0]
    scores = [max(0.0, min(1.0, score)) for score in scores]
    
    return scores
    



def direct_surprise_tracker(video_frames=None, model=None, processor=None, requires_grad=False):
    """
    Runs surprise scoring over a sequence of video frames.
    Args:
        video_frames: List of video frames (PIL images).
        window_size: Number of frames to consider for context.
        num_hypothesis: Number of hypotheses to generate."""
    surprise_scores = []
    surprise_scores = score_frames_with_validation(video_frames, model=model, processor=processor)

    return {
        "surprise_scores": surprise_scores,
        "frames": video_frames,
    }



    




