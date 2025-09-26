import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import math
import torch

random.seed(42)  
np.random.seed(42)



def load_bean_json(json_path: str) -> List[Dict]:
    # list *.json in folder and load them

    # load all jsons and add it to list
    dataset = []

    with open(os.path.join(json_path), "r") as f:
        data = json.load(f)
        dataset = data["scenes"]
        # 'original_file' is the video file name
        for item in dataset:
            item["visual_input"] = item["scene_file"]
        # rename volume_peak_1s to output
        for item in dataset:
            item["output"] = item["volume_peak_3s"]
        
    return dataset



def get_bean_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    return os.path.join(root_folder, visual_input)


def get_bean_amusing_frame_indices(
    annotation=None,
    task = "",
    transition=None,
    fps: int = 25,
    total_frames=None,
) -> List[int]:
    """
    Returns inclusive frame indices for the *longest* [start_sec, end_sec] interval.
    - annotation can be:
        • dict like {"0":{"start_sec":..,"end_sec":..}, ...}
        • list/tuple of [start_sec, end_sec] pairs or dicts with those keys
        • JSON string of the above
        • simple string like "[start, end]"
    - fps defaults to 25 for these videos.
    """

    start_sec = annotation["start_time"]
    end_sec = annotation["end_time"]

    # Convert seconds → inclusive frame indices (consistent with your original code)
    start_idx = int(start_sec * fps)
    end_idx = int(end_sec * fps)
    return list(range(start_idx, end_idx + 1))