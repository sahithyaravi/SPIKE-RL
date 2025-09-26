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



def load_activitynet_json(json_path: str) -> List[Dict]:
    # load json file
    with open(json_path, 'r') as f:
        data = json.load(f)
    # rename "video" to "visual_input"
    for entry in data:
        if "video" in entry:
            entry["visual_input"] = entry.pop("video")
    return data



def get_activitynet_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    return os.path.join(root_folder, "Activity_Videos", visual_input)


def get_activitynet_amusing_frame_indices(
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
    return []  # ActivityNet does not have amusing frame annotations




def load_activitynetqa_json(json_path: str) -> List[Dict]:
    # load json file
    with open(json_path, 'r') as f:
        data = json.load(f)
    # rename "video" to "visual_input"
    for entry in data:
        if "video_path" in entry:
            entry["visual_input"] = entry.pop("video_path")
    return data



def get_activitynetqa_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    return visual_input


def get_activitynetqa_amusing_frame_indices(
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
    return []  # ActivityNet does not have amusing frame annotations