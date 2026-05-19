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




def load_exfuntube_json(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

    # rename "youtube_id" to "visual_input"
    for entry in data:
        if "youtube_id" in entry:
            entry["visual_input"] = entry.pop("youtube_id") + ".mp4"
        # rename "moments["explanation"]" to "explanation"
        if "moments" in entry and "explanation" in entry["moments"]:
            entry["caption"] = entry["moments"][0].pop("explanation")
        # rename moments to output
        if "moments" in entry:
            entry["output"] = entry.pop("moments")
        
    return data

def get_exfuntube_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    return os.path.join(root_folder, visual_input)


def get_exfuntube_amusing_frame_indices(
    annotation=None,
    task = "",
    transition=None,
    fps: int = None,
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

    def extract_segments(data):
        segs = []
        # Single interval dict
        if "start" in data and "end" in data:
            segs.append((float(data["start"]), float(data["end"])))
        else:
            # Dict of intervals
            for v in data:
                if isinstance(v, dict) and "start" in v and "end" in v:
                    segs.append((float(v["start"]), float(v["end"])))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    segs.append((float(v[0]), float(v[1])))
        return segs

    segments = extract_segments(annotation)
    if not segments:
        return []

    # Longest duration; tie-break by earliest start
    start_sec, end_sec = max(segments, key=lambda ab: (ab[1] - ab[0], -ab[0]))

    # Convert seconds → inclusive frame indices (consistent with your original code)
    start_idx = int(start_sec * fps)
    end_idx = int(end_sec * fps)
    return list(range(start_idx, end_idx + 1))