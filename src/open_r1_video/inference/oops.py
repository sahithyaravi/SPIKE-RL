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

def load_oops_json(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def get_oops_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    return os.path.join(root_folder, set_id+"_merged", f"{index}_E_merged.mp4")


def get_oops_amusing_frame_indices(annotation=None ,
    task = "",
    transition=None,
    fps=None, total_frames=None):
    """
    Return og frame indices that lie in

        Vmain = [ 0.8·transition_t , 0.8·video_duration ]

    Parameters
    ----------
    total_frames : int
        Length of the video in frames.
    fps : float
        Frames-per-second of the source video.
    transition_t : float
        Transition timestamp *t* in seconds.

    Returns
    -------
    list[int]  –  frame numbers (0-based) inside Vmain.
    """
    transition_t = float(transition)  
    duration      = total_frames / fps                
    lo_time, hi_time = 0.8 * transition_t, 0.8 * duration

    # convert the interval to whole-frame indices
    start_f = max(0,              math.ceil (lo_time * fps))
    end_f   = min(total_frames-1, math.floor(hi_time * fps))

    if end_f < start_f:         
        return []

    return list(range(start_f, end_f + 1))