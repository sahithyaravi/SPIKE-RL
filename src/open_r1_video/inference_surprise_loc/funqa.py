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


def load_funqa_json_qa(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        dataset =  json.load(f)
    entries = ([d for d in dataset if d["task"] == "H2"])[:1000]
    # + [d for d in dataset if d["task"] == "H3"])
    # + [d for d in dataset if d["task"] == "M2"])
    print (f"Loaded {len(entries)} H2 entries from {json_path}")
    # label 'output' to 'caption' for consistency
    for entry in entries:
        entry["caption"] = entry["output"]
    return entries



def load_funqa_json(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        dataset =  json.load(f)
    entries = ([d for d in dataset if d["task"] == "H1"]
    + [d for d in dataset if d["task"] == "C1"]
    + [d for d in dataset if d["task"] == "M1"])
    print (f"Loaded {len(entries)} H1, C1, M1 entries from {json_path}")
    return entries



def get_funqa_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
    if visual_input.startswith("H"):
        subfolder = "val_humor" if "val" in root_folder else "test_humor"
    elif visual_input.startswith("M"):
        subfolder = "val_magic" if "val" in root_folder else "test_magic"
    elif visual_input.startswith("C"):
        subfolder = "val_creative" if "val" in root_folder else "test_creative"
    else:
        raise ValueError(f"Unknown prefix in visual_input: {visual_input}")
    return os.path.join(root_folder, subfolder, visual_input)


def get_funqa_amusing_frame_indices(annotation=None,
    task = "",
    transition=None,
    fps: int = 25, total_frames=None) -> List[int]:
    cleaned = annotation.replace("[", "").replace("]", "").strip()

    start, end = map(int, cleaned.replace("，", ",").split(","))
    amusing_ids = list(range(start, end + 1))
    if task == "C1":
        # Convert seconds to frame indices using fps
        fps = 30  # as per the authors
        start_index = int(start * fps)
        end_index = int(end * fps)
        amusing_ids = list(range(start_index, end_index + 1))
    return amusing_ids