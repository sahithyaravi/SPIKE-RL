import json
import os
from metrics import precision_recall_hit_at_k, accuracy_at_delta_t, iou_temporal, compute_iou_coverage
from hypothesis_metrics import compute_hyp_metrics_from_record
from video_processing import extract_k_frames_decord_cpu
import numpy as np
from tqdm import tqdm

import math
from typing import Iterable, List, Tuple, Optional

def frames_from_loud_windows(
    volume_curve: Iterable[Tuple[float, float, float]],
    threshold: float,
    fps: float,
    *,
    unique: bool = True,
    max_frames: Optional[int] = None,
) -> List[int]:
    """
    Convert 'loud' time windows to frame indices.

    Args:
        volume_curve: iterable of (start_s, end_s, volume) triples.
                      Times are in seconds; volume can be any scale where ">" means louder
                      (e.g., dBFS: -24 > -30).
        threshold: keep windows where volume > threshold.
        fps: frames per second used to convert seconds -> frame indices.
        unique: if True, deduplicate and sort the final indices.
        max_frames: optional clamp for the maximum valid frame index (exclusive).
                    If provided, indices >= max_frames are discarded.

    Returns:
        List of frame indices across all windows where volume > threshold.
        If unique=True, this list is sorted and deduplicated.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    indices: List[int] = []

    for start_s, end_s, vol in volume_curve:
        if vol <= threshold:
            continue
        if end_s <= start_s:
            continue

        # Map [start_s, end_s) to integer frame indices.
        start_idx = max(0, math.floor(start_s * fps))
        # Use ceil(end*fps)-1 so that [start,end) covers all frames whose timestamp < end_s
        end_idx = math.ceil(end_s * fps) - 1

        if max_frames is not None:
            end_idx = min(end_idx, max_frames - 1)

        if end_idx >= start_idx:
            # Extend with the contiguous integer range
            indices.extend(range(start_idx, end_idx + 1))

    if unique:
        # Deduplicate and sort
        indices = sorted(set(indices))

    return indices


def get_amusing_indices(video_path, total_frames):
    annotations_json = "/home/sahiravi/projects/aip-vshwartz/sahiravi/Video-BeliefPO/data/bean/final_clips_metadata_with_volume.json"
    with open(annotations_json, "r") as f:
        annotations = json.load(f)
    
    dataset = annotations["scenes"]
    # find item with matching video_path name 'scene_file' is the name
    entry = None
    for item in dataset:
        if item["scene_file"] in video_path:
            entry = item
            break
    if entry is None:
        return None
    volume_curve = entry["volume_curve"]
    volume_max_db = entry["volume_peak_3s"]["max_volume_db"]
   
    threshold = 1.2 * volume_max_db
    print(f"Volume max dB for {video_path}: {volume_max_db}, threshold: {threshold}")
    fps = 25.0
    amusing_indices = frames_from_loud_windows(volume_curve, threshold=threshold, fps=fps, max_frames=total_frames)
    print(f"Amusing indices for {video_path}: {amusing_indices}")
    return amusing_indices
    
def load_jsonl(jsonl_path: str):
    data_dict = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            video_path = entry.pop("video_path")  # remove and keep the key
            data_dict[video_path] = entry
    return data_dict


def read_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

root = "/home/sahiravi/projects/aip-vshwartz/sahiravi/results_v2/bean/"
all_folders = os.listdir(root)
print(all_folders)
for folder in all_folders:
    result_folder = root + folder
    print(f"Processing folder: {result_folder}")

    # if folder contains results_final.json, read_json
    if os.path.exists(os.path.join(result_folder, "results_final.json")):
        data = read_json(os.path.join(result_folder, "results_final.json"))
        data = list(data.values())
    elif os.path.exists(os.path.join(result_folder, "results.json")):
        data = load_jsonl(os.path.join(result_folder, "results.json"))
        data = list(data.values())


    results_with_scores = {}
    hypo_enabled = False #å if "prior_frame" in result_folder else False

    for idx, output in tqdm(enumerate(data)):
        video_path = output["video_path"] if "video_path" in output else output["output_video_path"]
        video, indices, total_frames, fps, vr = extract_k_frames_decord_cpu(video_path)
        amusing_sampled = get_amusing_indices(video_path, total_frames)
        if amusing_sampled is None or len(amusing_sampled) == 0:
            amusing_sampled = output["amusing_indices"] if "amusing_indices" in output else []
        frame_indices = output["frame_indices"]
        surprise_scores = output["surprise_scores"]

        # print(len(surprise_scores), len(frame_indices), len(amusing_sampled))
        explanations = output["Explanations"] if "Explanations" in output else None
        precision, recall, hit = precision_recall_hit_at_k(output["surprise_scores"], frame_indices, amusing_sampled, k=5)
        accuracy_at_delta_0_25 = accuracy_at_delta_t(output["surprise_scores"], frame_indices, amusing_sampled, delta_t=0.25, fps=fps)
        accuracy_at_delta_1 = accuracy_at_delta_t(output["surprise_scores"], frame_indices, amusing_sampled, delta_t=1, fps=fps)
        iou_score = iou_temporal(output["surprise_scores"], frame_indices, amusing_sampled, fps=fps)
        contigous_iou = compute_iou_coverage(output["surprise_scores"], frame_indices, amusing_sampled, fps=fps)
        results_with_scores[video_path] = output
        if explanations is not None and hypo_enabled:
            hypothesis_metrics = compute_hyp_metrics_from_record(
                    record=output,
                )
            results_with_scores[video_path]["hypothesis_plausibility"] =  hypothesis_metrics["plausibility"]
            results_with_scores[video_path]["hypothesis_diversity"] =  hypothesis_metrics["diversity"]
            results_with_scores[video_path]["hypothesis_quality"] =  hypothesis_metrics["bsq"]

        
        results_with_scores[video_path]["precision_at_k"] = precision
        results_with_scores[video_path]["recall_at_k"] = recall
        results_with_scores[video_path]["hit_at_1"] = hit
        results_with_scores[video_path]["accuracy_at_delta_0.25"] = accuracy_at_delta_0_25
        results_with_scores[video_path]["accuracy_at_delta_1"] = accuracy_at_delta_1
        results_with_scores[video_path]["iou"] = iou_score
        results_with_scores[video_path]["contiguous_iou"] = contigous_iou
        if hypo_enabled:
            results_with_scores[video_path].update(hypothesis_metrics)


    # save results_with_scores to jsonl
    # dump final results
    with open(os.path.join(result_folder, "results_final_scored.json"), "w") as f:
        json.dump(results_with_scores, f, indent=4)

    # Average metrics across all entries  
    avg_precision = np.mean([o["precision_at_k"] for k, o in results_with_scores.items()])
    avg_hit = np.mean([o["hit_at_1"] for k, o in results_with_scores.items()])
    avg_recall = np.mean([o["recall_at_k"] for k, o in results_with_scores.items()])
    avg_accuracy_at_delta_0_25 = np.mean([o["accuracy_at_delta_0.25"] for k, o in results_with_scores.items()])
    avg_accuracy_at_delta_1 = np.mean([o["accuracy_at_delta_1"] for k, o in results_with_scores.items()])
    avg_iou = np.mean([o["iou"] for k, o in results_with_scores.items() if o["iou"] is not None])
    avg_contiguous_iou = np.mean([o["contiguous_iou"] for k, o in results_with_scores.items() if o["contiguous_iou"] is not None])
    print(f"Average AP: {avg_precision}")
    print(f"Average Hit@k: {avg_hit}")
    print(f"Average Recall@k: {avg_recall}")
    print(f"Average Accuracy@delta_0.25: {avg_accuracy_at_delta_0_25}")
    print(f"Average Accuracy@delta_1: {avg_accuracy_at_delta_1}")
    print(f"Average IoU Peak: {avg_iou}")
    print(f"Average Contiguous IoU: {avg_contiguous_iou}")

    # Save averaged metrics to JSON
    avg_metrics = {
        "average_precision_at_k": avg_precision,
        "average_hit_at_1": avg_hit,
        "average_recall_at_k": avg_recall,
        "average_accuracy_at_delta_0.25": avg_accuracy_at_delta_0_25,
        "average_accuracy_at_delta_1": avg_accuracy_at_delta_1,
        "average_iou_peak": avg_iou,
        "average_contiguous_iou": avg_contiguous_iou,
    }
    with open(os.path.join(result_folder, "average_metrics_final.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(
        f"Average metrics saved to {os.path.join(result_folder, 'average_metrics_final.json')}"
    )



    # Average hypothesis metrics
    if hypo_enabled:
        avg_plausibility = np.mean([o["plausibility"] for k, o in results_with_scores.items() if o["plausibility"] is not None])
        avg_diversity = np.mean([o["diversity"] for k, o in results_with_scores.items() if o["diversity"] is not None])
        avg_bsq = np.mean([o["bsq"] for k, o in results_with_scores.items() if o["bsq"] is not None])
        hypothesis_metrics = {
            "average_hypothesis_plausibility": avg_plausibility,
            "average_hypothesis_diversity": avg_diversity,
            "average_hypothesis_bsq": avg_bsq,
        }
        with open(os.path.join(result_folder, "average_hypothesis_metrics.json"), "w") as f:
            json.dump(hypothesis_metrics, f, indent=4)
        print(
            f"Average hypothesis metrics saved to {os.path.join(result_folder, 'average_hypothesis_metrics.json')}"
        )
