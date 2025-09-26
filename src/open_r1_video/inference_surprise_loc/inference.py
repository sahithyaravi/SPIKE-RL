import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from visualization_utils import (
    make_scrollbar_video,
)  

from open_r1_video.video_processing import extract_k_frames_decord_cpu
from open_r1_video.hypothesis_metrics import compute_hyp_metrics_from_record
from open_r1_video.metrics import precision_recall_hit_at_k, accuracy_at_delta_t, iou_temporal, compute_iou_coverage

import random
import argparse
import matplotlib.pyplot as plt
import math
import torch


from funqa import load_funqa_json, get_funqa_video_path, get_funqa_amusing_frame_indices
from oops import load_oops_json, get_oops_video_path, get_oops_amusing_frame_indices
from bean import load_bean_json, get_bean_video_path, get_bean_amusing_frame_indices
from activitynet import load_activitynet_json, get_activitynet_video_path, get_activitynet_amusing_frame_indices, load_activitynetqa_json, get_activitynetqa_video_path, get_activitynetqa_amusing_frame_indices
from nextqa import load_nextqa_json, get_nextqa_video_path, get_nextqa_amusing_frame_indices
from exfuntube import load_exfuntube_json, get_exfuntube_video_path, get_exfuntube_amusing_frame_indices
from tempcompass import load_tempcompass_json, get_tempcompass_video_path, get_tempcompass_amusing_frame_indices
from videomme import load_videomme_json, get_videomme_video_path, get_videomme_amusing_frame_indices

random.seed(42)  
np.random.seed(42)

def load_jsonl(file_path: str) -> Dict:
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            video_path = entry['video_path']
            data[video_path] = entry
    return data

def make_result_folder(result_folder: str):
    if not os.path.exists(result_folder):
        print(f"Creating result folder: {result_folder}")
        os.makedirs(result_folder, exist_ok=True)
    else:
        print(f"Result folder already exists: {result_folder}")

def evaluate(
    json_path: str,
    video_root: str,
    result_folder: str,
    method_function: str,
    num_sampled_frames=20,
    window_size=4,
    num_entries=1,
    topk_hyp=5,
    model=None
):
    """
    Evaluates all H1-type entries in the FunQA dataset using a surprise scoring method.
    Saves annotated videos with highlighted amusing frames and surprise score bars.
    """

    # recursively make result folder
    make_result_folder(result_folder)

    if "funqa" in json_path.lower():
        print("Loading FunQA dataset...")
        dataset = load_funqa_json(json_path)
        get_video_path = get_funqa_video_path
        get_amusing_frame_indices = get_funqa_amusing_frame_indices
        min_frames = None
    elif "bean" in json_path.lower():
        print("Loading BEAN dataset...")
        dataset = load_bean_json(json_path)
        get_video_path = get_bean_video_path
        get_amusing_frame_indices = get_bean_amusing_frame_indices
        min_frames = 25
    elif "activitynet" in json_path.lower() and "caption" in json_path.lower():
        print("Loading ActivityNet dataset...")
        dataset = load_activitynet_json(json_path)
        get_video_path = get_activitynet_video_path
        get_amusing_frame_indices = get_activitynet_amusing_frame_indices
        min_frames = None
    elif "exfuntube" in json_path.lower():
        print("Loading ExFunTube dataset...")
        dataset = load_exfuntube_json(json_path)
        get_video_path = get_exfuntube_video_path
        get_amusing_frame_indices = get_exfuntube_amusing_frame_indices
        min_frames = None  # ExFunTube videos can be long; limit to first 200 frames
    elif "tempcompass" in json_path.lower():
        print("Loading TempCompass dataset...")
        dataset = load_tempcompass_json(json_path)
        get_video_path = get_tempcompass_video_path
        get_amusing_frame_indices = get_tempcompass_amusing_frame_indices
        min_frames = None
    elif "mme" in json_path.lower():
        print("Loading VideoMME dataset...")
        dataset = load_videomme_json(json_path)
        get_video_path = get_videomme_video_path
        get_amusing_frame_indices = get_videomme_amusing_frame_indices
        min_frames = None
    elif "activitynetqa" in json_path.lower():
        print("Loading ActivityNet-QA dataset...")
        dataset = load_activitynetqa_json(json_path)
        get_video_path = get_activitynetqa_video_path
        get_amusing_frame_indices = get_activitynetqa_amusing_frame_indices
        min_frames = None 
    elif "nextqa" in json_path.lower():
        print("Loading NextQA dataset...")
        dataset = load_nextqa_json(json_path)
        get_video_path = get_nextqa_video_path
        get_amusing_frame_indices = get_nextqa_amusing_frame_indices
        min_frames = None
    else:
        print("Loading Oops dataset...")
        dataset = load_oops_json(json_path)
        get_video_path = get_oops_video_path
        get_amusing_frame_indices = get_oops_amusing_frame_indices
        min_frames = 32

    outputs = {}
    print("Model used:", model)

    # if result.json exists in result_folder, load it - it is json lines
    result_json_path = os.path.join(result_folder, "results.json")
    if os.path.exists(result_json_path):
        print(f"Loading existing results from {result_json_path}")
        existing_results = load_jsonl(result_json_path)
        outputs.update(existing_results)
        print(f"Loaded {len(existing_results)} existing results.")


    # sample num_entries from the dataset
    if num_entries is not None:
        dataset = random.sample(dataset, num_entries)
        print(f"Sampling {num_entries} entries from the dataset.")

    # based on method, we load models
    if method_function == "direct" or method_function == "prior_frame_bayesian_approach":
        from open_r1_video.belief_tracker import qwen_surprise_tracker
        from open_r1_video.naive_surprise_scorer import direct_surprise_tracker
        from open_r1_video.utils import load_model_and_processor
        model, processor = load_model_and_processor(model)
    if method_function == "flow":
        from flow_based_trackers import flow_scorer

    # collect percentage of amusing frames
    amusing_fractions = []

    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        set_id = entry.get("set_id", "unknown_set")
        index = entry.get("index") if "index" in entry else None
        visual_input = entry.get("visual_input", None)
        video_file = get_video_path(video_root, set_id, index, visual_input)
        print(f"Processing video {idx+1}/{len(dataset)}: {video_file}")
        if "960_E_merged.mp4" not in video_file:
            continue
        if not os.path.exists(video_file):
            print(f"Missing video: {video_file}")
            continue

        #try:
        if video_file in outputs:
            print(f"Already processed: {video_file}")
            continue

        frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu(
            video_path=video_file, min_frames=min_frames
        )
        print(f"Extracted {len(frames)} frames from video with total {total_frames} frames at {fps} fps.")
        annotations = entry["output"] if "output" in entry else entry
        transition = entry.get("transition", None)
        task = entry.get("task", "")
        amusing_ids = get_amusing_frame_indices(annotation=annotations, task=task, transition=transition, fps=fps, total_frames=total_frames)
        
        if not amusing_ids or amusing_ids is None:
            # select 10% of frame_indices as amusing
            amusing_sampled = random.sample(frame_indices.tolist(), int(len(frame_indices) * 0.2) + 1)
            print(f"No amusing frames found in annotations. Randomly selected {len(amusing_sampled)} amusing frames.")
        else:
            amusing_sampled = [
                int(fid) for fid in frame_indices if int(fid) in amusing_ids
            ]
        frac = len(amusing_sampled) / num_sampled_frames
        amusing_fractions.append(frac)
        # frac = len(amusing_ids) / total_frames
 
        if not (0.05 <= frac <= 0.75):
            continue

        # Run Surprise Scoring
        if method_function == "direct":
            output = direct_surprise_tracker(video_frames=frames, model=model, processor=processor)
        elif method_function in ["prior_frame_bayesian_approach", "prior_frame_bayesian_approach_vf"]:
            output = qwen_surprise_tracker(
                frames=frames, window_size=window_size, method=method_function, top_k=topk_hyp, vr=vr, model=model, processor=processor
            )
            # Replace None explanations with empty strings
            output["explanations"] = ["" if expl is None else expl for expl in output["explanations"]]
        elif method_function == "flow":
            output = flow_scorer(frames=frames, model=model, processor=processor)

        elif method_function == "random_baseline":
            # Randomly choose 5 unique indices from frame_indices
            selected_indices = random.sample(frame_indices.tolist(), 5)

            scores = [1.0 if i in selected_indices else 0.0 for i in range(len(frame_indices))]
            output = {
                "frames": frames,
                "surprise_scores": scores,
            }
        else:
            raise ValueError(f"Unknown method_function: {method_function}")

        # Replace None with 0.0
        scores_t = torch.tensor(output["surprise_scores"], dtype=torch.float32, device="cpu")
        scores_np = np.array([s.item() for s in scores_t], dtype=float)
        
        output["surprise_scores"] = [
            0.0 if score is None else score for score in scores_np
        ]



        precision, recall, hit = precision_recall_hit_at_k(
            output["surprise_scores"], frame_indices, amusing_sampled, k=5
        )
        accuracy_at_delta_0_25 = accuracy_at_delta_t(output["surprise_scores"], frame_indices, amusing_sampled, delta_t=0.25, fps=fps)
        accuracy_at_delta_1 = accuracy_at_delta_t(output["surprise_scores"], frame_indices, amusing_sampled, delta_t=1, fps=fps)
        iou_score = iou_temporal(output["surprise_scores"], frame_indices, amusing_sampled, fps=fps)
        contigous_iou = compute_iou_coverage(output["surprise_scores"], frame_indices, amusing_sampled, fps=fps)
        
        if method_function == "prior_frame_bayesian_approach":
            hypothesis_metrics = compute_hyp_metrics_from_record(
                record=output,
            )
        else:
            hypothesis_metrics = {
                "plausibility": None,
                "diversity": None,
                "bsq": None,
            }
        prec = round(precision, 4)
        recall = round(recall, 4)
        hit = round(hit, 4)
        accuracy_at_delta_0_25 = round(accuracy_at_delta_0_25, 4)
        accuracy_at_delta_1 = round(accuracy_at_delta_1, 4)
        iou_score = round(iou_score, 4)
        contigous_iou = round(contigous_iou, 4)
        output_video_path = f"{os.path.basename(video_file).replace('.mp4', '')}_p{prec}_r{recall}_output.mp4"

        if idx % 1 == 0 and method_function == "prior_frame_bayesian_approach":
            make_scrollbar_video(
                frames=output["frames"],
                scores=output["surprise_scores"],
                explanations=output["explanations"],
                out=os.path.join(
                    result_folder,
                    output_video_path,
                ),
                amusing_indices=amusing_sampled,
            )
        outputs[video_file] = {
            "video_path": video_file,
            "amusing_ids": amusing_sampled,
            "surprise_scores": output["surprise_scores"],
            "Explanations": output["explanations"] if "explanations" in output else None,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "hit_at_1": hit,
            "accuracy_at_delta_0.25": accuracy_at_delta_0_25,
            "accuracy_at_delta_1": accuracy_at_delta_1,
            "iou_peak": iou_score,
            "contiguous_iou": contigous_iou,
            "frame_indices": frame_indices.tolist(),
            "memory_evolution": output["memory_evolution"] if "memory_evolution" in output else None,
            "caption_weighted": output["caption_weighted"] if "caption_weighted" in output else None,
            "caption_unweighted": output["caption_unweighted"] if "caption_unweighted" in output else None,
            "output_video_path": f"{result_folder}/{output_video_path}",
            "sampled_frames_weighted": output["sampled_frames_weighted"] if "sampled_frames_weighted" in output else None,
            "sampled_frames_unweighted": output["sampled_frames_unweighted"] if "sampled_frames_unweighted" in output else None,
            "hypothesis_plausibility": hypothesis_metrics["plausibility"],
            "hypothesis_diversity": hypothesis_metrics["diversity"],
            "hypothesis_quality": hypothesis_metrics["bsq"],
        }

        # Convert the single dict to a one-row DataFrame
        row_df = pd.DataFrame([outputs[video_file]])
        result_json_path = os.path.join(result_folder, "results.json")

        # Append to JSON file (newline-delimited JSON)
        if os.path.exists(result_json_path):
            row_df.to_json(result_json_path, orient="records", lines=True, mode="a")
        else:
            row_df.to_json(result_json_path, orient="records", lines=True)

        print(f"Appended results to {result_json_path}")
        break


    # dump final results
    with open(os.path.join(result_folder, "results_final.json"), "w") as f:
        json.dump(outputs, f, indent=4)

    
    # Average metrics across all entries  
    avg_precision = np.mean([o["precision_at_k"] for k, o in outputs.items()])
    avg_hit = np.mean([o["hit_at_1"] for k, o in outputs.items()])
    avg_recall = np.mean([o["recall_at_k"] for k, o in outputs.items()])
    avg_accuracy_at_delta_0_25 = np.mean([o["accuracy_at_delta_0.25"] for k, o in outputs.items()])
    avg_accuracy_at_delta_1 = np.mean([o["accuracy_at_delta_1"] for k, o in outputs.items()])
    avg_iou = np.mean([o["iou_peak"] for k, o in outputs.items() if o["iou_peak"] is not None])
    avg_contiguous_iou = np.mean([o["contiguous_iou"] for k, o in outputs.items() if o["contiguous_iou"] is not None])


    print(f"Average AP: {avg_precision}")
    print(f"Average Hit@k: {avg_hit}")
    print(f"Average Recall@k: {avg_recall}")
    print(f"Average Accuracy@delta_0.25: {avg_accuracy_at_delta_0_25}")
    print(f"Average Accuracy@delta_1: {avg_accuracy_at_delta_1}")
    print(f"Average IoU: {avg_iou}")
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
    with open(os.path.join(result_folder, "average_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(
        f"Average metrics saved to {os.path.join(result_folder, 'average_metrics.json')}"
    )

    # Average hypothesis metrics
    if method_function == "prior_frame_bayesian_approach":
        avg_plausibility = np.mean([o["hypothesis_plausibility"] for k, o in outputs.items() if o["hypothesis_plausibility"] is not None])
        avg_diversity = np.mean([o["hypothesis_diversity"] for k, o in outputs.items() if o["hypothesis_diversity"] is not None])
        avg_bsq = np.mean([o["hypothesis_quality"] for k, o in outputs.items() if o["hypothesis_quality"] is not None])
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

    # Plot distribution of amusing frame fractions
    plt.figure(figsize=(10, 4))
    plt.hist(amusing_fractions, bins=20, color="cornflowerblue", edgecolor="black")
    plt.title("Fraction of Amusing Frames per Video")
    plt.xlabel("Amusing Fraction")
    plt.ylabel("Number of Videos")
    plt.grid(True)
    plt.savefig(os.path.join(result_folder, "amusing_hist.png"))

if __name__ == "__main__":
    print("Starting oops evaluation...")
    parser = argparse.ArgumentParser(description="Run FunQA evaluation methods.")

    parser.add_argument(
        "--json_path", type=str, required=True, help="Path to the FunQA JSON file."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Path to the root folder of videos.",
    )
    parser.add_argument(
        "--result_folder", type=str, required=True, help="Folder to save results."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "prior_frame_bayesian_approach",
            "prior_frame_bayesian_approach_vf",
            "random_baseline",
            "prior_frame_approach",
            "direct",
        ],
        required=True,
        help="Evaluation method to use.",
    )
    parser.add_argument(
        "--num_entries",
        type=int,
        default=None,
        help="Number of video entries to evaluate.",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--topk_hyp",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="(Random baseline only) Number of random trials.",
    )

    parser.add_argument(
        "--model",
        type=str,
        
    )
    args = parser.parse_args()

    print(f"Running evaluation with method: {args.method}")
    evaluate(
        json_path=args.json_path,
        video_root=args.video_root,
        result_folder=args.result_folder,
        method_function=args.method,
        num_entries=args.num_entries,
        num_sampled_frames=args.num_sampled_frames,
        topk_hyp=args.topk_hyp,
        window_size=2,
        model=args.model,
    )
