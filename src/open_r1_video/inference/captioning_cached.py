
#!/usr/bin/env python3
"""
Generic captioning inference pipeline for heterogeneous JSONL datasets.

Example:
    python src/open_r1_video/inference_captions/captioning.py \
        --data data/ActivityNet_Captions/activitynet_captions_val1.json\
        --config src/open_r1_video/inference_captions/activitynet.yaml \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --output out/activity_val1_caps.jsonl \
        --enable_surprise 

"""

import argparse
import json
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import yaml

from typing import Any, Dict, List, Optional, Tuple
import torch

from datasets import Dataset, DatasetDict
import json
from data_utils import process_json_and_create_data
from decord import VideoReader, cpu
# Import the *same* helpers your trainer uses
from open_r1_video.weighted_captioning import adaptive_frame_sampling_pdf
from open_r1_video.video_processing import extract_k_frames_decord_cpu
from open_r1_video.llm_match_reward import HuggingFaceLLMReward
from open_r1_video.utils import load_model_and_processor

def caption_one_video(
    model,
    processor,
    video_path: str,
    prompt=None,
    window_size: int = 4,
    top_k: int = 3,
    surprise_scoring: str = 'uniform',
    max_new_tokens: int = 80,
    max_frames=None,
    scores_data_rec=None
) -> Dict[str, Any]:
    """
    Runs the exact same pipeline as your trainer loop, but once, for inference.
    Returns caption + debug info.
    """
    # 1) Load frames (same helper as training)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    # print(f"Extracted {len(frame_indices)} frames from {total_frames} total frames at {fps:.2f} fps for surprise scoring", file=sys.stderr)
    # 3) Adaptive frame sampling (same helper as training)
    if max_frames is None:
        max_frames = 32
    
    pick_mode ="weighted"

        
    # 2) Surprise tracking (optional, same helper as training)
    if surprise_scoring != "uniform":
        scores = scores_data_rec["surprise_scores"]
        top_frames, sampled_idx = adaptive_frame_sampling_pdf(scores=scores, vr=vr, max_frames=max_frames, pick_mode=pick_mode, temperature=0.7)
        print(f"Extracted frames after surprise sampling with max_frames {max_frames}", sampled_idx)
            
    else:
        scores = [1.0] * max(1, max_frames)
        top_frames, sampled_idx = adaptive_frame_sampling_pdf(scores=scores, vr=vr, max_frames=max_frames, pick_mode=pick_mode)
        print(f"Extracted frames after uniform sampling with max_frames {max_frames}", sampled_idx)

    # 4) Captioning prompt/template (identical to training)
    cap_conv = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},
        ],
    }]
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)

    with torch.no_grad():
        gen_inputs = processor(text=cap_prompt, videos=top_frames, return_tensors="pt").to(model.device)
        cap_ids = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
        
        )
    caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].lower().split("assistant\n")[1].replace(":", "")
 
    return {
        "caption": caption,
        "sampled_frame_indices": sampled_idx,
        "surprise_scores": scores,
        "fps": fps,
        "total_frames": total_frames,
    }


def llm_match_with_reward_fn(pred: str, ref: str, reward_fn) -> float:
    """
    Reuse your training-time reward function as the LLM-match metric.

    Expects a callable compatible with how you use it in compute_loss:
        reward_fn(responses=[[pred]], ground_truths=[str(ref)]) -> tensor/float

    Returns a float.
    """
    val = reward_fn(responses=[[pred]], ground_truths=[str(ref)])
    # robustly convert to float
    try:
        import torch
        return float(torch.as_tensor(val, dtype=torch.float32).view(-1)[0].item())
    except Exception:
        try:
            return float(val)
        except Exception:
            # last-resort: handle list/tuple
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return float(val[0])
            raise TypeError("Unsupported return type from reward_fn; please return a scalar or 1-element tensor/list.")





def run_pipeline(
    data_path: str,
    video_root: str,
    model_path: str,
    output_path: str,
    max_frames=None,
    surprise_scoring: str = "uniform",
    temperature: float = 0.1,
    scores_path: Optional[str] = None,
    prompt=None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dataset = process_json_and_create_data(data_path, video_root)
    scores_data_rec = None

    print("Prompt:", prompt)
    # get scores if provided
    if scores_path is not None and os.path.isfile(scores_path):
        with open(scores_path, "r") as f:
            scores_data = json.load(f)
    model, processor  = load_model_and_processor(model_path)
    reward_fn = HuggingFaceLLMReward()
    results = []

    total_llm_match = 0.0
    total = 0.0
    for idx, rec in enumerate(dataset):
        vid_path = rec["video_path"]
        ref = rec["caption"] if "caption" in rec else rec["explanation"]
        if ref is None:
            print(f"Warning: No reference caption found for {vid_path}, skipping")
            continue

        print(f"[{idx+1}] Processing - {vid_path}", file=sys.stderr)

        if scores_path is not None and os.path.isfile(scores_path):
            if "/home/sahiravi/scratch/"+ vid_path in scores_data:
                scores_data_rec = scores_data.get("/home/sahiravi/scratch/"+ vid_path)
            elif vid_path in scores_data:
                scores_data_rec = scores_data.get(vid_path)
            else:
                scores_data_rec = None
                

        if scores_data_rec is None:
            print(f"Warning: No surprise scores found for {vid_path} in {scores_path}, continue...", file=sys.stderr)
            continue

        pred = caption_one_video(
            model=model,
            processor=processor,
            video_path=vid_path,
            prompt=prompt,
            window_size=4,
            top_k=3,
            surprise_scoring=surprise_scoring,
            max_frames=max_frames,
            scores_data_rec=scores_data_rec if scores_path is not None else None,
        )
        out_rec: Dict[str, Any] = {
            "video": vid_path,
        }

        caption_pred = pred["caption"]
        rewards_tensor = reward_fn(responses=[[caption_pred]], ground_truths=[str(ref)])
        r_scalar = torch.as_tensor(rewards_tensor, dtype=torch.float32).view(-1)[0]
        # convert r_scalar to scalar float
        llm_match = str(float(r_scalar.item()))
        print(caption_pred, llm_match)
        if ref is not None:
            out_rec["reference"] = ref
        if llm_match is not None:
            out_rec["llm_match"] = llm_match
            total_llm_match += float(llm_match)
            total += 1.0
        
        out_rec["prediction"] = caption_pred

        results.append(out_rec)

        # stream to file progressively
        with open(output_path, "a") as w:
            w.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    avg_llm_match = total_llm_match / len(dataset)
    print(f"Average LLM-Match : {avg_llm_match:.4f}")
    print(f"Wrote {len(results)} records to {output_path}")

    # Dump average score to a separate file
    with open(output_path + ".summary.txt", "w") as w:
        w.write(f"Average LLM-Match: {avg_llm_match:.4f}\n")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Captioning inference with surprise-based frame sampling.")
    p.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    p.add_argument("--model_path", required=True, help="Path or name for your captioning model.")
    p.add_argument("--output", required=True, help="Where to save predictions (JSONL).")
    p.add_argument("--max_frames", type=int, default=None, help="Max frames after sampling.")
    p.add_argument("--surprise_scoring", type=str, default="uniform", help="Enable surprise scoring + adaptive sampling.", choices=["uniform", "prior_frame_bayesian_approach", "direct"])
    p.add_argument("--scores_path", type=str, default=None, help="Enable surprise scoring + adaptive sampling, provide path.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for caption generation.")
    p.add_argument("--video_root", type=str, required=True, help="Path to the root folder of videos.",)
    p.add_argument("--prompt", type=str, default="Explain the key event in this video in 1-2 sentences.", help="Custom prompt for captioning.")
    return p.parse_args()

def main():
    args = parse_args()
    # Load dataset

    # Run inference
    run_pipeline(
        data_path=args.json_path,
        video_root=args.video_root,
        model_path=args.model_path,
        output_path=args.output,
        max_frames=args.max_frames,
        surprise_scoring=args.surprise_scoring,
        temperature=args.temperature,
        scores_path=args.scores_path,
        prompt=args.prompt,
    )

if __name__ == "__main__":
    main()
