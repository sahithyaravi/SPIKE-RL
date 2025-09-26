from datasets import load_dataset

# 
# print(ds["test"][0])

import json
import os
import re
from typing import List, Optional
from pathlib import Path
import argparse

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from open_r1_video.weighted_captioning import adaptive_frame_sampling_pdf
from open_r1_video.video_processing import extract_k_frames_decord_cpu, extract_k_frames_decord_cpu_duration_based
from qwen_vl_utils.vision_process import smart_nframes
from open_r1_video.utils import load_model_and_processor

from decord import VideoReader, cpu



def extract_letter(text: str) -> Optional[str]:
    """Extracts the letter from the provided text."""
    match = re.search(r'\(([A-D])\)|\b([A-D])\b', text)
    return match.group(1) if match else None


def video_eval_nframes(model, processor, question_text: str, options: List[str], frames) -> str:

    prompt = (f"""{question_text}\n{options[0]}\n{options[1]}\n{options[2]}

Provide the correct option letter enclosed in ()""")

    # print("Prompt:", prompt)
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": frames, "nframes": len(frames)},
                    {"type": "text", "text": prompt}],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], videos=frames, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if isinstance(output_text, list) else output_text

def process_questions(dataset: List[dict], output_path: str, surprise_scoring=False, scores_data=[], model_path=None, pick_mode="linspace", num_frames=30, duration_based=False, temperature=0.7) -> None:
  
    """Processes questions from the dataset."""
    model, processor = load_model_and_processor(model_path=model_path)
    task1_correct = task1_failed = task1_total = task2_correct = task2_failed = task2_total = 0
    print("Surprise scoring:", surprise_scoring)
    print("pick_mode:", pick_mode)
    print("num_frames:", num_frames)
    print("duration_based:", duration_based)
    print("temperature:", temperature)
    # if output_path exists, load it
    if os.path.isfile(output_path):
        with open(output_path, 'r') as f:
            result = json.load(f)
        # filter the entries with "correct" key
        dataset = [item for item in result if "correct" not in item]
        # count correct and total in result
        task2_correct += sum(1 for item in result if item.get("correct") == True)
        task2_total += sum(1 for item in result if "correct" in item)

        print(f"Loaded existing output file with {len(dataset)} entries.")
        print ("Resuming from existing file, skipping {} entries".format(len(result) - len(dataset)))

    for item in tqdm(dataset):
       
        try:
            vid_path = item["video_path"]
            if "/home/sahiravi/scratch/"+ vid_path in scores_data:
                scores_data_rec = scores_data.get("/home/sahiravi/scratch/"+ vid_path)
            elif vid_path in scores_data:
                scores_data_rec = scores_data.get(vid_path)
            else:
                scores_data_rec = None
                raise ValueError(f"Video path {vid_path} not found in scores data.")
            if surprise_scoring:
                vr = VideoReader(vid_path, ctx=cpu(0))
                total_frames = len(vr)
                fps = vr.get_avg_fps()
    
                if duration_based:
                    num_frames = smart_nframes({}, total_frames, fps)
                    # print("Duration based frame extraction for video of length {}s, {} frames, fps {}, extracted frames {}".format(total_frames/fps, total_frames, fps, num_frames))
                scores = scores_data_rec["surprise_scores"]    
                top_frames, sampled_idx = adaptive_frame_sampling_pdf(scores=scores, vr=vr, max_frames=num_frames, pick_mode=pick_mode, temperature=temperature)
                # print("Extracted frames after adaptive sampling", sampled_idx, len(top_frames))
  
            else:
                if duration_based:
                    top_frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu_duration_based(video_path=vid_path)
                else:
                    top_frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu( video_path=vid_path, min_frames=num_frames) 
                # print("Extracted frames without adaptive sampling", len(top_frames), frame_indices)

            options = ""
            answer = item["answer"]

            question = item["question"]
            options   = item["options"]

            options = [option.replace("A. ", "(A) ").replace("B. ", "(B) ").replace("C. ", "(C) ").replace("D. ", "(D) ") for option in options]
            model_answer = video_eval_nframes(model, processor, question, options, frames=top_frames)
                
            out = extract_letter(model_answer)
            # print("Model answer:", model_answer, "Extracted letter:", out, "Correct answer:", answer)
            correct = out in answer
            
 
            task2_correct += correct
            task2_failed += not correct
            task2_total += 1


            # dump model answer
            item['model_answer'] = model_answer
            item['model_answer_extracted'] = out
            item['correct'] = correct
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=4)
        except Exception as e:
            print(f"Error processing item with video path {item['video_path']}: {e}")
            continue

    # print(f"Task 1 - Accuracy: {task1_correct / task1_total if task1_total else 0}")
    accuracy = task2_correct / task2_total if task2_total else 0
    print(f"Accuracy: {accuracy}")
 
     # dump dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    with open(output_path.replace(".json", "_accuracies.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/h/sahiravi/VAR/data/mcq_list_all_gpt.json")
    parser.add_argument("--save_path", type=str, default="qwen2.json", help="Path to the output file")
    parser.add_argument("--surprise_sampling", action='store_true', help="Whether to use surprise sampling")
    parser.add_argument("--scores_path", type=str, default=None, help="Enable surprise scoring + adaptive sampling, provide path.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--pick_mode", type=str, default="linspace", help="Pick mode for adaptive sampling")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to extract from each segment")
    parser.add_argument("--duration_based", action='store_true', help="Whether to use duration based frame extraction")
    parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature for adaptive sampling")


    args = parser.parse_args()
    scores_path = args.scores_path

    print(args.duration_based)


    with open(args.dataset_path, 'r') as f:
        mcq_list = json.load(f)
    
    scores_data = []
    if scores_path is not None and os.path.isfile(scores_path):
        with open(scores_path, "r") as f:
            scores_data = json.load(f)

    if scores_data:
        process_questions(mcq_list, args.save_path, surprise_scoring=args.surprise_sampling, scores_data=scores_data, model_path=args.model_path, pick_mode=args.pick_mode, num_frames=args.num_frames, duration_based=args.duration_based, temperature=args.temperature)
    print(f"Results saved to {args.save_path}")