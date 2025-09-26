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
from open_r1_video.video_processing import extract_k_frames_decord_cpu
# from open_r1_video.llm_match_reward import HuggingFaceLLMReward
from open_r1_video.utils import load_model_and_processor
# from open_r1_video.inference.data_utils import process_json_and_create_data
from decord import VideoReader, cpu

# def get_activitynet_video_path(root_folder: str, set_id=None, index=None, visual_input=None) -> str:
#     return os.path.join(root_folder, "Activity_Videos", visual_input)


def extract_letter(text: str) -> Optional[str]:
    """Extracts the letter from the provided text."""
    match = re.search(r'\(([A-D])\)|\b([A-D])\b', text)
    return match.group(1) if match else None

def extract_frames(frame_path: str, num_frames: int = 10) -> List[Image.Image]:
    """Extract frames from a given path."""
    frames = [f"{frame_path}/frame_{i}.jpg" for i in range(1, num_frames + 1)]
    images = []
    for frame in frames:
        response = requests.get(frame)
        image = Image.open(BytesIO(response.content))
        images.append(image)
    return images

def video_eval_nframes(model, processor, question_text: str, options: List[str], frames1: List[Image.Image], 
                       frames2: List[Image.Image], frames3: Optional[List[Image.Image]] = None) -> str:
    """Evaluates the video frames using the model and returns the model's answer."""
   
    prompt = (f"\nAnswer the following question with 'yes' or 'no'. {question_text}.")
    frames = frames1 + frames2 + frames3


    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": frames, "nframes": len(frames)},
                    {"type": "text", "text": prompt}],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], videos=frames, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if isinstance(output_text, list) else output_text

def process_questions(dataset: List[dict], output_path: str, surprise_scoring=False, scores_data=[], model_path=None) -> None:
    """Processes questions from the dataset."""
    model, processor = load_model_and_processor(model_path=model_path)
    task1_correct = task1_failed = task1_total = task2_correct = task2_failed = task2_total = 0

    for item in tqdm(dataset):
        # preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        # event_frames = extract_frames(item['frames_url'] + f"{item['index']}_B_event")
        # postevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")
       
        try:
            vid_path = item["video_path"]

            if surprise_scoring and scores_data:
                if "/home/sahiravi/scratch/"+ vid_path in scores_data:
                    scores_data_rec = scores_data.get("/home/sahiravi/scratch/"+ vid_path)
                elif vid_path in scores_data:
                    scores_data_rec = scores_data.get(vid_path)
                else:
                    scores_data_rec = None
                    raise ValueError(f"Video path {vid_path} not found in scores data.")
                
                vr = VideoReader(vid_path, ctx=cpu(0))
                scores = scores_data_rec["surprise_scores"]    
                top_frames, sampled_idx = adaptive_frame_sampling_pdf(scores=scores, vr=vr, max_frames=30, pick_mode="linspace")
                print("Extracted frames after adaptive sampling", sampled_idx, len(top_frames))
                preevent_frames = top_frames[:10]
                event_frames = top_frames[10:20]
                postevent_frames = top_frames[20:25]

            else:
                frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu( video_path=vid_path, min_frames=30) 
                print("Extracted frames without adaptive sampling", len(frames))
                preevent_frames = frames[:10]
                event_frames = frames[10:20]
                postevent_frames = frames[20:30]
                
            options = ""
            answer = item["answer"]
            
            model_answer = video_eval_nframes(model, processor, "", options, preevent_frames, event_frames, postevent_frames)
                
            out = model_answer
            print("Model answer:", model_answer, "Extracted letter:", out, "Correct answer:", answer)
            correct = out == answer
            
 
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

    total_correct = task1_correct + task2_correct
    total_failed = task1_failed + task2_failed
    total_total = task1_total + task2_total

    print(f"Task 1 - Accuracy: {task1_correct / task1_total if task1_total else 0}")
    print(f"Task 2 - Accuracy: {task2_correct / task2_total if task2_total else 0}")
    print(f"Total Accuracy: {total_correct / total_total if total_total else 0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/h/sahiravi/VAR/data/mcq_list_all_gpt.json")
    parser.add_argument("--save_path", type=str, default="qwen2.json", help="Path to the output file")
    parser.add_argument("--surprise_sampling", type=bool, default=False, help="Whether to use surprise sampling")
    parser.add_argument("--scores_path", type=str, default=None, help="Enable surprise scoring + adaptive sampling, provide path.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")


    args = parser.parse_args()
    scores_path = args.scores_path
    with open(args.dataset_path, 'r') as f:
        mcq_list = json.load(f)
    scores_data = []
    if scores_path is not None and os.path.isfile(scores_path):
        with open(scores_path, "r") as f:
            scores_data = json.load(f)

    process_questions(mcq_list, args.save_path, surprise_scoring=args.surprise_sampling, scores_data=scores_data, model_path=args.model_path)
    print(f"Results saved to {args.save_path}")