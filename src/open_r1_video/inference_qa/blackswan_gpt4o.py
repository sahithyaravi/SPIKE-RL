import json
import pandas as pd
import requests
from openai import OpenAI
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
import random
from typing import List
import re
import base64
import os

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
from qwen_vl_utils.vision_process import smart_nframes
from open_r1_video.weighted_captioning import adaptive_frame_sampling_pdf
from open_r1_video.video_processing import extract_k_frames_decord_cpu
from open_r1_video.llm_match_reward import HuggingFaceLLMReward
from open_r1_video.utils import load_model_and_processor
from decord import VideoReader, cpu

client = OpenAI()

def extract_letter(text: str) :
    """Extracts the letter from the provided text."""
    match = re.search(r'\(([A-D])\)|\b([A-D])\b', text)
    return match.group(1) if match else None
    
def get_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=50,
    )
    return response

def encode_image_to_data_url(image, fmt="JPEG"):
    # Accept PIL.Image or NumPy array
    if not isinstance(image, Image.Image):
        # If using OpenCV BGR array, flip to RGB before this step:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    if fmt.upper() in {"JPG", "JPEG"} and image.mode != "RGB":
        image = image.convert("RGB")

    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() in {"JPG", "JPEG"} else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def video_eval_nframes(options, preevent_frames, event_frames, postevent_frames):

    prompt = (f"\nChoose which of the following options indicate what happened in the video frames shown here?\n"
                  f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}. \nProvide the correct option letter enclosed in ().")
 
    task2_prompt =  prompt
    frames = preevent_frames + event_frames + postevent_frames

    messages = [{"role": "user", "content": [{"type": "text", "text": task2_prompt}]}]
    messages.append({"role": "user", "content": [{"type": "text", "text": "Here are the frames of the video."}]})


    for frame in frames:
        # send as base64 encoded image
        base64_image = encode_image_to_data_url(frame)
        messages.append({"role": "user", "content": [{"type": "image_url", 
          "image_url": {
                        "url": base64_image # Adjust format if needed
                    }}]})

    response = get_response(messages)
    answer = response.choices[0].message.content
    return answer


def process_questions(dataset: List[dict], output_path: str, surprise_scoring=False, scores_data=[], model_path=None, pick_mode="linspace", num_frames=30, duration_based=False, temperature=0.7) -> None:
    """Processes questions from the dataset."""
    task1_correct = task1_failed = task1_total = task2_correct = task2_failed = task2_total = 0
    eval_ai_json = {}
    print("Pick mode:{}, Num frames: {}".format(pick_mode, num_frames))


    for item in tqdm(dataset):
        # preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        # event_frames = extract_frames(item['frames_url'] + f"{item['index']}_B_event")
        # postevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")
        try:
            vid_path = item['video_path']
            mcq_type = item["task"]
            mcq_type = 2 if mcq_type == "Reporter" else 1
            if mcq_type == 1:
                print("Skip Processing Task 1 question id:")
                continue
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
                    print("Duration based frame extraction for video of length {}s, {} frames, fps {}, extracted frames {}".format(total_frames/fps, total_frames, fps, num_frames))
                scores = scores_data_rec["surprise_scores"]    
                top_frames, sampled_idx = adaptive_frame_sampling_pdf(scores=scores, vr=vr, max_frames=num_frames, pick_mode=pick_mode, temperature=temperature)
                print("Extracted frames after adaptive sampling", sampled_idx, len(top_frames))
                preevent_frames = top_frames[:int(len(top_frames)/3)]
                event_frames = top_frames[int(len(top_frames)/3):int(2*len(top_frames)/3)]
                postevent_frames = top_frames[int(2*len(top_frames)/3):]
            else:
                top_frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu( video_path=vid_path, min_frames=num_frames) 
                print("Extracted frames without adaptive sampling", len(top_frames), frame_indices)
                preevent_frames = top_frames[:int(len(top_frames)/3)]
                event_frames = top_frames[int(len(top_frames)/3):int(2*len(top_frames)/3)]
                postevent_frames = top_frames[int(2*len(top_frames)/3):]
                
            options = item["mcq_options"]
            answer_index = item["mcq_gt_option"]


            if mcq_type == 2:
                model_answer = video_eval_nframes( options, preevent_frames, event_frames, postevent_frames)
                
            else:
                model_answer = video_eval_nframes( options, preevent_frames[:int(len(top_frames)*0.1)], postevent_frames[int(len(top_frames)*0.9):])
                model_answer = ""
               
            out = extract_letter(model_answer)
            print("Model answer:", model_answer, "Extracted letter:", out, "Correct answer index:", answer_index)
            correct = out and (out.startswith('A') and answer_index == 0 or 
                            out.startswith('B') and answer_index == 1 or 
                            out.startswith('C') and answer_index == 2)
            eval_ai_json[str(item["q_id"])] = 0 if out.startswith('A') else 1 if out.startswith('B') else 2 if out.startswith('C') else -1
            
            if mcq_type == 2:
                task2_correct += correct
                task2_failed += not correct
                task2_total += 1
            else:
                task1_correct += correct
                task1_failed += not correct
                task1_total += 1

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

    task1_accuracy = task1_correct / task1_total if task1_total else 0
    task2_accuracy = task2_correct / task2_total if task2_total else 0
    total_accuracy = total_correct / total_total if total_total else 0

    print(f"Task 1 - Accuracy: {task1_accuracy}")
    print(f"Task 2 - Accuracy: {task2_accuracy}")
    print(f"Total Accuracy: {total_accuracy}")


    # dump eval_ai_json
    with open(output_path.replace(".json", "_evalai.json"), 'w') as f:
        json.dump(eval_ai_json, f, indent=4)

    # dump dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    

    # dump accuracies
    with open(output_path.replace(".json", "_accuracies.txt"), 'w') as f:
        f.write(f"Task 1 - Accuracy: {task1_accuracy}\n")
        f.write(f"Task 2 - Accuracy: {task2_accuracy}\n")
        f.write(f"Total Accuracy: {total_accuracy}\n")


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


    with open(args.dataset_path, 'r') as f:
        mcq_list = json.load(f)

 
    scores_data = []
    if scores_path is not None and os.path.isfile(scores_path):
        with open(scores_path, "r") as f:
            scores_data = json.load(f)

    process_questions(mcq_list, args.save_path, surprise_scoring=args.surprise_sampling, scores_data=scores_data, model_path=args.model_path, pick_mode=args.pick_mode, num_frames=args.num_frames, duration_based=args.duration_based, temperature=args.temperature)
    print(f"Results saved to {args.save_path}")