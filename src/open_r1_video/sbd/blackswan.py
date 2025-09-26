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
from shot_boundary_detection import extract_frames

def extract_letter(text: str) -> Optional[str]:
    """Extracts the letter from the provided text."""
    match = re.search(r'\(([A-D])\)|\b([A-D])\b', text)
    return match.group(1) if match else None

def video_eval_nframes(model, processor, question_text: str, options: List[str], frames1: List[Image.Image], 
                       frames2: List[Image.Image], frames3: Optional[List[Image.Image]] = None) -> str:
    """Evaluates the video frames using the model and returns the model's answer."""

    prompt = (f"You are given frames of a video. Choose which of the following options correctly explains what happened in the video frames shown here?\n"
                f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}. \nProvide the correct option letter enclosed in ().")
    frames = frames1 + frames2 + frames3

    print(prompt)
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

def process_questions(dataset: List[dict], output_path: str, surprise_scoring=False, scores_data=[], model_path=None, pick_mode="motion", num_frames=30, duration_based=False, temperature=0.7) -> None:
    """Processes questions from the dataset."""
    model, processor = load_model_and_processor(model_path=model_path)
    task1_correct = task1_failed = task1_total = task2_correct = task2_failed = task2_total = 0
    eval_ai_json = {}
    print("Pick mode:{}, Num frames: {}".format(pick_mode, num_frames))


    for item in tqdm(dataset):
        try:
            vid_path = item['video_path']
            mcq_type = item["task"]
            mcq_type = 2 if mcq_type == "Reporter" else 1
            if mcq_type == 1:
                print("Skip Processing Task 1 question id:")
                continue

        
            vr = VideoReader(vid_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()
        
            frame_indices = extract_frames(vid_path, B=num_frames, mode=pick_mode)
            print("Extracted frames with adaptive sampling", len(frame_indices), frame_indices)


            top_frames = vr.get_batch(frame_indices).asnumpy()
            preevent_frames = top_frames[:int(len(top_frames)/3)]
            event_frames = top_frames[int(len(top_frames)/3):int(2*len(top_frames)/3)]
            postevent_frames = top_frames[int(2*len(top_frames)/3):]
                
            options = item["mcq_options"]
            answer_index = item["mcq_gt_option"]


            if mcq_type == 2:
                model_answer = video_eval_nframes(model, processor, "", options, preevent_frames, event_frames, postevent_frames)
                
            else:
                model_answer = video_eval_nframes(model, processor, "", options, preevent_frames[:int(len(top_frames)*0.1)], postevent_frames[int(len(top_frames)*0.9):])
                model_answer = ""
               
            out = extract_letter(model_answer)
            if out is None:
                out = "F"
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
    parser.add_argument("--scores_path", type=str, default=None, help="Enable surprise scoring + adaptive sampling, provide path.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--pick_mode", type=str, default="motion", help="Pick mode for adaptive sampling")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to extract from each segment")
    
    args = parser.parse_args()
    scores_path = args.scores_path


    with open(args.dataset_path, 'r') as f:
        mcq_list = json.load(f)
    
    scores_data = []

    process_questions(mcq_list, args.save_path, surprise_scoring=None, scores_data=[], model_path=args.model_path, pick_mode=args.pick_mode, num_frames=args.num_frames, duration_based=False, temperature=1)
    print(f"Results saved to {args.save_path}")