#!/bin/bash
# 1 h100 needed
# source ../.bashrc
source ../videobpo/bin/activate
source ../.bashrc

# Define variables
JSON_PATH="data/FunQA/FunQA_test.json"
VIDEO_ROOT="data/FunQA/test"
MODEL="Qwen/Qwen2.5-VL-32B-Instruct"

MAX_FRAMES=64
RESULT_FOLDER="results/captions_32B/funqa/$MAX_FRAMES"

SCORES_PATH=data/FunQA/results_final_funqa.json
output_file=$RESULT_FOLDER/funqa_nonuniform_captions.jsonl

python -u src/open_r1_video/inference/captioning_cached.py \
    --json_path "$JSON_PATH" \
    --video_root "$VIDEO_ROOT" \
    --model_path $MODEL\
    --output $output_file \
    --scores_path $SCORES_PATH\
    --max_frames $MAX_FRAMES \
    --prompt "Provide a detailed account of the video's funny moment in 1 sentence." \
    --surprise_scoring "prior_frame_bayesian_approach" \


