#!/bin/bash

# Define variables
JSON_PATH="data/Video-MME/test_short.json"
VIDEO_ROOT="data/Video-MME/data"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Run surprise
METHOD="prior_frame_bayesian_approach"
RESULT_FOLDER="/results/Video-MME/$METHOD"

python -u src/open_r1_video/inference/inference.py\
  --json_path "$JSON_PATH" \
  --video_root "$VIDEO_ROOT" \
  --result_folder "$RESULT_FOLDER" \
  --method $METHOD\
  --topk_hyp 3 \
  --model $MODEL\

