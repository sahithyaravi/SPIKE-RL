#!/bin/bash
source ../.bashrc
source ../videobpo/bin/activate

# Define variables
JSON_PATH="data/ExFunTube/ExFunTube_one_moment.json"
VIDEO_ROOT="data/ExFunTube/data/ExFunTube/videos"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Run surprise
METHOD="prior_frame_bayesian_approach"
RESULT_FOLDER="/results/exfuntube/$METHOD"

python -u src/open_r1_video/inference_surprise/inference.py\
  --json_path "$JSON_PATH" \
  --video_root "$VIDEO_ROOT" \
  --result_folder "$RESULT_FOLDER" \
  --method $METHOD\
  --topk_hyp 3 \
  --model $MODEL\
