#!/bin/bash
source ../.bashrc
source ../videobpo/bin/activate

# Define variables
JSON_PATH="data/bean/final_clips_metadata_with_volume.json"
VIDEO_ROOT="data/bean/final_clips"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Run surprise
METHOD="prior_frame_bayesian_approach"
RESULT_FOLDER="/results/bean/$METHOD"

python -u src/open_r1_video/inference/inference.py\
  --json_path "$JSON_PATH" \
  --video_root "$VIDEO_ROOT" \
  --result_folder "$RESULT_FOLDER" \
  --method $METHOD\
  --topk_hyp 3 \
  --model $MODEL\
