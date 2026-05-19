#!/bin/bash
source ../.bashrc
source ../videobpo/bin/activate

# Define variables
JSON_PATH="data/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv"
VIDEO_ROOT="data/ydata-tvsum50-v1_1/video"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Run surprise
METHOD="prior_frame_bayesian_approach"
RESULT_FOLDER="results/tvsum_v5/$METHOD"

python -u src/open_r1_video/inference/inference.py\
  --json_path "$JSON_PATH" \
  --video_root "$VIDEO_ROOT" \
  --result_folder "$RESULT_FOLDER" \
  --method $METHOD\
  --topk_hyp 3 \
  --model $MODEL\
  --use_history \



