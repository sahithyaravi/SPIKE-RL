#!/bin/bash
source ../.bashrc
source ../videobpo/bin/activate

# Run surprise
METHOD="prior_frame_bayesian_approach"
RESULT_FOLDER="results/oops/$METHOD"

echo "Running inference with the following parameters:"
python -u src/open_r1_video/inference/inference.py\
  --json_path "$JSON_PATH" \
  --video_root "$VIDEO_ROOT" \
  --result_folder "$RESULT_FOLDER" \
  --method $METHOD\
  --model $MODEL\
  --topk_hyp 3 \
