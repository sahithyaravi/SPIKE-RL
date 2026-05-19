#!/bin/bash
# 1 h100 needed
# source ../.bashrc
source ../videobpo/bin/activate

MAX_FRAMES=32
RESULT_FOLDER="results/captions/ExFunTube/$MAX_FRAMES"

# surprise captions
JSON_PATH=data/ExFunTube/ExFunTube_one_moment.json
SCORES_PATH=/home/sahiravi/projects/def-vshwartz/sahiravi/scratch/results/ExFunTube/prior_frame_bayesian_approach/results_final.json
VIDEO_ROOT="data/ExFunTube/data/ExFunTube/videos"
output_file=$RESULT_FOLDER/exfuntube_bayesian.jsonl


# uniform captions
output_file=$RESULT_FOLDER/exfuntube_uniform.jsonl

python -u src/open_r1_video/inference/captioning_cached.py \
    --json_path "$JSON_PATH" \
    --video_root "$VIDEO_ROOT" \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output $output_file \
    --max_frames $MAX_FRAMES \
    --surprise_scoring "uniform"\
 

# surprise captions
python -u src/open_r1_video/inference/captioning_cached.py \
    --json_path "$JSON_PATH" \
    --video_root "$VIDEO_ROOT" \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output $output_file \
    --max_frames $MAX_FRAMES \
    --scores_path $SCORES_PATH\
    --surprise_scoring "prior_frame_bayesian_approach"


