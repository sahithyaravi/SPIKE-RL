#!/bin/bash
# 1 h100 needed
# source ../.bashrc
source ../videobpo/bin/activate


RESULT_FOLDER="results/oops/captions/"
JSON_PATH=data/oops/blackswan_captions.json
SCORES_PATH=results/oops/prior_frame_bayesian_approach/results_final.json
VIDEO_ROOT="data/oops"
output_file=$RESULT_FOLDER/oops_bayesian.jsonl

# Uniform Sampling
output_file=$RESULT_FOLDER/oops_uniform.jsonl

python -u src/open_r1_video/inference/captioning_cached.py \
    --json_path "$JSON_PATH" \
    --video_root "$VIDEO_ROOT" \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output $output_file \
    --surprise_scoring "uniform"\


# Surprise Sampling
python -u src/open_r1_video/inference/captioning_cached.py \
    --json_path "$JSON_PATH" \
    --video_root "$VIDEO_ROOT" \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output $output_file \
    --scores_path $SCORES_PATH\
    --surprise_scoring "prior_frame_bayesian_approach"
