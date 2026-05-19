#!/bin/bash
#SBATCH --account=aip-vshwartz
#SBATCH --job-name=caption
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:10:00


source ../.bashrc
source ../videobpo/bin/activate

JSON_PATH=data/oops/BlackSwanSuite_MCQ_Test_final_with_answers.json
SCORES_PATH=results/oops_grpo/prior_frame_bayesian_approach/results_final.json
RESULT_FOLDER=results/oops_grpo/blackswan_vqa_v2_32B
PICK_MODE="weighted"
NUM_FRAMES=64
mkdir -p $RESULT_FOLDER
VIDEO_ROOT="data/oops"
TEMP=0.7
output_file="${RESULT_FOLDER}/blackswan_bayesian_t${TEMP}_${PICK_MODE}_${NUM_FRAMES}.json"
echo $output_file

python -u src/open_r1_video/inference_qa/blackswan.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH\
    --surprise_sampling \
    --num_frames $NUM_FRAMES\
    --pick_mode $PICK_MODE \
    --temperature $TEMP \


output_file="${RESULT_FOLDER}/blackswan_uniform_${NUM_FRAMES}.json"
echo $output_file

python -u src/open_r1_video/inference_qa/blackswan.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH \
    --num_frames 64\