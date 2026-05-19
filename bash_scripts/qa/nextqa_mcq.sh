source ../.bashrc
source ../videobpo/bin/activate



JSON_PATH=data/NExTQA/test.json
SCORES_PATH=results_v2/NExTQA/prior_frame_bayesian_approach/results_final.json
RESULT_FOLDER=results_v2/NExTQA/NextQA_vqa_32B
TEMP=0.7

PICK_MODE="weighted"
NUM_FRAMES=64
mkdir -p $RESULT_FOLDER

output_file="${RESULT_FOLDER}/nextqa_test_t${TEMP}_${PICK_MODE}duration.json"
python -u src/open_r1_video/inference_qa/nexqa.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH \
    --num_frames $NUM_FRAMES\
    --pick_mode $PICK_MODE \
    --surprise_sampling \
    --temperature $TEMP \
    --duration_based


output_file="${RESULT_FOLDER}/NextQA_uniform_test${NUM_FRAMES}.json"
python -u src/open_r1_video/inference_qa/nexqa.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH \
    --num_frames $NUM_FRAMES\
