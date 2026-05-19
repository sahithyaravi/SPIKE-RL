source ../.bashrc
source ../videobpo/bin/activate

JSON_PATH=data/Video-MME/test_short.json # replace with other splits as needed
SCORES_PATH=results/Video-MME/prior_frame_bayesian_approach/results_final.json
RESULT_FOLDER=results/Video-MME/Video-MME_vqa_32B
TEMP=0.7

PICK_MODE="weighted"
NUM_FRAMES=64
mkdir -p $RESULT_FOLDER

output_file="${RESULT_FOLDER}/Video-MME_test_${TEMP}_${PICK_MODE}.json"

python -u src/open_r1_video/inference_qa/videomme.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH \
    --num_frames $NUM_FRAMES\
    --pick_mode $PICK_MODE \
    --surprise_sampling \
    --temperature $TEMP \

output_file="${RESULT_FOLDER}/Video-MME_test_uni${NUM_FRAMES}.json"
python -u src/open_r1_video/inference_qa/videomme.py \
    --dataset_path "$JSON_PATH" \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --save_path $output_file \
    --scores_path $SCORES_PATH \
    --num_frames $NUM_FRAMES\

