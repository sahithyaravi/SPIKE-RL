#!/bin/bash
# 4 H100s needed 

source .env
source ../envs/videobpo/bin/activate

# display current python venv
echo "Current Python Virtual Environment:"
which python

# display current python version
echo "Current Python Version:"
python --version

# Make sure WANDB_API_KEY is set
export WANDB_PROJECT=Qwen2-VL-7B-Video-GRPO-belief
export WANDB_NAME=belif_optimization_oops_activitynet_2k_examples
export FLASH_ATTENTION_USE_TILED=1
export FLASH_ATTENTION_BLOCK_HEURISTIC=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p ckpt/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12355" \
    src/open_r1_video/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir data/ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name xxx \
    --jsonl_path data/ActivityNet_Captions/activitynet_captions_train.json data/oops_dataset/train_captions.json\
    --max_prompt_length 8192 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --torch_dtype 'bfloat16' \
    --bf16 true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 25 \
    --use_peft \
    --save_total_limit 2 \
    --training_data_size 2000\
    --dataloader_num_workers 4 \
