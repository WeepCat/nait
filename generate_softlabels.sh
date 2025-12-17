#!/bin/bash

# Configuration
PRETRAIN="/root/autodl-tmp/checkpoint/qwen-math-1.5b-prm-hard-stage1"
DATASET="weepcat/MCRD_math-1.5b_14b"
OUTPUT_PATH="/root/autodl-tmp/softlabels/qwen-math-1.5b-prm/stage-1/"
BATCH_SIZE=16

# Set environment for HuggingFace mirror
export HF_HOME="/root/autodl-tmp/hf-mirror"
export HF_ENDPOINT="https://hf-mirror.com"

# Run distributed training with accelerate
accelerate launch \
    generate_softlabel_dataset.py \
    --pretrain ${PRETRAIN} \
    --dataset ${DATASET} \
    --dataset_split train \
    --output_path ${OUTPUT_PATH} \
    --max_length 4244 \
    --reward_tokens "[POS]" "[NEG]" \
    --placeholder_token "[PRM]" \
    --batch_size ${BATCH_SIZE}

echo "Soft label generation completed!"