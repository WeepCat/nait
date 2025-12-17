#!/bin/bash
# 46234
LOG_DIR="/root/autodl-tmp/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/prm_hard_$(date +%Y%m%d_%H%M%S).log"

deepspeed --num_gpus=1 nait.py \
    --pretrain /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm-hard-stage1 \
    --dataset weepcat/qwen-math-1.5b-prm-hard-stage1 \
    --eval_dataset weepcat/ProcessBench_eval_500 \
    --save_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm-soft-stage2 \
    --stage_mode soft \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --micro_train_batch_size 2 \
    2>&1 | tee ${LOG_FILE}

echo "Log saved to: ${LOG_FILE}"