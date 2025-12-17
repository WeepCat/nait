#!/bin/bash
# 46234
LOG_DIR="/root/autodl-tmp/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/prm_hard_$(date +%Y%m%d_%H%M%S).log"

deepspeed --num_gpus=1 iterative_prm_training.py \
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --dataset weepcat/MCRD_math-1.5b_14b \
    --eval_dataset weepcat/ProcessBench_eval_500 \
    --save_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm-hard-stage1 \
    --stage_mode hard \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --micro_train_batch_size 2 \
    2>&1 | tee ${LOG_FILE}

echo "Log saved to: ${LOG_FILE}"