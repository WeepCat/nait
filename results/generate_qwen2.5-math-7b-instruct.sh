#!/bin/bash

# 任何命令失败立即退出
set -e

# 退出时（无论成功/失败/中断）都关机
# trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT

for dataset in weepcat/minervamath weepcat/Gaokao2023-Math-En weepcat/minervamath weepcat/MATH-500 weepcat/gsm8k
do 
    echo "正在评估 $dataset ..."
    mkdir -p "/root/prm/results/qwen2.5-math-7b-instruct"
    python results/generate.py \
        --dataset "$dataset" \
        --model "qwen2.5-math-7b-instruct" \
        --batch_size 100 \
        --n_responses 1 \
        --output "/root/prm/results/qwen2.5-math-7b-instruct/$(basename $dataset).json" \
        --interval 10 \
    > "/root/prm/results/qwen2.5-math-7b-instruct/generate_$(basename $dataset).log"
done
