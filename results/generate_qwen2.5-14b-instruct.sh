#!/bin/bash

# 任何命令失败立即退出
set -e

# 退出时（无论成功/失败/中断）都关机
# trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT

for dataset in weepcat/MATH-500 weepcat/gsm8k
# weepcat/Gaokao2023-Math-En weepcat/minervamath 
do 
    echo "正在评估 $dataset ..."
    mkdir -p "/root/prm/results/qwen2.5-14b-instruct"
    python results/generate.py \
        --dataset "$dataset" \
        --model "qwen2.5-14b-instruct" \
        --batch_size 25 \
        --n_responses 4 \
        --output "/root/prm/results/qwen2.5-14b-instruct/$(basename $dataset).json" \
        --interval 10 \
    > "/root/prm/results/qwen2.5-14b-instruct/generate_$(basename $dataset).log"
done
