#!/bin/bash

# 任何命令失败立即退出
set -e

# 退出时（无论成功/失败/中断）都关机
# trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT
models=(
    # "qwen2.5-math-7b-instruct"
    # "qwen2.5-math-1.5b-instruct"
    # "qwen2.5-7b-instruct"
    # "qwen2.5-3b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

for model in "${models[@]}"
do
    for filename in /root/prm/results/$model/*.json
    do  
        # 如果 filename 以 _math.json 结尾，那么设置 batch_size 为 400, 否则为 100
        if [[ "$filename" == *_math.json ]]; then
            batch_size=400
        else
            batch_size=100
        fi
        echo "正在校验 $filename ..."
        python results/run_eval_pass@n_deepseekv3.py \
            --filename "$filename" \
            --model "deepseek-chat" \
            --batch_size $batch_size \
            --output "/root/prm/results/$model/$(basename ${filename%.*})_eval.json" \
            --interval 10 \
        > "/root/prm/results/$model/$(basename ${filename%.*}).log"
    done
done