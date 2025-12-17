#!/bin/bash

set -e

models=(
    # "qwen2.5-math-1.5b-instruct"
    # "qwen2.5-math-7b-instruct"
    # "qwen2.5-7b-instruct"
    # "qwen2.5-3b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

for model in "${models[@]}"
do
    for filename in /root/prm/results/$model/*_eval.json
    do  
        echo "正在转换 $filename ..."
        python results/formatted_pass@n.py \
            --filename $filename \
            --output "/root/prm/results/$model/$(basename ${filename%.*})_pass@k_format.json" \
        > "/root/prm/results/$model/$(basename ${filename%.*})_transform_pass@k.log"
    done
done