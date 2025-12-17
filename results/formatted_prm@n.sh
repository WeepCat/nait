#!/bin/bash

set -e

models=(
    "qwen2.5-math-1.5b-instruct"
    "qwen2.5-math-7b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
    # "qwen2.5-3b-instruct"
)

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

for model in "${models[@]}"
do  
    for dataset in "${datasets[@]}"
    do  
        echo "正在转换 $filename ..."
        python results/formatted_prm@n.py \
            --filename /root/prm/results/$model/$(basename $dataset).json \
            --output "/root/prm/results/$model/$(basename ${dataset%.*})_prm@k_format.json" \
        > "/root/prm/results/$model/$(basename ${dataset%.*})_transform_prm@k.log"
    done
done