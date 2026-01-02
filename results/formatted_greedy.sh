#!/bin/bash

set -e

models=(
    "qwen2.5-math-1.5b-instruct"
    "qwen2.5-math-7b-instruct"
    "qwen2.5-3b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

for model in "${models[@]}"
do
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        echo "正在转换 $dataset ..."
        python results/formatted_greedy.py \
            --filename "./results/$model/$(basename ${dataset%.*})_greedy_check.json" \
            --output "./results/$model/$(basename ${dataset%.*})_greedy_format.json" \
        > "./results/$model/$(basename ${dataset%.*})_transform_greedy.log"
    done
done