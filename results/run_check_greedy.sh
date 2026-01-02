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
        batch_size=400
        echo "正在校验 $dataset ..."
        python results/run_check_greedy.py \
            --filename ./results/$model/$(basename $dataset).json \
            --model "deepseek-chat" \
            --batch_size $batch_size \
            --output "./results/$model/$(basename ${dataset%.*})_greedy_check.json" \
            --interval 10 \
        > "./results/$model/$(basename ${dataset%.*})_greedy_check.log"
    done
done