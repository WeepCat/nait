#!/bin/bash

set -e

models=(
    "qwen2.5-math-1.5b-instruct"
    "qwen2.5-math-7b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-3b-instruct"
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
        echo "正在统计 $dataset ..."
        python results/run_eval_pass@n.py \
            --filename ./results/$model/$(basename $dataset)_pass@k_format.json \
            --output ./results/$model/stats.txt \
        > "./results/$model/$(basename ${dataset%.*})_pass@k.log"
    done
done