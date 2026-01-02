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
    "weepcat/gsm8k"
    "weepcat/MATH-500"
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
)

for model in "${models[@]}"
do  
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        echo "正在统计 $dataset ..."
        python results/run_eval_maj@n.py \
            --filename ./results/$model/$(basename $dataset)_pass@k_format.json \
            --output ./results/$model/stats_maj@k.txt \
        > "./results/$model/$(basename ${dataset%.*})_maj@k.log"
    done
done