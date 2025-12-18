#!/bin/bash

set -e

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

for dataset in "${datasets[@]}"
do 
    echo "正在评估 $dataset ..."
    mkdir -p "./results/qwen2.5-7b-instruct"
    python results/generate.py \
        --dataset "$dataset" \
        --model "qwen2.5-7b-instruct" \
        --batch_size 25 \
        --n_responses 4 \
        --output "./results/qwen2.5-7b-instruct/$(basename $dataset).json" \
        --interval 10 \
    > "./results/qwen2.5-7b-instruct/generate_$(basename $dataset).log"
done
