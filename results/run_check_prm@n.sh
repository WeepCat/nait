#!/bin/bash

set -e

models=(
    "qwen2.5-math-1.5b-instruct"
    "qwen2.5-math-7b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
    "qwen2.5-3b-instruct"
)

datasets=(
    "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    "weepcat/MATH-500"
    "weepcat/gsm8k"
)

prm_models=(
    "Qwen/Qwen2.5-Math-PRM-7B"
)

for model in "${models[@]}"
do  
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        echo "正在检查 $dataset ..."
        filename="./results/$model/$(basename ${dataset%.*})_prm@k_format.json"
        for prm_model in "${prm_models[@]}"
        do
            echo "使用 PRM 模型: $prm_model"
            python results/run_check_prm@n.py \
                --filename "$filename" \
                --model_name "$prm_model" \
                --output "$filename" \
                --reference_file "./results/$model/$(basename ${dataset%.*})_pass@k_format.json" \
            > "./results/$model/$(basename ${dataset%.*})_check_prm@n.log"
        done
    done
done