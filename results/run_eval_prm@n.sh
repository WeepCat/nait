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

prm_models=(
    "Qwen/Qwen2.5-Math-PRM-7B"
    "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    "peiyi9979/math-shepherd-mistral-7b-prm"
    "PRIME-RL/EurusPRM-Stage1"
    "PRIME-RL/EurusPRM-Stage2"
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    # "/root/autodl-tmp/checkpoint/qwen2.5-math-1.5b-nait-stage1"
)

for model in "${models[@]}"
do  
    echo "Processing model: $model"
    for dataset in "${datasets[@]}"
    do  
        echo "正在统计 $dataset ..."
        for prm_model in "${prm_models[@]}"
        do
            echo "使用 PRM 模型: $prm_model"
            python results/run_eval_prm@n.py \
                --filename ./results/$model/$(basename $dataset)_prm@k_format.json \
                --model_name "$prm_model" \
                --output ./results/$model/stats_prm@k.txt \
            > "./results/$model/$(basename ${dataset%.*})_prm@k.log"
        done
    done
done