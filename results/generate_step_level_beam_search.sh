#!/bin/bash
# nohup vllm serve Qwen/Qwen2.5-Math-1.5B --port 12345 --tensor_parallel_size 1 --enable_prefix_caching --gpu_memory_utilization 0.9 --max_model_len 4096 > vllm_server.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/root/autodl-tmp/hf-mirror
export HF_ENDPOINT="https://hf-mirror.com"

set -x

# trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT
models=(
    # "qwen2.5-math-1.5b-instruct"
    "qwen2.5-math-7b-instruct"
    # "qwen2.5-3b-instruct"
    # "qwen2.5-7b-instruct"
    # "qwen2.5-14b-instruct"
    # "qwen2.5-32b-instruct"
)

datasets=(
    # "weepcat/Gaokao2023-Math-En"
    "weepcat/minervamath"
    # "weepcat/MATH-500"
    # "weepcat/gsm8k"
)

prm_models=(
    "Qwen/Qwen2.5-Math-PRM-7B"
    # "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
    # "/root/autodl-tmp/checkpoint/qwen2.5-math-1.5b-nait-stage1"
)

pkill -f "vllm serve"
for model in "${models[@]}"
do  
    echo "Processing model: $model"
    output_path="./results/$model/$(basename ${dataset%.*})_step_level_beam_search.json"
    if [ "$model" == "qwen2.5-math-1.5b-instruct" ]; then
        model="Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif [ "$model" == "qwen2.5-math-7b-instruct" ]; then
        model="Qwen/Qwen2.5-Math-7B-Instruct"
    elif [ "$model" == "qwen2.5-7b-instruct" ]; then
        model="Qwen/Qwen2.5-7B-Instruct"
    elif [ "$model" == "qwen2.5-3b-instruct" ]; then
        model="Qwen/Qwen2.5-3B-Instruct"
    elif [ "$model" == "qwen2.5-14b-instruct" ]; then
        model="Qwen/Qwen2.5-14B-Instruct"
    fi

    CUDA_VISIBLE_DEVICES=1 nohup vllm serve $model --port 12345 --tensor_parallel_size 1 --enable_prefix_caching --gpu_memory_utilization 0.9 --max_model_len 4096 > vllm_server.log 2>&1 &
    sleep 120 # 等待服务器启动
    echo "Current time: $(date)"
    for dataset in "${datasets[@]}"
    do  
        echo "正在生成 $dataset 的 step-level beam search 结果..."
        for prm_model in "${prm_models[@]}"
        do
            echo "使用 PRM 模型: $prm_model"
            CUDA_VISIBLE_DEVICES=0 python results/generate_step_level_beam_search.py \
                --dataset "$dataset" \
                --split "test" \
                --model "$model" \
                --prm_model "$prm_model" \
                --batch_size 1 \
                --num_beams 4 \
                --output "$output_path"
        done
    done
    pkill -f "vllm serve $model"
    sleep 10
done
