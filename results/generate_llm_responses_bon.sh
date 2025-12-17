#!/bin/bash

# 任何命令失败立即退出（注意：这在并行场景下需要特殊处理）
set -e

# 退出时（无论成功/失败/中断）都关机
trap 'echo "[$(date)] 脚本退出，10秒后关机... "; sleep 10; /bin/shutdown -h now' EXIT

models=(
    "qwen2.5-math-7b-instruct"
    "qwen2.5-math-1.5b-instruct"
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

# 单个模型串行处理所有数据集的函数
run_model_pipeline() {
    local model=$1
    shift
    local datasets=("$@")
    
    echo "[$(date)] 模型 $model 开始处理，共 ${#datasets[@]} 个数据集"
    
    # 确定该模型的参数
    if [[ "$model" == *math* ]]; then
        batch_size=100
        n_responses=1
    else
        batch_size=25
        n_responses=4
    fi
    
    mkdir -p "/root/prm/results/$model"
    
    # 串行处理每个数据集
    for dataset in "${datasets[@]}"
    do
        echo "[$(date)] [$model] 开始处理数据集:  $dataset"
        
        python results/generate.py \
            --dataset "$dataset" \
            --model "$model" \
            --batch_size $batch_size \
            --n_responses $n_responses \
            --output "/root/prm/results/$model/$(basename $dataset).json" \
            --interval 10 \
        > "/root/prm/results/$model/generate_$(basename $dataset).log" 2>&1
        
        echo "[$(date)] [$model] 完成数据集:  $dataset"
    done
    
    echo "[$(date)] 模型 $model 已完成所有数据集！"
}

echo "=========================================="
echo "[$(date)] 开始并行评估"
echo "模型数量: ${#models[@]}"
echo "数据集数量:  ${#datasets[@]}"
echo "=========================================="

# 存储所有后台进程的 PID
pids=()

# 为每个模型启动独立的后台任务
for model in "${models[@]}"
do
    run_model_pipeline "$model" "${datasets[@]}" &
    pids+=($!)
    echo "启动模型 $model 的 pipeline，PID: $!"
done

echo ""
echo "[$(date)] 已启动 ${#pids[@]} 个模型 pipeline，等待全部完成..."
echo ""

# 等待所有后台任务完成，并记录失败的任务
failed=0
for i in "${! pids[@]}"
do
    pid=${pids[$i]}
    model=${models[$i]}
    if wait $pid; then
        echo "[$(date)] 模型 $model (PID: $pid) 成功完成"
    else
        echo "[$(date)] 模型 $model (PID:  $pid) 执行失败！"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=========================================="
echo "[$(date)] 所有评估任务已完成！"
echo "成功:  $((${#models[@]} - failed)) / ${#models[@]}"
echo "失败: $failed / ${#models[@]}"
echo "=========================================="

# 如果有失败的任务，返回非零退出码
if [ $failed -gt 0 ]; then
    exit 1
fi