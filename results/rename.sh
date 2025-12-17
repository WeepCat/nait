#!/bin/bash

set -e

models=(
    # "qwen2.5-math-1.5b-instruct"
    # "qwen2.5-math-7b-instruct"
    # "qwen2.5-7b-instruct"
    "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)
for model in "${models[@]}"
do
    for filename in /root/prm/results/$model/*_eval.json
    do
        # 写一个命令，将所有以 _eval.json 结尾的 filename 改为以 pass@k_format.json 结尾
        mv "$filename" "${filename%_eval.json}_pass@k_format.json"
    done
done