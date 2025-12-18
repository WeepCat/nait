set -e


python results/run_eval.py \
    --filename ./results/minervamath_7b_math.json \
    --model deepseek-chat \
    --batch_size 100 \
    --output "./results/result.json" \
    --interval 10