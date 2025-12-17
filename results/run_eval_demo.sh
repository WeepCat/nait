set -e


python results/run_eval.py \
    --filename /root/prm/results/minervamath_7b_math.json \
    --model deepseek-chat \
    --batch_size 100 \
    --output "/root/prm/results/result.json" \
    --interval 10