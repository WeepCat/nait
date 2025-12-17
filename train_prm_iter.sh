#!/usr/bin/env bash
set -euo pipefail
set -x

usage() {
    cat <<'USAGE'
Usage: train_prm_iterative.sh [options]

Options:
    --stage {hard|soft}        Select which stage to run (default: hard)
    --stage-id N               1-based stage identifier used for naming outputs
    --pretrain PATH            Override the checkpoint/model used for --pretrain
  --no-shutdown              Skip shutdown after a local run
  --slurm                    Skip execution (for Slurm submission scripts)
  -h, --help                 Show this help message and exit

Environment overrides:
    STAGE_MODE, STAGE_ID, PRETRAIN_PATH, SHUTDOWN_AFTER

Examples:
    PRETRAIN_PATH=Qwen/Qwen2.5-Math-1.5B bash ./train_prm_iterative.sh --stage hard --stage-id 1 
    PRETRAIN_PATH=/root/autodl-tmp/checkpoint/qwen-math-1.5b-prm/stages/stage1 bash train_prm_iterative.sh --stage soft --stage-id 2
USAGE
}

STAGE_MODE=${STAGE_MODE:-hard}
STAGE_ID=${STAGE_ID:-}
PRETRAIN_PATH=${PRETRAIN_PATH:-Qwen/Qwen2.5-Math-1.5B}
SHUTDOWN_AFTER=${SHUTDOWN_AFTER:-true}
SLURM_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE_MODE="$2"
            shift 2
            ;;
        --stage-id)
            STAGE_ID="$2"
            shift 2
            ;;
        --pretrain)
            PRETRAIN_PATH="$2"
            shift 2
            ;;
        --no-shutdown)
            SHUTDOWN_AFTER=false
            shift
            ;;
        --slurm|slurm)
            SLURM_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$STAGE_MODE" in
    hard|soft) ;;
    *)
        echo "Invalid stage mode: $STAGE_MODE (expected hard or soft)" >&2
        exit 1
        ;;
esac

if [[ "$STAGE_MODE" == "soft" && -z "$PRETRAIN_PATH" ]]; then
    echo "Soft stage requires PRETRAIN_PATH pointing to the checkpoint used for scoring." >&2
    exit 1
fi

if [[ -z "$STAGE_ID" ]]; then
    if [[ "$STAGE_MODE" == "soft" ]]; then
        STAGE_ID=2
    else
        STAGE_ID=1
    fi
fi

if ! [[ "$STAGE_ID" =~ ^[0-9]+$ ]] || [[ "$STAGE_ID" -lt 1 ]]; then
    echo "Invalid stage ID: $STAGE_ID (must be a positive integer)." >&2
    exit 1
fi

base_cmd=(
    iterative_prm_training
    --save_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm
    --ckpt_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm
    --logging_steps 1
    --eval_steps 100
    --train_batch_size 16
    --micro_train_batch_size 2
    --pretrain "$PRETRAIN_PATH"
    --bf16
    --max_epochs 1
    --train_max_len 4244
    --eval_max_len 4244
    --zero_stage 3
    --learning_rate 1e-6
    --dataset weepcat/MCRD_math-1.5b_14b
    --eval_dataset weepcat/ProcessBench_eval_500
    --eval_split val
    --input_key input
    --label_key value
    --flash_attn
    --packing_samples
    --placeholder_token "[PRM]"
    --reward_tokens "[POS]" "[NEG]"
    --iter_label_batch_size 2
    --iter_output_dir /root/autodl-tmp/iterative_runs/qwen-math-1.5b-prm
    --use_wandb True
    --wandb_org WeepCat-PRM
    --wandb_project OpenRLHF-PRM
    --stage_mode "$STAGE_MODE"
    --stage_id "$STAGE_ID"
)

printf 'Launching iterative PRM training with command:\n  %s\n' "$(printf '%q ' "${base_cmd[@]}")"

if [[ "$SLURM_MODE" == true ]]; then
    exit 0
fi

deepspeed --module "${base_cmd[@]}"
status=$?

if [[ "$SHUTDOWN_AFTER" == true ]]; then
    /usr/bin/shutdown
fi

exit $status
