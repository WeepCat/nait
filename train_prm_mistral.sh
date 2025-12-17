# set -x
# 445960
# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_prm \
#    --save_path ./checkpoint/llama3-8b-instruct \
#    --save_steps 500 \
#    --logging_steps 1 \
#    --eval_steps 100 \
#    --train_batch_size 16 \
#    --micro_train_batch_size 1 \
#    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
#    --bf16 \
#    --lora_rank 8 \
#    --max_epochs 1 \
#    --max_len 2048 \
#    --zero_stage 3 \
#    --learning_rate 1e-6 \
#    --dataset zhuzilin/Math-Shepherd \
#    --input_key input \
#    --label_key value \
#    --flash_attn \
#    --load_checkpoint \
#    --packing_samples \
#    --wandb_group prm \
#    --placeholder_token ки \
#    --reward_tokens + -
# EOF
#      # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
#      # --packing_samples
#      # --pretrain mistralai/Mistral-7B-v0.1  \
#      # --save_path ./checkpoint/mistal-7b-prm \


# if [[ ${1} != "slurm" ]]; then
#     deepspeed --module $training_commands
# fi
set -x

read -r -d '' training_commands <<EOF
train_prm \
   --save_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm \
   --ckpt_path /root/autodl-tmp/checkpoint/qwen-math-1.5b-prm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen2.5-Math-1.5B  \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset weepcat/MCRD_math-1.5b_14b \
   --eval_dataset weepcat/ProcessBench_eval_500 \
   --eval_split val \
   --input_key input \
   --label_key value \
   --flash_attn \
   --packing_samples \
   --placeholder_token [PRM] \
   --reward_tokens [POS] [NEG] \
   --use_wandb True \
   --wandb_org WeepCat-PRM \
   --wandb_project OpenRLHF-PRM
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples
     # --dataset zhuzilin/Math-Shepherd \


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
    # ; /usr/bin/shutdown
fi