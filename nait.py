import argparse
import os
os.environ['HF_HOME'] = "/root/autodl-tmp/hf-mirror"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from openrlhf.datasets.utils import blending_datasets

from prm_actor import PrmActor
from utils import get_tokenizer, get_strategy, compute_positive_negative_samples, refine_soft_labels
import math
from transformers.trainer import get_scheduler
# from openrlhf.datasets.process_reward_dataset import ProcessRewardDataset
from prm_dataset import ProcessRewardDatasetWithHardLabel, ProcessRewardDatasetWithSoftLabel
from prm_trainer import ProcessRewardModelTrainer

def train(args):

    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    print("Loading model and tokenizer...")
    tokenizer = get_tokenizer(args.pretrain, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    model = PrmActor(
        args.pretrain,
        tokenizer,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    print("Model and tokenizer loaded.")

    if args.stage_mode == "hard":
        dataset_cls = ProcessRewardDatasetWithHardLabel
        # 第一次 hard label 训练时需要对 model 的词表进行额外处理
        tokenizer.add_special_tokens({'additional_special_tokens':[args.placeholder_token] + args.reward_tokens})
        model.model.resize_token_embeddings(len(tokenizer))

    elif args.stage_mode == "soft":
        dataset_cls = ProcessRewardDatasetWithSoftLabel

    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    if args.stage_mode == "soft":
        train_data = refine_soft_labels(
            train_data,
            label_key=args.label_key,
            softlabel_key="softlabel",
            positive_token=args.reward_tokens[0],
            negative_token=args.reward_tokens[1],
            threshold=args.nait_threshold,
        )

    train_dataset = dataset_cls(train_data, tokenizer, args.train_max_len, strategy)
    pos_cnt, neg_cnt = compute_positive_negative_samples(
        train_data,
        label_key=args.label_key,
        positive_token=args.reward_tokens[0],
        negative_token=args.reward_tokens[1],
    )
    # 计算正负样本的比例，用于 loss 计算
    pos_neg_ratio = 1.0
    if args.weighted_loss:
        strategy.print(f"Positive samples: {pos_cnt}, Negative samples: {neg_cnt}")
        if neg_cnt > 0:
            pos_neg_ratio = pos_cnt / neg_cnt
        else:
            pos_neg_ratio = 1.0
        strategy.print(f"Positive/Negative ratio: {pos_neg_ratio:.4f}")

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataset = None
    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = ProcessRewardDatasetWithHardLabel(eval_data, tokenizer, args.eval_max_len, strategy)
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = ProcessRewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        pos_neg_ratio=pos_neg_ratio,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/checkpoint/qwen-math-1.5b-prm")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--ckpt_path", type=str, default="/root/autodl-tmp/checkpoint/qwen-math-1.5b-prm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed & training setup
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=3, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", help="Disable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=True, help="Enable FlashAttention2")
    parser.add_argument("--no_flash_attn", dest="flash_attn", action="store_false", help="Disable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # PRM training
    parser.add_argument("--pretrain", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--micro_train_batch_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16, help="Global training batch size")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--placeholder_token", type=str, default="[PRM]")
    parser.add_argument("--reward_tokens", type=str, nargs="*", default=["[POS]", "[NEG]"])

    parser.add_argument("--packing_samples", action="store_true", default=True)
    parser.add_argument("--no_packing_samples", dest="packing_samples", action="store_false")

    # Dataset
    parser.add_argument("--dataset", type=str, default="weepcat/MCRD_math-1.5b_14b", help="Training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default="weepcat/ProcessBench_eval_500", help="Evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=1_000_000, help="Maximum samples to use")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key for inputs")
    parser.add_argument("--label_key", type=str, default="value", help="JSON dataset key for labels")
    parser.add_argument("--train_max_len", type=int, default=4096, help="Max tokens per sample")
    parser.add_argument("--eval_max_len", type=int, default=4096, help="Max tokens per sample")

    # Logging
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default="WeepCat-PRM")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="OpenRLHF-PRM")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="prm_iter_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")
    parser.add_argument("--use_ms", action="store_true", default=False, help="Use ModelScope datasets")

    # NAIT specific
    parser.add_argument("--nait_threshold", type=float, default=0.1, help="Threshold for NAIT soft label refinement")
    parser.add_argument("--weighted_loss", action="store_true", default=True, help="Use weighted loss based on pos/neg ratio")
    parser.add_argument("--stage_mode", type=str, choices=("hard", "soft"), default="hard", help="Select which stage to execute: hard-label or soft-label training.")
    args = parser.parse_args()
    print(args)
    train(args)

    # trainer = IterativePRMTrainer(args)
    # trainer.fit()
