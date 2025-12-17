import os
os.environ['HF_HOME'] = "/root/autodl-tmp/hf-mirror"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import sys
import torch
import argparse
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from openrlhf.utils.utils import convert_token_to_id
from torch.utils.data import DataLoader, DistributedSampler
from transformers.data import DataCollatorWithPadding
import gc


def process_batch(batch, model, reward_token_ids, placeholder_token_id, device=None):
    """处理单个batch，返回soft labels"""
    # 如果指定了device，将数据移到该device
    if device is not None:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    else:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, S, V]

    # 提取reward tokens的logits并计算概率
    reward_idx = torch.tensor(reward_token_ids, device=logits.device, dtype=torch.long)
    selected = logits.index_select(-1, reward_idx)  # [B, S, 2]
    logits_diff = selected[..., 0] - selected[..., 1]  # [B, S]
    probs = torch.sigmoid(logits_diff)  # [B, S]

    # 提取placeholder位置的概率
    batch_mask = (input_ids == placeholder_token_id)  # [B, S]
    
    results = []
    for i in range(input_ids.size(0)):
        mask_i = batch_mask[i]
        if mask_i.any():
            probs_i = probs[i][mask_i]
            results.append([float(x) for x in probs_i.cpu().tolist()])
    
    return results


def generate_softlabel(args):
    """生成soft labels"""
    accelerator = Accelerator()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        model = AutoModelForCausalLM.from_pretrained(args.pretrain, dtype=torch.bfloat16)
        model = accelerator.prepare(model)
        accelerator.print("Model prepared with Accelerator.")
        
        reward_token_ids = [convert_token_to_id(t, tokenizer) for t in args.reward_tokens]
        placeholder_token_id = convert_token_to_id(args.placeholder_token, tokenizer)

        dataset = load_dataset(args.dataset, split=args.dataset_split)
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
        def tokenize_fn(examples):
            return tokenizer(
                examples["input"], 
                truncation=False, 
                max_length=args.max_length, 
                add_special_tokens=False
            )
        
        processed_dataset = dataset.map(
            tokenize_fn, 
            remove_columns=dataset.column_names, 
            num_proc=8
        )
        
        processed_dataset = processed_dataset.add_column(
            "original_index", 
            list(range(len(processed_dataset)))
        )
        
        collator = DataCollatorWithPadding(tokenizer)
        sampler = DistributedSampler(
            processed_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False
        )
        dataloader = DataLoader(
            processed_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collator
        )
        
        accelerator.print(f"Dataset loaded with {len(processed_dataset)} samples.")

        model.eval()
        local_results = {}
        
        with torch.no_grad():
            for batch in tqdm(
                dataloader, 
                desc="Generating soft labels", 
                disable=not accelerator.is_main_process
            ):
                batch_results = process_batch(
                    batch, model, reward_token_ids, placeholder_token_id, device=accelerator.device
                )
                
                indices = batch["original_index"].cpu().numpy()
                for idx, result in zip(indices, batch_results):
                    local_results[int(idx)] = result
                
                del batch, batch_results
                torch.cuda.empty_cache()
        
        local_results_list = [(idx, labels) for idx, labels in local_results.items()]
        all_results = accelerator.gather_for_metrics(
            local_results_list, 
            use_gather_object=True
        )
        
        if accelerator.is_main_process:
            result_dict = {}
            for idx, labels in all_results:
                if idx not in result_dict:
                    result_dict[idx] = labels
            
            missing = set(range(len(dataset))) - set(result_dict.keys())
            if missing:
                accelerator.print(f"WARNING: Missing {len(missing)} indices")
            
            final_results = [result_dict[i] for i in range(len(dataset))]
            
            dataset = dataset.add_column("softlabel", final_results)

            for data in dataset:
                assert len(data['softlabel']) == len(data["value"])
                
            model_name = os.path.basename(args.pretrain)
            dataset.push_to_hub(f"weepcat/{model_name}")
            # dataset.save_to_disk(args.output_path)
            accelerator.print(f"Soft-labeled dataset saved to {args.output_path}")
        
        # 等待所有进程
        accelerator.wait_for_everyone()
        
    finally:
        # 清理模型
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'dataset' in locals():
            del dataset
        if 'processed_dataset' in locals():
            del processed_dataset
        if 'dataloader' in locals():
            del dataloader
            
        # 清理 accelerator
        accelerator.free_memory()
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 等待所有进程同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        
        # 强制退出进程
        accelerator.print("Cleaning up and exiting...")
        sys.stdout.flush()


def verify_results(softlabels_1, softlabels_2, tolerance=1e-6):
    """验证两个结果是否一致"""
    assert len(softlabels_1) == len(softlabels_2), \
        f"Total samples mismatch: {len(softlabels_1)} vs {len(softlabels_2)}"
    
    for i, (sl1, sl2) in enumerate(zip(softlabels_1, softlabels_2)):
        assert len(sl1) == len(sl2), \
            f"Length mismatch at index {i}: {len(sl1)} vs {len(sl2)}"
        
    print("✓ Soft label verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="weepcat/MCRD_math-1.5b_14b")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, 
                       default="/root/autodl-tmp/iterative_runs/qwen-math-1.5b-prm/softlabel/")
    parser.add_argument("--max_length", type=int, default=4244)
    parser.add_argument("--reward_tokens", type=str, nargs="*", default=["[POS]", "[NEG]"])
    parser.add_argument("--placeholder_token", type=str, default="[PRM]")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--verify", action="store_true", help="验证单GPU和多GPU结果一致性")
    args = parser.parse_args()
    args.verify = False
    try:
        if args.verify:
            # 验证模式：对比单GPU和多GPU结果
            print("=== Running single GPU mode ===")
            softlabels_single = generate_softlabel(args)
            
            # print("\n=== Running distributed mode ===")
            # softlabels_multi = generate_soft_labels(args)
            
            # if softlabels_multi is not None:  # 只在主进程验证
            #     verify_results(softlabels_single, softlabels_multi)
        else:
            # 正常模式：只运行多GPU
            generate_softlabel(args)
    finally:
        # 确保清理分布式资源
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            # 强制刷新并退出
        sys.stdout.flush()
        sys.stderr.flush()