import os
os.environ['HF_HOME'] = '/root/autodl-tmp/hf-mirror'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import argparse
import time
import asyncio
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
import requests
load_dotenv()


SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""
END_MARK = "\n\n"
api_key = "EMPTY"
base_url = "http://localhost:12345"
client = OpenAI(
    api_key=api_key,
    base_url=f"{base_url}/v1",
)


# 检测 response 是否已经完成
def check_is_finished(step_text):
    # 如果 step_text 为空或者 step_text 包含 final answer / \boxed{} 则返回 True，否则返回 False
    if not step_text.strip():
        return True
    if "final answer" in step_text.lower() or "\\boxed" in step_text.lower():
        return True
    return False


def step_level_beam_search_nait_prm(dataset, split, model_name, prm_name, batch_size, num_beams, output_path):
    # 检测 json 文件
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            ds = json.load(f)
        print(f"Found existing output file {output_path}, loaded {len(ds)} records.")
    else:
        ds = load_dataset(dataset, split=split)
    prm_model = AutoModelForCausalLM.from_pretrained(
        prm_name, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0").eval()
    prm_tokenizer = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)

    pos_tag_id = prm_tokenizer.encode('[POS]')[-1]
    neg_tag_id = prm_tokenizer.encode('[NEG]')[-1]
    candidate_tokens = [pos_tag_id, neg_tag_id]

    all_results = []
    for item in tqdm(ds):
        problem = item['problem']
        answer = item["answer"]
        prompt = SYSTEM_PROMPT + "\n\n" + problem + "\n"
        while True:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                stop=END_MARK,
                n=num_beams,
            )
            steps = [choice.text for choice in response.choices]

            # 将每个 prompt 拼上 step 文本，形成新的 prompt 交给 prm 评分
            step_rewards = []
            for step in steps:
                step_text = step.strip()
                input_ids = prm_tokenizer.encode(
                    prompt + step_text,
                    return_tensors="pt"
                ).to(prm_model.device)
                with torch.no_grad():
                    logits = prm_model(input_ids).logits
                candidate_logits = logits[:, :, candidate_tokens]
                probabilities = F.softmax(candidate_logits, dim=-1)
                step_reward = probabilities[:, -1, 0].item()  # 取正向标签的概率作为奖励
                step_rewards.append(step_reward)
            
            # 选择奖励最高的 step
            best_step_idx = step_rewards.index(max(step_rewards))
            best_step = steps[best_step_idx].strip()
            prompt += best_step + END_MARK
            print(f"Selected Step: {best_step} [Reward: {step_rewards[best_step_idx]:.4f}]")
            num_tokens = requests.post(f"{base_url}/tokenize", json={"prompt": prompt}).json()['count']
            if check_is_finished(best_step) or num_tokens >= 4096:
                break
        base_prm_name = os.path.basename(prm_name)
        all_results.append({
            "problem": problem,
            "answer": answer,
            f"{base_prm_name}": prompt.replace(SYSTEM_PROMPT + "\n\n" + problem + "\n", "").strip()
        })
        
    # 保存结果到 JSON 文件
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)


def step_level_beam_search_qwen_prm(dataset, split, model_name, prm_name, batch_size, num_beams, output_path):
    # 检测 json 文件
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            ds = json.load(f)
        print(f"Found existing output file {output_path}, loaded {len(ds)} records.")
    else:
        ds = load_dataset(dataset, split=split)
    prm_model = AutoModel.from_pretrained(
        prm_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0").eval()
    prm_tokenizer = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)


    all_results = []
    for item in tqdm(ds):
        problem = item['problem']
        answer = item["answer"]
        prompt = SYSTEM_PROMPT + "\n\n" + problem + "\n"
        while True:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                stop=END_MARK,
                n=num_beams,
            )
            steps = [choice.text for choice in response.choices]

            # 将每个 prompt 拼上 step 文本，形成新的 prompt 交给 prm 评分
            conversation = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": item["problem"]},
            ]
            step_rewards = []
            selected_steps = []
            for step in steps:
                step_text = step.strip()
                conversation_str = prm_tokenizer.apply_chat_template(
                    conversation + [
                        {"role": "assistant", "content": "<extra_0>".join(selected_steps + [step_text]) + "<extra_0>"},
                    ], 
                    tokenize=False, 
                    add_generation_prompt=False,
                )

                input_ids = prm_tokenizer.encode(
                    conversation_str,
                    return_tensors="pt"
                ).to(prm_model.device)

                with torch.no_grad():
                    outputs = prm_model(input_ids=input_ids)
                    logits = outputs[0]

                probabilities = F.softmax(logits, dim=-1)
                step_reward = probabilities[:, -1].view(-1, 2)[:, 1].item()  # 取正向标签的概率作为奖励
                step_rewards.append(step_reward)
            
            # 选择奖励最高的 step
            best_step_idx = step_rewards.index(max(step_rewards))
            best_step = steps[best_step_idx].strip()
            prompt += best_step + END_MARK
            selected_steps.append(best_step)
            print(f"Selected Step: {best_step} [Reward: {step_rewards[best_step_idx]:.4f}]")
            num_tokens = requests.post(f"{base_url}/tokenize", json={"prompt": prompt}).json()['count']
            if check_is_finished(best_step) or num_tokens >= 4096:
                break

        base_prm_name = os.path.basename(prm_name)
        all_results.append({
            "problem": problem,
            "answer": answer,
            f"{base_prm_name}": prompt.replace(SYSTEM_PROMPT + "\n\n" + problem + "\n", "").strip()
        })

    # 保存结果到 JSON 文件
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)


def step_level_beam_search_mistral_prm(dataset, split, model_name, prm_name, batch_size, num_beams, output_path):
    # 检测 json 文件
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            ds = json.load(f)
        print(f"Found existing output file {output_path}, loaded {len(ds)} records.")
    else:
        ds = load_dataset(dataset, split=split)
    prm_model = AutoModelForCausalLM.from_pretrained(
        prm_name, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0").eval()
    prm_tokenizer = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)
    prm_tokenizer.padding_side = "right"
    prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_model.config.pad_token_id = prm_model.config.eos_token_id

    # 获取 +/- token ids
    plus_tag_id = prm_tokenizer.encode('+')[-1]
    minus_tag_id = prm_tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id, minus_tag_id]

    all_results = []
    for item in tqdm(ds):
        problem = item['problem']
        answer = item["answer"]
        prompt = SYSTEM_PROMPT + "\n\n" + problem + "\n"
        while True:
            cnt = 0
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                stop=END_MARK,
                n=num_beams,
            )
            steps = [choice.text for choice in response.choices]

            conversation = []
            step_rewards = []
            for step in steps:
                if cnt == 0:
                    input_ids = prm_tokenizer.apply_chat_template(
                        conversation + [
                            {"role": "user", "content": prompt + step.strip()},
                            {"role": "assistant", "content": "+"}
                        ],
                        return_tensors="pt"
                    ).to(prm_model.device)
                else:
                    input_ids = prm_tokenizer.apply_chat_template(
                        conversation + [
                            {"role": "user", "content": step.strip()},
                            {"role": "assistant", "content": "+"}
                        ],
                        return_tensors="pt"
                    ).to(prm_model.device)

                with torch.no_grad():
                    logits = prm_model(input_ids).logits

                candidate_logits = logits[:, -3, candidate_tokens]
                probabilities = F.softmax(candidate_logits, dim=-1)
                step_reward = probabilities[:, 0].item()  # 取正向标签的概率作为奖励
                step_rewards.append(step_reward)
            
            cnt += 1
            # 选择奖励最高的 step
            best_step_idx = step_rewards.index(max(step_rewards))
            best_step = steps[best_step_idx].strip()
            prompt += best_step + END_MARK
            conversation.append({"role": "user", "content": best_step})
            conversation.append({"role": "assistant", "content": "+"})
            print(f"Selected Step: {best_step} [Reward: {step_rewards[best_step_idx]:.4f}]")
            num_tokens = requests.post(f"{base_url}/tokenize", json={"prompt": prompt}).json()['count']
            if check_is_finished(best_step) or num_tokens >= 4096:
                break

        base_prm_name = os.path.basename(prm_name)
        all_results.append({
            "problem": problem,
            "answer": answer,
            f"{base_prm_name}": prompt.replace(SYSTEM_PROMPT + "\n\n" + problem + "\n", "").strip()
        })
    
    # 保存结果到 JSON 文件
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="HuggingFace dataset split")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--prm_model", type=str, required=True, help="PRM model name")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for concurrent API calls")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--output", type=str, default="responses.json", help="Output JSON file path")
    parser.add_argument("--interval", type=int, default=5, help="Time interval between batches in seconds")
    args = parser.parse_args()

    if "nait" in args.prm_model:
        step_level_beam_search_nait_prm(
            dataset=args.dataset,
            split=args.split,
            model_name=args.model,
            prm_name=args.prm_model,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            output_path=args.output,
        )
    elif args.prm_model == "Qwen/Qwen2.5-Math-PRM-7B":
        step_level_beam_search_qwen_prm(
            dataset=args.dataset,
            split=args.split,
            model_name=args.model,
            prm_name=args.prm_model,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            output_path=args.output,
        )
    elif args.prm_model == "RLHFlow/Llama3.1-8B-PRM-Mistral-Data":
        step_level_beam_search_mistral_prm(
            dataset=args.dataset,
            split=args.split,
            model_name=args.model,
            prm_name=args.prm_model,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            output_path=args.output,
        )
    else:
        raise NotImplementedError("Only NAIT PRM model is supported currently.")

# client = AsyncOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=os.getenv("DASHSCOPE_BASE_URL"),
# )
# prm_name = "/root/autodl-tmp/checkpoint/qwen2.5-math-1.5b-prm"
# prm_model = AutoModelForCausalLM.from_pretrained(
#         prm_name, 
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True
#     ).to("cuda:0").eval()
# prm_tokenizer = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)
# pos_tag_id = prm_tokenizer.encode('[POS]')[-1]
# neg_tag_id = prm_tokenizer.encode('[NEG]')[-1]
# candidate_tokens = [pos_tag_id, neg_tag_id]


# prompt = SYSTEM_PROMPT + "\n\n"
# problem = "Given sets $M=\\{x|x+2\\geq 0\\},N=\\{x|x-1<0\\}$, find $M \\cap N$."
# prompt += problem + "\n"

# while True:
#     response = client.completions.create(
#         model="Qwen/Qwen2.5-Math-1.5B",
#         prompt=prompt,
#         max_tokens=512,
#         temperature=0.7,
#         stop=END_MARK,
#         n=BEAM_WIDTH,
#     )
#     steps = [choice.text for choice in response.choices]

#     # 将每个 prompt 拼上 step 文本，形成新的 prompt 交给 prm 评分
#     step_rewards = []
#     for step in steps:
#         step_text = step.strip()
#         input_ids = prm_tokenizer.encode(
#             prompt + step_text,
#             return_tensors="pt"
#         ).to(prm_model.device)
#         with torch.no_grad():
#             temp = prm_model(input_ids)
#             logits = prm_model(input_ids).logits
#         candidate_logits = logits[:, :, candidate_tokens]
#         probabilities = F.softmax(candidate_logits, dim=-1)
#         step_reward = probabilities[:, -1, 0].item()  # 取正向标签的概率作为奖励
#         step_rewards.append(step_reward)
    
#     # 选择奖励最高的 step
#     best_step_idx = step_rewards.index(max(step_rewards))
#     best_step = steps[best_step_idx].strip()
#     prompt += best_step + END_MARK
#     print(f"Selected Step: {best_step} [Reward: {step_rewards[best_step_idx]:.4f}]")

#     if check_is_finished(best_step):
#         print("Final reasoning process:")
#         print(prompt)
#         break
    