# 读取 json 文件，并计算每个 response 的得分，保存为新的 json 文件
# 下面是一个示例，你需要利用 prm 计算每条 response 的得分，并将得分保存为新的字段 "prm_scores"
#   {
#     "problem": "Given sets $M=\\{x|x+2\\geq 0\\},N=\\{x|x-1<0\\}$, find $M \\cap N$.",
#     "responses": [
#       "To find the intersection of the sets \\(M\\) and \\(N\\), we first need to determine the elements of each set individually.\n\nThe set \\(M\\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSolving the inequality \\(x + 2 \\geq 0\\), we get:\n\\[ x \\geq -2 \\]\nSo, the set \\(M\\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\nor in interval notation:\n\\[ M = [-2, \\infty) \\]\n\nThe set \\(N\\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nSolving the inequality \\(x - 1 < 0\\), we get:\n\\[ x < 1 \\]\nSo, the set \\(N\\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\nor in interval notation:\n\\[ N = (-\\infty, 1) \\]\n\nNow, we need to find the intersection of the sets \\(M\\) and \\(N\\), which is the set of all elements that are in both \\(M\\) and \\(N\\).In other words, we need to find the values of \\(x\\) that satisfy both \\(x \\geq -2\\) and \\(x < 1\\).This gives us the interval:\n\\[ M \\cap N = [-2, 1) \\]\n\nTherefore, the intersection of the sets \\(M\\) and \\(N\\) is:\n\\[ \\boxed{[-2, 1)} \\]"
#     ],
#     "answer": "$\\{x|-2\\leq x < 1\\}$",
#     "steps": [
#       [
#         "To find the intersection of the sets \\(M\\) and \\(N\\), we first need to determine the elements of each set individually.",
#         "The set \\(M\\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSolving the inequality \\(x + 2 \\geq 0\\), we get:\n\\[ x \\geq -2 \\]\nSo, the set \\(M\\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\nor in interval notation:\n\\[ M = [-2, \\infty) \\]",
#         "The set \\(N\\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nSolving the inequality \\(x - 1 < 0\\), we get:\n\\[ x < 1 \\]\nSo, the set \\(N\\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\nor in interval notation:\n\\[ N = (-\\infty, 1) \\]",
#         "Now, we need to find the intersection of the sets \\(M\\) and \\(N\\), which is the set of all elements that are in both \\(M\\) and \\(N\\).In other words, we need to find the values of \\(x\\) that satisfy both \\(x \\geq -2\\) and \\(x < 1\\).This gives us the interval:\n\\[ M \\cap N = [-2, 1) \\]",
#         "Therefore, the intersection of the sets \\(M\\) and \\(N\\) is:\n\\[ \\boxed{[-2, 1)} \\]"
#       ]
#     ]
#   },

import os
os.environ['HF_HOME'] = '/root/autodl-tmp/hf-mirror'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import json


def cal_nait_prm_scores(input_path, output_path, model_name):
    # pass  # Placeholder for NAIT PRM score calculation implementation
    def make_step_rewards(logits, pos_token_mask, candidate_tokens):
        """
        从 logits 中提取每个步骤的奖励分数
        
        Args:
            logits:  模型输出的 logits，shape:  (batch_size, seq_len, vocab_size)
            pos_token_mask: 标记 [POS] token **前一个位置**的 mask，shape:  (batch_size, seq_len)
                            即该位置的输出用于预测 [POS]
            candidate_tokens: [pos_tag_id, neg_tag_id]
        
        Returns:
            list of list: 每个样本的步骤分数列表
        """
        # 获取候选 token 的 logits
        candidate_logits = logits[:, :, candidate_tokens]  # (batch_size, seq_len, 2)
        
        # 计算 softmax 得到概率
        probabilities = F.softmax(candidate_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        # 提取 [POS] token 的概率（索引 0，因为 candidate_tokens = [pos_tag_id, neg_tag_id]）
        pos_probabilities = probabilities[:, :, 0]  # (batch_size, seq_len)
        
        # 只保留 mask 位置的概率
        masked_probs = pos_probabilities * pos_token_mask  # (batch_size, seq_len)
        
        # 提取每个样本的非零元素（即步骤分数）
        all_scores_res = []
        for i in range(masked_probs.size(0)):
            sample_probs = masked_probs[i]  # (seq_len,)
            # 提取非零位置的分数
            step_scores = sample_probs[sample_probs != 0].cpu().tolist()
            all_scores_res.append(step_scores)
        
        return all_scores_res
        
    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    ).eval()

    # 获取 [POS]/[NEG] token ids
    pos_tag_id = tokenizer.encode('[POS]')[-1]
    neg_tag_id = tokenizer.encode('[NEG]')[-1]
    candidate_tokens = [pos_tag_id, neg_tag_id]

    base_model_name = os.path.basename(model_name)

    # 计算每个 response 的 prm 得分
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        problem = item["problem"]
        for steps in item["steps"]:
            prompt_str = ""
            prompt_str += problem
            step_rewards = []
            for step in steps: # 在每个 step 后面添加 [POS]，计算该位置预测 [POS] 的概率作为奖励
                prompt_str += "\n" + step + " [POS]"
            # 一次性编码整个对话
            input_ids = tokenizer.encode(
                prompt_str, 
                return_tensors="pt"
            ).to(model.device)
            
            # 创建 mask：标记 [POS] token 前一个位置
            # 模板格式:  ... [content_tokens] [POS]
            # 我们需要标记 [POS] 前面的那个位置，该位置的输出预测
            pos_token_mask = torch.zeros_like(input_ids, dtype=torch.float)
            # 找到所有 [POS] token 的位置
            pos_positions = (input_ids == pos_tag_id).nonzero(as_tuple=True)
            # 对于每个 [POS] token，标记其前一个位置
            for batch_idx, pos in zip(pos_positions[0], pos_positions[1]):
                if pos > 0:  # 确保不越界
                    pos_token_mask[batch_idx, pos - 1] = 1.0
            # 只进行一次前向传播
            with torch.no_grad():
                logits = model(input_ids).logits
            # 使用 make_step_rewards 提取分数
            step_rewards = make_step_rewards(logits, pos_token_mask, candidate_tokens)
            prm_scores.append(step_rewards[0])  # 只有一个样本
        item[f"{base_model_name}_prm_scores"] = prm_scores
        torch.cuda.empty_cache()

    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cal_mistral_prm_scores(input_path, output_path, model_name):
    
    def make_step_rewards(logits, plus_token_mask, candidate_tokens):
        """
        从 logits 中提取每个步骤的奖励分数
        
        Args:
            logits:  模型输出的 logits，shape:  (batch_size, seq_len, vocab_size)
            plus_token_mask: 标记 + token **前一个位置**的 mask，shape:  (batch_size, seq_len)
                            即该位置的输出用于预测 + token
            candidate_tokens: [plus_tag_id, minus_tag_id]
        
        Returns:
            list of list: 每个样本的步骤分数列表
        """
        # 获取候选 token 的 logits
        candidate_logits = logits[:, :, candidate_tokens]  # (batch_size, seq_len, 2)
        
        # 计算 softmax 得到概率
        probabilities = F.softmax(candidate_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        # 提取 + token 的概率（索引 0，因为 candidate_tokens = [plus_tag_id, minus_tag_id]）
        plus_probabilities = probabilities[:, :, 0]  # (batch_size, seq_len)
        
        # 只保留 mask 位置的概率
        masked_probs = plus_probabilities * plus_token_mask  # (batch_size, seq_len)
        
        # 提取每个样本的非零元素（即步骤分数）
        all_scores_res = []
        for i in range(masked_probs.size(0)):
            sample_probs = masked_probs[i]  # (seq_len,)
            # 提取非零位置的分数
            step_scores = sample_probs[sample_probs != 0].cpu().tolist()
            all_scores_res.append(step_scores)
        
        return all_scores_res
    
    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # 获取 +/- token ids
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id, minus_tag_id]

    base_model_name = os.path.basename(model_name)

    # 计算每个 response 的 prm 得分
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        problem = item["problem"]
        
        for steps in item["steps"]:
            # 构建完整的对话，一次性包含所有步骤
            conversation = []
            for k in range(len(steps)):
                if k == 0:
                    text = problem + " " + steps[0]
                else:
                    text = steps[k]
                conversation.append({"content": text, "role": "user"})
                conversation.append({"content": "+", "role": "assistant"})
            
            # 一次性编码整个对话
            input_ids = tokenizer.apply_chat_template(
                conversation, 
                return_tensors="pt"
            ).to(model.device)
            
            # 创建 mask：标记 + token 前一个位置
            # 模板格式:  ... [content_tokens] [+] [EOS]
            # 我们需要标记 [+] 前面的那个位置，该位置的输出预测 [+]
            plus_token_mask = torch.zeros_like(input_ids, dtype=torch.float)
            
            # 找到所有 + token 的位置
            plus_positions = (input_ids == plus_tag_id).nonzero(as_tuple=True)
            
            # 对于每个 + token，标记其前一个位置
            for batch_idx, pos in zip(plus_positions[0], plus_positions[1]):
                if pos > 0:  # 确保不越界
                    plus_token_mask[batch_idx, pos - 1] = 1.0
            
            # 只进行一次前向传播
            with torch.no_grad():
                logits = model(input_ids).logits
            
            # 使用 make_step_rewards 提取分数
            step_rewards = make_step_rewards(logits, plus_token_mask, candidate_tokens)
            prm_scores.append(step_rewards[0])  # 只有一个样本
        
        item[f"{base_model_name}_prm_scores"] = prm_scores
        torch.cuda.empty_cache()
    
    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def cal_qwen_prm_scores(input_path, output_path, model_name):

    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    base_model_name = os.path.basename(model_name)
    # 计算每个 response 的 prm 得分
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        for steps in item["steps"]:
            conversation_str = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": item["problem"]},
                    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
                ],
                tokenize=False, 
                add_generation_prompt=False
            )
            input_ids = tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
            ).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            step_sep_id = tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_reward = make_step_rewards(outputs[0], token_masks)
            prm_scores.append(step_reward[0])
        item[f"{base_model_name}_prm_scores"] = prm_scores
        torch.cuda.empty_cache()

    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cal_shepherd_prm_scores(input_path, output_path, model_name):
    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    
    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]

    base_model_name = os.path.basename(model_name)
    # 计算每个 response 的 prm 得分
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        problem = item["problem"]
        for steps in item["steps"]:
            prompt_str = ""
            prompt_str += problem + " "
            step_rewards = []
            for i, step in enumerate(steps): # 在每个 step 后面添加 [POS]，计算该位置预测 [POS] 的概率作为奖励
                if i != len(steps) - 1:
                    prompt_str += step + f" {step_tag}\n"
                else:
                    prompt_str += step + f" {step_tag}"
            input_id = torch.tensor([tokenizer.encode(prompt_str)]).to(model.device)

            with torch.no_grad():
                logits = model(input_id).logits[:, :, candidate_tokens]
                scores = logits.softmax(dim=-1)[:, :, 0] 
                step_scores = scores[input_id == step_tag_id]
                prm_scores.append(step_scores.cpu().tolist())
        item[f"{base_model_name}_prm_scores"] = prm_scores
        torch.cuda.empty_cache()
    
    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def cal_eurus_prm_scores(input_path, output_path, model_name):
    def get_logps(model, inputs):
        logits = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).logits
        labels = inputs['labels'][:, 1:].clone().long()
        logits = logits[:, :-1, :]
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return per_token_logps

    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 加载模型和分词器
    coef = 0.001
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct', device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    
    base_model_name = os.path.basename(model_name)
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        problem = item["problem"]
        for steps in item["steps"]:
            input_ids = tokenizer.apply_chat_template([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": "\n\n".join(steps)},
            ], tokenize=True, add_generation_prompt=False, return_tensors='pt').to(model.device)
            attention_mask = input_ids != tokenizer.pad_token_id
            step_last_tokens = []

            for step_num in range(0, len(steps) + 1):
                conv = tokenizer.apply_chat_template([
                    {"role":"user", "content": problem},
                    {"role":"assistant", "content":"\n\n".join(steps[:step_num])},
                ], tokenize=False, add_generation_prompt=False)
                conv = conv.strip()
                if step_num != 0 and step_num != len(steps):
                    conv += '\n\n'
                currect_ids = tokenizer.encode(conv, add_special_tokens=False)
                step_last_tokens.append(len(currect_ids) - 2)
            

            inputs = {'input_ids': input_ids,'attention_mask': attention_mask, 'labels': input_ids}
            label_mask = torch.tensor([[0] * step_last_tokens[0] + [1] * (input_ids.shape[-1] - step_last_tokens[0])]).to(model.device)
            step_last_tokens = torch.tensor([step_last_tokens]).to(model.device)

            with torch.no_grad():
                per_token_logps = get_logps(model, inputs)
                ref_per_token_logps = get_logps(ref_model, inputs)

            raw_reward = per_token_logps - ref_per_token_logps
            step_scores = coef * raw_reward * label_mask[:, 1:]
            step_scores = step_scores.cumsum(-1)
            step_scores = step_scores.gather(dim=-1, index=step_last_tokens[:, 1:])
            prm_scores.append(step_scores.squeeze(0).cpu().tolist())
            
        item[f"{base_model_name}_prm_scores"] = prm_scores
        torch.cuda.empty_cache()
    
    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PRM scores for responses")
    parser.add_argument("--filename", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    if args.model_name == "Qwen/Qwen2.5-Math-PRM-7B":
        cal_qwen_prm_scores(args.filename, args.output, args.model_name)
    elif args.model_name == "RLHFlow/Llama3.1-8B-PRM-Mistral-Data" or args.model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        cal_mistral_prm_scores(args.filename, args.output, args.model_name)
    elif args.model_name == "peiyi9979/math-shepherd-mistral-7b-prm":
        cal_shepherd_prm_scores(args.filename, args.output, args.model_name)
    elif "EurusPRM" in args.model_name:
        cal_eurus_prm_scores(args.filename, args.output, args.model_name)
    elif "nait" in args.model_name.lower():
        cal_nait_prm_scores(args.filename, args.output, args.model_name)
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported yet.")

# def make_step_rewards(logits, token_masks):
#     probabilities = F.softmax(logits, dim=-1)
#     probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
#     all_scores_res = []
#     for i in range(probabilities.size(0)):
#         sample = probabilities[i] # seq_len, num_labels
#         positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
#         non_zero_elements_list = positive_probs.cpu().tolist()
#         all_scores_res.append(non_zero_elements_list)
#     return all_scores_res

# model_name = "Qwen/Qwen2.5-Math-PRM-7B"
# device = "auto"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     model_name, 
#     device_map=device, 
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# ).eval()

# data = {
#     "system": "Please reason step by step, and put your final answer within \\boxed{}.",
#     "query": "Sue lives in a fun neighborhood. One weekend, the neighbors decided to play a prank on Sue. On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard. On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard. Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection.At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
#     "response": [
#       "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps.First, on Friday, the neighbors start with 18 pink plastic flamingos.",
#       "On Saturday, they take back one third of the flamingos.Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back.So, they have (18 - 6 = 12) flamingos left in their possession.Then, they paint these 6 flamingos white and put them back out on Sue's front yard.Now, Sue has the original 12 pink flamingos plus the 6 new white ones.Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
#       "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard.By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
#       "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30).Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos.The answer is (\\boxed{30})."
#     ]
# }

# messages = [
#     {"role": "system", "content": data['system']},
#     {"role": "user", "content": data['query']},
#     {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
# ]
# conversation_str = tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=False
# )

# input_ids = tokenizer.encode(
#     conversation_str, 
#     return_tensors="pt", 
# ).to(model.device)

# outputs = model(input_ids=input_ids)

# step_sep_id = tokenizer.encode("<extra_0>")[0]
# token_masks = (input_ids == step_sep_id)
# step_reward = make_step_rewards(outputs[0], token_masks)
# print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]
