# 读取 json 文件，并计算每个 response 的得分，保存为新的 json 文件
# 下面是一个示例，你需要利用 prm 计算每条 response 的得分，并将得分保存为新的字段 "prm_scores"
#   {
#     "problem": "Given sets $M=\\{x|x+2\\geq 0\\},N=\\{x|x-1<0\\}$, find $M \\cap N$.",
#     "responses": [
#       "To find the intersection of the sets \\(M\\) and \\(N\\), we first need to determine the elements of each set individually.\n\nThe set \\(M\\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSolving the inequality \\(x + 2 \\geq 0\\), we get:\n\\[ x \\geq -2 \\]\nSo, the set \\(M\\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\nor in interval notation:\n\\[ M = [-2, \\infty) \\]\n\nThe set \\(N\\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nSolving the inequality \\(x - 1 < 0\\), we get:\n\\[ x < 1 \\]\nSo, the set \\(N\\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\nor in interval notation:\n\\[ N = (-\\infty, 1) \\]\n\nNow, we need to find the intersection of the sets \\(M\\) and \\(N\\), which is the set of all elements that are in both \\(M\\) and \\(N\\). In other words, we need to find the values of \\(x\\) that satisfy both \\(x \\geq -2\\) and \\(x < 1\\). This gives us the interval:\n\\[ M \\cap N = [-2, 1) \\]\n\nTherefore, the intersection of the sets \\(M\\) and \\(N\\) is:\n\\[ \\boxed{[-2, 1)} \\]"
#     ],
#     "answer": "$\\{x|-2\\leq x < 1\\}$",
#     "steps": [
#       [
#         "To find the intersection of the sets \\(M\\) and \\(N\\), we first need to determine the elements of each set individually.",
#         "The set \\(M\\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSolving the inequality \\(x + 2 \\geq 0\\), we get:\n\\[ x \\geq -2 \\]\nSo, the set \\(M\\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\nor in interval notation:\n\\[ M = [-2, \\infty) \\]",
#         "The set \\(N\\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nSolving the inequality \\(x - 1 < 0\\), we get:\n\\[ x < 1 \\]\nSo, the set \\(N\\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\nor in interval notation:\n\\[ N = (-\\infty, 1) \\]",
#         "Now, we need to find the intersection of the sets \\(M\\) and \\(N\\), which is the set of all elements that are in both \\(M\\) and \\(N\\). In other words, we need to find the values of \\(x\\) that satisfy both \\(x \\geq -2\\) and \\(x < 1\\). This gives us the interval:\n\\[ M \\cap N = [-2, 1) \\]",
#         "Therefore, the intersection of the sets \\(M\\) and \\(N\\) is:\n\\[ \\boxed{[-2, 1)} \\]"
#       ]
#     ]
#   },

import os
os.environ['HF_HOME'] = '/root/autodl-tmp/hf-mirror'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForAudioClassification, AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import json

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
    model = AutoModelForAudioClassification.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    # 计算每个 response 的 prm 得分
    for item in tqdm(data, desc="Calculating PRM scores"):
        prm_scores = []
        for steps in item["steps"]:
            # 构造输入
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

            # 前向计算
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            # 计算 step rewards
            step_sep_id = tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_reward = make_step_rewards(outputs[0], token_masks)
            prm_scores.append(step_reward[0])  # 取第一个样本的得分
        item[model_name] = prm_scores



model_name = "Qwen/Qwen2.5-Math-1.5B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()


data = {
    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    "query": "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
    "response": [
      "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
      "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
      "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
      "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."
    ]
}

messages = [
    {"role": "system", "content": data['system']},
    {"role": "user", "content": data['query']},
    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
]
conversation_str = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pt", 
).to(model.device)

outputs = model(input_ids=input_ids)

step_sep_id = tokenizer.encode("<extra_0>")[0]
token_masks = (input_ids == step_sep_id)
step_reward = make_step_rewards(outputs[0], token_masks)
print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PRM scores for responses")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    cal_qwen_prm_scores(args.input, args.output, args.model_name)