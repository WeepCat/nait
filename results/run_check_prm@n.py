# 读取 json 文件，根据每个 response 的 prm 得分的统计值（最大值、乘积、最小值、平均值、几何平均值），选择最佳响应
# 在 reference_file 文件中找到对应的评估结果，保存在一个新的 json 文件中
# 下面是一个示例
#   {
#     "problem": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "responses": [
#       "To determine how much Janet makes every day at the farmers' market from selling eggs, we need to follow these steps:\n\n1. **Calculate the total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Determine the number of eggs used for breakfast each day:**\n   Janet eats 3 eggs for breakfast each morning.\n   \n   Therefore, the number of eggs left after breakfast is:\n   \\[\n   16 - 3 = 13 \\text{ eggs}\n   \\]\n\n3. **Account for the eggs used for baking muffins:**\n   Janet uses 4 eggs for baking muffins each day.\n   \n   The number of eggs remaining after baking muffins is:\n   \\[\n   13 - 4 = 9 \\text{ eggs}\n   \\]\n\n4. **Calculate the revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \n   Therefore, the total revenue is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\n\nSo, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling her ducks' eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used for breakfast.\n3. Subtract the eggs used for breakfast and baked muffins from the total eggs laid to find out how many eggs are sold.\n4. Calculate the revenue from selling the remaining eggs.\n\nLet's go through each step in detail:\n\n1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Number of eggs used for breakfast:**\n   Janet eats 3 eggs for breakfast every morning.\n\n3. **Number of eggs baked into muffins:**\n   Janet bakes muffins using 4 eggs every day.\n\n4. **Calculate the number of eggs left after breakfast and baking:**\n   \\[\n   \\text{Eggs left} = \\text{Total eggs} - (\\text{Eggs for breakfast} + \\text{Eggs baked})\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Eggs left} = 16 - (3 + 4) = 16 - 7 = 9\n   \\]\n\n5. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \\[\n   \\text{Revenue} = \\text{Number of eggs left} \\times \\text{Price per egg}\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]\n\nTherefore, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid each day.\n2. Subtract the number of eggs eaten for breakfast.\n3. Subtract the number of eggs used for baking muffins.\n4. The remaining number of eggs is what Janet sells at the market.\n\nLet's break it down:\n\n1. Total eggs laid per day: \\( 16 \\) eggs.\n2. Eggs eaten for breakfast: \\( 3 \\) eggs.\n3. Eggs used for baking muffins: \\( 4 \\) eggs.\n\nFirst, we subtract the eggs eaten for breakfast from the total eggs:\n\\[ 16 - 3 = 13 \\]\n\nNext, we subtract the eggs used for baking muffins from the remaining eggs:\n\\[ 13 - 4 = 9 \\]\n\nThe number of eggs sold at the market is \\( 9 \\). Since Janet sells each egg for $2, we calculate her daily earnings:\n\\[ 9 \\text{ eggs} \\times 2 \\text{ dollars/egg} = 18 \\text{ dollars} \\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling her duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used or eaten per day.\n3. Find out how many eggs remain after accounting for those used or eaten.\n4. Calculate the revenue from selling the remaining eggs.\n\nLet's go through each step:\n\n1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Number of eggs used or eaten per day:**\n   - Janet eats 3 eggs for breakfast.\n   - She uses 4 eggs to bake muffins.\n   \n   Total eggs used/eaten = Eggs eaten for breakfast + Eggs used for muffins\n   \\[\n   \\text{Total eggs used/eaten} = 3 + 4 = 7\n   \\]\n\n3. **Number of eggs remaining:**\n   To find out how many eggs remain after Janet's usage, subtract the total eggs used/eaten from the total eggs laid:\n   \\[\n   \\text{Remaining eggs} = 16 - 7 = 9\n   \\]\n\n4. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg:\n   \\[\n   \\text{Revenue} = \\text{Remaining eggs} \\times \\text{Price per egg}\n   \\]\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#     ],
#     "answer": "18",
#     "steps": [
#       [
#         "To determine how much Janet makes every day at the farmers' market from selling eggs, we need to follow these steps:",
#         "1. **Calculate the total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.",
#         "2. **Determine the number of eggs used for breakfast each day:**\n   Janet eats 3 eggs for breakfast each morning.\n   \n   Therefore, the number of eggs left after breakfast is:\n   \\[\n   16 - 3 = 13 \\text{ eggs}\n   \\]",
#         "3. **Account for the eggs used for baking muffins:**\n   Janet uses 4 eggs for baking muffins each day.\n   \n   The number of eggs remaining after baking muffins is:\n   \\[\n   13 - 4 = 9 \\text{ eggs}\n   \\]",
#         "4. **Calculate the revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \n   Therefore, the total revenue is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]",
#         "So, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#       ],
#       [
#         "To determine how much Janet makes every day from selling her ducks' eggs at the farmers' market, we need to follow these steps:",
#         "1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used for breakfast.\n3. Subtract the eggs used for breakfast and baked muffins from the total eggs laid to find out how many eggs are sold.\n4. Calculate the revenue from selling the remaining eggs.",
#         "Let's go through each step in detail:",
#         "1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.",
#         "2. **Number of eggs used for breakfast:**\n   Janet eats 3 eggs for breakfast every morning.",
#         "3. **Number of eggs baked into muffins:**\n   Janet bakes muffins using 4 eggs every day.",
#         "4. **Calculate the number of eggs left after breakfast and baking:**\n   \\[\n   \\text{Eggs left} = \\text{Total eggs} - (\\text{Eggs for breakfast} + \\text{Eggs baked})\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Eggs left} = 16 - (3 + 4) = 16 - 7 = 9\n   \\]",
#         "5. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \\[\n   \\text{Revenue} = \\text{Number of eggs left} \\times \\text{Price per egg}\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]",
#         "Therefore, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#       ],
#       [
#         "To determine how much Janet makes every day from selling eggs at the farmers' market, we need to follow these steps:",
#         "1. Calculate the total number of eggs laid each day.\n2. Subtract the number of eggs eaten for breakfast.\n3. Subtract the number of eggs used for baking muffins.\n4. The remaining number of eggs is what Janet sells at the market.",
#         "Let's break it down:",
#         "1. Total eggs laid per day: \\( 16 \\) eggs.\n2. Eggs eaten for breakfast: \\( 3 \\) eggs.\n3. Eggs used for baking muffins: \\( 4 \\) eggs.",
#         "First, we subtract the eggs eaten for breakfast from the total eggs:\n\\[ 16 - 3 = 13 \\]",
#         "Next, we subtract the eggs used for baking muffins from the remaining eggs:\n\\[ 13 - 4 = 9 \\]",
#         "The number of eggs sold at the market is \\( 9 \\). Since Janet sells each egg for $2, we calculate her daily earnings:\n\\[ 9 \\text{ eggs} \\times 2 \\text{ dollars/egg} = 18 \\text{ dollars} \\]",
#         "Thus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#       ],
#       [
#         "To determine how much Janet makes every day from selling her duck eggs at the farmers' market, we need to follow these steps:",
#         "1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used or eaten per day.\n3. Find out how many eggs remain after accounting for those used or eaten.\n4. Calculate the revenue from selling the remaining eggs.",
#         "Let's go through each step:",
#         "1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.",
#         "2. **Number of eggs used or eaten per day:**\n   - Janet eats 3 eggs for breakfast.\n   - She uses 4 eggs to bake muffins.\n   \n   Total eggs used/eaten = Eggs eaten for breakfast + Eggs used for muffins\n   \\[\n   \\text{Total eggs used/eaten} = 3 + 4 = 7\n   \\]",
#         "3. **Number of eggs remaining:**\n   To find out how many eggs remain after Janet's usage, subtract the total eggs used/eaten from the total eggs laid:\n   \\[\n   \\text{Remaining eggs} = 16 - 7 = 9\n   \\]",
#         "4. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg:\n   \\[\n   \\text{Revenue} = \\text{Remaining eggs} \\times \\text{Price per egg}\n   \\]\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]",
#         "Thus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#       ]
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_scores": [
#       [
#         0.9921875,
#         1.0,
#         0.99609375,
#         0.99609375,
#         1.0,
#         1.0
#       ],
#       [
#         0.9921875,
#         0.96875,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.99609375,
#         1.0,
#         1.0
#       ],
#       [
#         0.99609375,
#         0.98046875,
#         0.99609375,
#         0.99609375,
#         0.99609375,
#         1.0,
#         1.0,
#         1.0
#       ],
#       [
#         0.99609375,
#         0.99609375,
#         1.0,
#         1.0,
#         0.99609375,
#         1.0,
#         1.0,
#         1.0
#       ]
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_mean": [
#       0.9973958333333334,
#       0.9952256944444444,
#       0.99560546875,
#       0.99853515625
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_max": [
#       1.0,
#       1.0,
#       1.0,
#       1.0
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_min": [
#       0.9921875,
#       0.96875,
#       0.98046875,
#       0.99609375
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_product": [
#       0.984451174736023,
#       0.9574270248413086,
#       0.9652384519577026,
#       0.9883269667625427
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_geom_mean": [
#       0.9973915815353394,
#       0.995177686214447,
#       0.9955872297286987,
#       0.9985333681106567
#     ]
#   }
# reference_file 文件的一个示例
#   {
#     "problem": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "responses": [
#       "To determine how much Janet makes every day at the farmers' market from selling eggs, we need to follow these steps:\n\n1. **Calculate the total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Determine the number of eggs used for breakfast each day:**\n   Janet eats 3 eggs for breakfast each morning.\n   \n   Therefore, the number of eggs left after breakfast is:\n   \\[\n   16 - 3 = 13 \\text{ eggs}\n   \\]\n\n3. **Account for the eggs used for baking muffins:**\n   Janet uses 4 eggs for baking muffins each day.\n   \n   The number of eggs remaining after baking muffins is:\n   \\[\n   13 - 4 = 9 \\text{ eggs}\n   \\]\n\n4. **Calculate the revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \n   Therefore, the total revenue is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\n\nSo, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling her ducks' eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used for breakfast.\n3. Subtract the eggs used for breakfast and baked muffins from the total eggs laid to find out how many eggs are sold.\n4. Calculate the revenue from selling the remaining eggs.\n\nLet's go through each step in detail:\n\n1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Number of eggs used for breakfast:**\n   Janet eats 3 eggs for breakfast every morning.\n\n3. **Number of eggs baked into muffins:**\n   Janet bakes muffins using 4 eggs every day.\n\n4. **Calculate the number of eggs left after breakfast and baking:**\n   \\[\n   \\text{Eggs left} = \\text{Total eggs} - (\\text{Eggs for breakfast} + \\text{Eggs baked})\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Eggs left} = 16 - (3 + 4) = 16 - 7 = 9\n   \\]\n\n5. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg.\n   \\[\n   \\text{Revenue} = \\text{Number of eggs left} \\times \\text{Price per egg}\n   \\]\n   Substituting the values:\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]\n\nTherefore, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid each day.\n2. Subtract the number of eggs eaten for breakfast.\n3. Subtract the number of eggs used for baking muffins.\n4. The remaining number of eggs is what Janet sells at the market.\n\nLet's break it down:\n\n1. Total eggs laid per day: \\( 16 \\) eggs.\n2. Eggs eaten for breakfast: \\( 3 \\) eggs.\n3. Eggs used for baking muffins: \\( 4 \\) eggs.\n\nFirst, we subtract the eggs eaten for breakfast from the total eggs:\n\\[ 16 - 3 = 13 \\]\n\nNext, we subtract the eggs used for baking muffins from the remaining eggs:\n\\[ 13 - 4 = 9 \\]\n\nThe number of eggs sold at the market is \\( 9 \\). Since Janet sells each egg for $2, we calculate her daily earnings:\n\\[ 9 \\text{ eggs} \\times 2 \\text{ dollars/egg} = 18 \\text{ dollars} \\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market.",
#       "To determine how much Janet makes every day from selling her duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid per day.\n2. Determine the number of eggs used or eaten per day.\n3. Find out how many eggs remain after accounting for those used or eaten.\n4. Calculate the revenue from selling the remaining eggs.\n\nLet's go through each step:\n\n1. **Total number of eggs laid per day:**\n   Janet's ducks lay 16 eggs per day.\n\n2. **Number of eggs used or eaten per day:**\n   - Janet eats 3 eggs for breakfast.\n   - She uses 4 eggs to bake muffins.\n   \n   Total eggs used/eaten = Eggs eaten for breakfast + Eggs used for muffins\n   \\[\n   \\text{Total eggs used/eaten} = 3 + 4 = 7\n   \\]\n\n3. **Number of eggs remaining:**\n   To find out how many eggs remain after Janet's usage, subtract the total eggs used/eaten from the total eggs laid:\n   \\[\n   \\text{Remaining eggs} = 16 - 7 = 9\n   \\]\n\n4. **Revenue from selling the remaining eggs:**\n   Janet sells the remaining eggs at $2 per egg:\n   \\[\n   \\text{Revenue} = \\text{Remaining eggs} \\times \\text{Price per egg}\n   \\]\n   \\[\n   \\text{Revenue} = 9 \\times 2 = 18\n   \\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
#     ],
#     "answer": "18",
#     "labels": [
#       1,
#       1,
#       1,
#       1
#     ]
#   }

# 你需要根据聚合指标找到最佳响应，之后在 reference_file 文件中找到对应的评估结果，保存在一个新的 json 文件中
import json
import os
import argparse
from tqdm import tqdm


def check_prm_scores(input_path, reference_path, output_path, model_name):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(reference_path, "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    reference_dict = {item["problem"]: item for item in reference_data}
    base_model_name = os.path.basename(model_name)
    
    # 分别根据均值 / 最小值 / 平均值 / 几何平均值选择最佳响应
    for item in tqdm(data):
        prm_mean_key = f"{base_model_name}_prm_mean"
        prm_min_key = f"{base_model_name}_prm_min"
        prm_max_key = f"{base_model_name}_prm_max"
        prm_product_key = f"{base_model_name}_prm_product"
        prm_sum_key = f"{base_model_name}_prm_sum"

        prm_means = item[prm_mean_key]
        prm_mins = item[prm_min_key]
        prm_sums = item[prm_sum_key]

        best_mean_idx = prm_means.index(max(prm_means))
        best_min_idx = prm_mins.index(max(prm_mins))
        best_max_idx = item[prm_max_key].index(max(item[prm_max_key]))
        best_product_idx = item[prm_product_key].index(max(item[prm_product_key]))
        best_sum_idx = prm_sums.index(max(prm_sums))

        best_mean_response = item["responses"][best_mean_idx]
        best_min_response = item["responses"][best_min_idx]
        best_max_response = item["responses"][best_max_idx]
        best_product_response = item["responses"][best_product_idx]
        best_sum_response = item["responses"][best_sum_idx]
        
        if "EurusPRM" not in base_model_name:
            prm_geom_mean_key = f"{base_model_name}_prm_geom_mean"
            prm_geom_means = item[prm_geom_mean_key]
            best_geom_mean_idx = prm_geom_means.index(max(prm_geom_means))
            best_geom_mean_response = item["responses"][best_geom_mean_idx]

        problem = item["problem"]
        ref_item = reference_dict.get(problem)

        # 从 ref_item 中找到 response 的 index, 利用 index 获取评估结果
        if ref_item:
            ref_mean_response_idx = ref_item["responses"].index(best_mean_response) if best_mean_response in ref_item["responses"] else -1
            ref_min_response_idx = ref_item["responses"].index(best_min_response) if best_min_response in ref_item["responses"] else -1
            ref_max_response_idx = ref_item["responses"].index(best_max_response) if best_max_response in ref_item["responses"] else -1
            ref_product_response_idx = ref_item["responses"].index(best_product_response) if best_product_response in ref_item["responses"] else -1
            ref_sum_response_idx = ref_item["responses"].index(best_sum_response) if best_sum_response in ref_item["responses"] else -1

            item[f"{base_model_name}_prm_mean_label"] = ref_item["labels"][ref_mean_response_idx] if ref_mean_response_idx != -1 else None
            item[f"{base_model_name}_prm_min_label"] = ref_item["labels"][ref_min_response_idx] if ref_min_response_idx != -1 else None
            item[f"{base_model_name}_prm_max_label"] = ref_item["labels"][ref_max_response_idx] if ref_max_response_idx != -1 else None
            item[f"{base_model_name}_prm_product_label"] = ref_item["labels"][ref_product_response_idx] if ref_product_response_idx != -1 else None
            item[f"{base_model_name}_prm_sum_label"] = ref_item["labels"][ref_sum_response_idx] if ref_sum_response_idx != -1 else None

            if "EurusPRM" not in base_model_name:
                ref_geom_mean_response_idx = ref_item["responses"].index(best_geom_mean_response) if best_geom_mean_response in ref_item["responses"] else -1
                item[f"{base_model_name}_prm_geom_mean_label"] = ref_item["labels"][ref_geom_mean_response_idx] if ref_geom_mean_response_idx != -1 else None
                
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PRM scores for responses")
    parser.add_argument("--filename", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--reference_file", type=str, required=True, help="Reference JSON file path")
    args = parser.parse_args()
    check_prm_scores(args.filename, args.reference_file, args.output, args.model_name)
