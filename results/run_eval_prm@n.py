# 读取一个 json 文件，根据 labels 字段的结果，统计 prm@n 的比例
# 下面是一个示例，你需要根据 mean_label, min_label, max_label, product_label, geom_mean_label 字段的数值结果来统计 prm@n, 其中 1 表示正确, -1 表示错误
#   {
#     "problem": "Given sets $M=\\{x|x+2\\geq 0\\},N=\\{x|x-1<0\\}$, find $M \\cap N$.",
#     "responses": [
#       "To find the intersection \\( M \\cap N \\) of the sets \\( M = \\{ x \\mid x + 2 \\geq 0 \\} \\) and \\( N = \\{ x \\mid x - 1 < 0 \\} \\), we need to determine the elements that satisfy both conditions simultaneously.\n\nStep 1: Solve the inequality for set \\( M \\).\n\\[ x + 2 \\geq 0 \\]\nSubtract 2 from both sides:\n\\[ x \\geq -2 \\]\nSo, set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\n\nStep 2: Solve the inequality for set \\( N \\).\n\\[ x - 1 < 0 \\]\nAdd 1 to both sides:\n\\[ x < 1 \\]\nSo, set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\n\nStep 3: Find the intersection \\( M \\cap N \\).\nThe intersection of two sets consists of all elements that are in both sets. Therefore, we need to find the values of \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\). This can be written as:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq x < 1 \\} \\]\n\nIn interval notation, this is:\n\\[ M \\cap N = [-2, 1) \\]\n\nThus, the final answer is:\n\\[\n\\boxed{[-2, 1)}\n\\]",
#       "To find the intersection \\( M \\cap N \\) of the sets \\( M = \\{ x | x + 2 \\geq 0 \\} \\) and \\( N = \\{ x | x - 1 < 0 \\} \\), we need to determine the elements that satisfy both conditions simultaneously. Let's solve each inequality separately and then find their intersection.\n\n### Step 1: Solve the inequality for set \\( M \\)\nThe inequality defining set \\( M \\) is:\n\\[ x + 2 \\geq 0 \\]\n\nSubtract 2 from both sides to isolate \\( x \\):\n\\[ x \\geq -2 \\]\n\nSo, set \\( M \\) can be written as:\n\\[ M = \\{ x | x \\geq -2 \\} \\]\n\n### Step 2: Solve the inequality for set \\( N \\)\nThe inequality defining set \\( N \\) is:\n\\[ x - 1 < 0 \\]\n\nAdd 1 to both sides to isolate \\( x \\):\n\\[ x < 1 \\]\n\nSo, set \\( N \\) can be written as:\n\\[ N = \\{ x | x < 1 \\} \\]\n\n### Step 3: Find the intersection \\( M \\cap N \\)\nThe intersection \\( M \\cap N \\) consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).\n\nTo visualize this, we can represent the intervals on a number line:\n- Set \\( M \\): All \\( x \\) such that \\( x \\geq -2 \\) includes",
#       "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we first need to determine the elements in each set individually.\n\n1. **Determine Set \\( M \\):**\n   - The set \\( M \\) is defined as \\( M = \\{ x \\mid x + 2 \\geq 0 \\} \\).\n   - To solve the inequality \\( x + 2 \\geq 0 \\):\n     \\[\n     x + 2 \\geq 0 \\implies x \\geq -2\n     \\]\n   - Therefore, set \\( M \\) includes all real numbers \\( x \\) such that \\( x \\geq -2 \\). In interval notation, this is:\n     \\[\n     M = [-2, \\infty)\n     \\]\n\n2. **Determine Set \\( N \\):**\n   - The set \\( N \\) is defined as \\( N = \\{ x \\mid x - 1 < 0 \\} \\).\n   - To solve the inequality \\( x - 1 < 0 \\):\n     \\[\n     x - 1 < 0 \\implies x < 1\n     \\]\n   - Therefore, set \\( N \\) includes all real numbers \\( x \\) such that \\( x < 1 \\). In interval notation, this is:\n     \\[\n     N = (-\\infty, 1)\n     \\]\n\n3. **Find the Intersection \\( M \\cap N \\):**\n   - The intersection of two sets consists of all elements that are common to both",
#       "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we first need to determine what each set represents.\n\nThe set \\( M \\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nWe solve the inequality \\( x + 2 \\geq 0 \\):\n\\[ x + 2 \\geq 0 \\implies x \\geq -2 \\]\nSo, the set \\( M \\) is all real numbers greater than or equal to \\(-2\\). In interval notation, this is:\n\\[ M = [-2, \\infty) \\]\n\nThe set \\( N \\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nWe solve the inequality \\( x - 1 < 0 \\):\n\\[ x - 1 < 0 \\implies x < 1 \\]\nSo, the set \\( N \\) is all real numbers less than \\(1\\). In interval notation, this is:\n\\[ N = (-\\infty, 1) \\]\n\nNext, we need to find the intersection of these two sets. The intersection \\( M \\cap N \\) consists of all elements that are in both \\( M \\) and \\( N \\). Therefore, we need to find the overlap between the intervals \\([-2, \\infty)\\) and \\((- \\infty, 1)\\).\n\nThe interval \\([-2, \\infty)\\) includes all numbers from \\(-2\\) to infinity,"
#     ],
#     "answer": "$\\{x|-2\\leq x < 1\\}$",
#     "steps": [
#       [
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M = \\{ x \\mid x + 2 \\geq 0 \\} \\) and \\( N = \\{ x \\mid x - 1 < 0 \\} \\), we need to determine the elements that satisfy both conditions simultaneously.",
#         "Step 1: Solve the inequality for set \\( M \\).\n\\[ x + 2 \\geq 0 \\]\nSubtract 2 from both sides:\n\\[ x \\geq -2 \\]\nSo, set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]",
#         "Step 2: Solve the inequality for set \\( N \\).\n\\[ x - 1 < 0 \\]\nAdd 1 to both sides:\n\\[ x < 1 \\]\nSo, set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]",
#         "Step 3: Find the intersection \\( M \\cap N \\).\nThe intersection of two sets consists of all elements that are in both sets. Therefore, we need to find the values of \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\). This can be written as:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq x < 1 \\} \\]",
#         "In interval notation, this is:\n\\[ M \\cap N = [-2, 1) \\]",
#         "Thus, the final answer is:\n\\[\n\\boxed{[-2, 1)}\n\\]"
#       ],
#       [
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M = \\{ x | x + 2 \\geq 0 \\} \\) and \\( N = \\{ x | x - 1 < 0 \\} \\), we need to determine the elements that satisfy both conditions simultaneously. Let's solve each inequality separately and then find their intersection.",
#         "### Step 1: Solve the inequality for set \\( M \\)\nThe inequality defining set \\( M \\) is:\n\\[ x + 2 \\geq 0 \\]",
#         "Subtract 2 from both sides to isolate \\( x \\):\n\\[ x \\geq -2 \\]",
#         "So, set \\( M \\) can be written as:\n\\[ M = \\{ x | x \\geq -2 \\} \\]",
#         "### Step 2: Solve the inequality for set \\( N \\)\nThe inequality defining set \\( N \\) is:\n\\[ x - 1 < 0 \\]",
#         "Add 1 to both sides to isolate \\( x \\):\n\\[ x < 1 \\]",
#         "So, set \\( N \\) can be written as:\n\\[ N = \\{ x | x < 1 \\} \\]",
#         "### Step 3: Find the intersection \\( M \\cap N \\)\nThe intersection \\( M \\cap N \\) consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).",
#         "To visualize this, we can represent the intervals on a number line:\n- Set \\( M \\): All \\( x \\) such that \\( x \\geq -2 \\) includes"
#       ],
#       [
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we first need to determine the elements in each set individually.",
#         "1. **Determine Set \\( M \\):**\n   - The set \\( M \\) is defined as \\( M = \\{ x \\mid x + 2 \\geq 0 \\} \\).\n   - To solve the inequality \\( x + 2 \\geq 0 \\):\n     \\[\n     x + 2 \\geq 0 \\implies x \\geq -2\n     \\]\n   - Therefore, set \\( M \\) includes all real numbers \\( x \\) such that \\( x \\geq -2 \\). In interval notation, this is:\n     \\[\n     M = [-2, \\infty)\n     \\]",
#         "2. **Determine Set \\( N \\):**\n   - The set \\( N \\) is defined as \\( N = \\{ x \\mid x - 1 < 0 \\} \\).\n   - To solve the inequality \\( x - 1 < 0 \\):\n     \\[\n     x - 1 < 0 \\implies x < 1\n     \\]\n   - Therefore, set \\( N \\) includes all real numbers \\( x \\) such that \\( x < 1 \\). In interval notation, this is:\n     \\[\n     N = (-\\infty, 1)\n     \\]",
#         "3. **Find the Intersection \\( M \\cap N \\):**\n   - The intersection of two sets consists of all elements that are common to both"
#       ],
#       [
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we first need to determine what each set represents.",
#         "The set \\( M \\) is defined as:\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nWe solve the inequality \\( x + 2 \\geq 0 \\):\n\\[ x + 2 \\geq 0 \\implies x \\geq -2 \\]\nSo, the set \\( M \\) is all real numbers greater than or equal to \\(-2\\). In interval notation, this is:\n\\[ M = [-2, \\infty) \\]",
#         "The set \\( N \\) is defined as:\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nWe solve the inequality \\( x - 1 < 0 \\):\n\\[ x - 1 < 0 \\implies x < 1 \\]\nSo, the set \\( N \\) is all real numbers less than \\(1\\). In interval notation, this is:\n\\[ N = (-\\infty, 1) \\]",
#         "Next, we need to find the intersection of these two sets. The intersection \\( M \\cap N \\) consists of all elements that are in both \\( M \\) and \\( N \\). Therefore, we need to find the overlap between the intervals \\([-2, \\infty)\\) and \\((- \\infty, 1)\\).",
#         "The interval \\([-2, \\infty)\\) includes all numbers from \\(-2\\) to infinity,"
#       ]
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_mean_label": 1,
#     "Qwen2.5-Math-PRM-7B_prm_min_label": 1,
#     "Qwen2.5-Math-PRM-7B_prm_geom_mean_label": 1,
#     "Qwen2.5-Math-PRM-7B_prm_max_label": 1,
#     "Qwen2.5-Math-PRM-7B_prm_product_label": 1
#   },


import json
import os
import argparse
from tqdm import tqdm


def run_eval_prm_k(input_file: str, model_name: str, output_file: str, k: int = 8):
    """Evaluate prm@n based on the labels in the JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = os.path.basename(input_file).replace("_prm@k_format.json", "")

    total_problems = len(data)
    passed_mean = 0
    passed_min = 0
    passed_max = 0
    passed_product = 0
    passed_geom_mean = 0

    base_model_name = os.path.basename(model_name)

    for item in tqdm(data, desc="Evaluating prm@n"):
        if item.get(f"{base_model_name}_prm_mean_label") == 1:
            passed_mean += 1
        if item.get(f"{base_model_name}_prm_min_label") == 1:
            passed_min += 1
        if item.get(f"{base_model_name}_prm_max_label") == 1:
            passed_max += 1
        if item.get(f"{base_model_name}_prm_product_label") == 1:
            passed_product += 1
        if item.get(f"{base_model_name}_prm_geom_mean_label") == 1:
            passed_geom_mean += 1

    prm_mean_rate = passed_mean / total_problems if total_problems > 0 else 0.0
    prm_min_rate = passed_min / total_problems if total_problems > 0 else 0.0
    prm_max_rate = passed_max / total_problems if total_problems > 0 else 0.0
    prm_product_rate = passed_product / total_problems if total_problems > 0 else 0.0
    prm_geom_mean_rate = passed_geom_mean / total_problems if total_problems > 0 else 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{dataset_name} prm@{k} mean: {prm_mean_rate:.4f} ({passed_mean}/{total_problems})\n")
        f.write(f"{dataset_name} prm@{k} min: {prm_min_rate:.4f} ({passed_min}/{total_problems})\n")
        f.write(f"{dataset_name} prm@{k} max: {prm_max_rate:.4f} ({passed_max}/{total_problems})\n")
        f.write(f"{dataset_name} prm@{k} product: {prm_product_rate:.4f} ({passed_product}/{total_problems})\n")
        f.write(f"{dataset_name} prm@{k} geom_mean: {prm_geom_mean_rate:.4f} ({passed_geom_mean}/{total_problems})\n")
    print(f"{dataset_name} prm@{k} mean: {prm_mean_rate:.4f} ({passed_mean}/{total_problems}) saved to {output_file}")
    print(f"{dataset_name} prm@{k} min: {prm_min_rate:.4f} ({passed_min}/{total_problems}) saved to {output_file}")
    print(f"{dataset_name} prm@{k} max: {prm_max_rate:.4f} ({passed_max}/{total_problems}) saved to {output_file}")
    print(f"{dataset_name} prm@{k} product: {prm_product_rate:.4f} ({passed_product}/{total_problems}) saved to {output_file}")
    print(f"{dataset_name} prm@{k} geom_mean: {prm_geom_mean_rate:.4f} ({passed_geom_mean}/{total_problems}) saved to {output_file}")

if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used in prm evaluation")
    parser.add_argument("--k", type=int, default=8, help="Value of n for prm@n")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")
    args = parser.parse_args()
    run_eval_prm_k(args.filename, args.model_name, args.output, args.k)