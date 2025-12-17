# 读取一个 json 文件，根据 labels 字段的结果，统计 pass@n 的比例
# 下面是一个示例, 你需要根据 "label" 字段的数值结果来统计 pass@n, 其中 1 表示正确, -1 表示错误, 0 表示不计入统计, 只要 labels 中有一个 1 就算通过
# {
#     "problem": "A ramp has a horizontal length of 12 feet and a vertical height of 1 foot. Which of the following expressions gives the measure of the acute angle the ramp makes with the horizontal: (F) $\\tan \\left(\\frac{1}{12}\\right)$ (G) $\\tan(12)$ (H) $\\tan^{-1} \\left(\\frac{1}{12}\\right)$ (J) $tan^{-1}(12)$ (K) $\\sqrt{1^{2}+12^{12}}$",
#     "responses": [
#       "To determine the measure of the acute angle that the ramp makes with the horizontal, we need to use the definition of the tangent function in a right triangle. The tangent of an angle in a right triangle is the ratio of the length of the opposite side to the length of the adjacent side.\n\nIn this problem, the vertical height of the ramp is the opposite side, and the horizontal length of the ramp is the adjacent side. Therefore, the tangent of the angle \\(\\theta\\) that the ramp makes with the horizontal is given by:\n\n\\[\n\\tan \\theta = \\frac{\\text{opposite}}{\\text{adjacent}} = \\frac{1}{12}\n\\]\n\nTo find the angle \\(\\theta\\), we need to take the inverse tangent (or arctangent) of \\(\\frac{1}{12}\\). The inverse tangent function, denoted as \\(\\tan^{-1}\\), gives us the angle whose tangent is a given value. Therefore, the angle \\(\\theta\\) is:\n\n\\[\n\\theta = \\tan^{-1} \\left(\\frac{1}{12}\\right)\n\\]\n\nLooking at the given choices, we see that the expression that gives the measure of the acute angle the ramp makes with the horizontal is:\n\n\\[\n\\tan^{-1} \\left(\\frac{1}{12}\\right)\n\\]\n\nThus, the correct choice is \\(\\boxed{H}\\)."
#     ],
#     "answer": "H",
#     "label": [
#       1
#     ]
# }
# 将统计结果追加写入到一个 txt 文件中

import json
import os
import argparse
from tqdm import tqdm


def run_eval_pass_k(input_file: str, output_file: str, k: int = 8):
    """Evaluate pass@n based on the labels in the JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = os.path.basename(input_file).replace("_pass@k_format.json", "")

    total_problems = len(data)
    passed_problems = 0

    for item in tqdm(data, desc="Evaluating pass@n"):
        labels = item["labels"]
        if sum(1 for label in labels if label == 1) >= 1:
            passed_problems += 1

    pass_n_rate = passed_problems / total_problems if total_problems > 0 else 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{dataset_name} pass@{k}: {pass_n_rate:.4f} ({passed_problems}/{total_problems})\n")

    print(f"{dataset_name} pass@{k}: {pass_n_rate:.4f} ({passed_problems}/{total_problems}) saved to {output_file}")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--k", type=int, default=8, help="Value of n for pass@n")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")
    args = parser.parse_args()
    run_eval_pass_k(args.filename, args.output, args.k)