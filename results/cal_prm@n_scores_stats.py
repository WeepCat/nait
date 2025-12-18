# 读取 json 文件，并计算每个 response 的得分的统计值，保存为新的 json 文件
# 下面是一个示例，你需要根据 prm_scores 的值计算各种统计值
#   {
#     "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
#     "responses": [
#       "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:\n\n\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\n\nGiven the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\).\n\nFirst, we calculate \\(r\\):\n\n\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\n\\]\n\nNext, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\). Since division by zero is undefined, we need to consider the position of the point in the coordinate system. The point \\((0, 3)\\) lies on the positive \\(y\\)-axis. Therefore, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians.\n\nThus, the polar coordinates of the point \\((0, 3)\\) are:\n\n\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]\n\nSo, the final answer is:\n\n\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]"
#     ],
#     "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
#     "steps": [
#       [
#         "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:",
#         "\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]",
#         "Given the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\).",
#         "First, we calculate \\(r\\):",
#         "\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\n\\]",
#         "Next, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\). Since division by zero is undefined, we need to consider the position of the point in the coordinate system. The point \\((0, 3)\\) lies on the positive \\(y\\)-axis. Therefore, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians.",
#         "Thus, the polar coordinates of the point \\((0, 3)\\) are:",
#         "\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]",
#         "So, the final answer is:",
#         "\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]"
#       ]
#     ],
#     "Qwen2.5-Math-PRM-7B_prm_scores": [
#       [
#         0.97265625,
#         0.98046875,
#         1.0,
#         1.0,
#         1.0,
#         0.99609375,
#         1.0,
#         1.0,
#         1.0,
#         1.0
#       ]
#     ],
#   },

import os
import argparse
from tqdm import tqdm
import json
import torch


def cal_prm_n_scores_stats(input_path, output_path, model_name):
    # 读取 json 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    base_model_name = os.path.basename(model_name)
    # 计算每个 response 的 prm 得分的统计值
    for item in tqdm(data, desc="Calculating PRM n-scores statistics"):
        prm_scores = item.get(f"{base_model_name}_prm_scores", [])
        prm_mean = [sum(scores)/len(scores) for scores in prm_scores]
        prm_max = [max(scores) for scores in prm_scores]
        prm_min = [min(scores) for scores in prm_scores]
        prm_product = [torch.prod(torch.tensor(scores)).item() for scores in prm_scores]
        prm_geom_mean = [torch.exp(torch.mean(torch.log(torch.tensor(scores)))).item() for scores in prm_scores]
        
        item[f"{base_model_name}_prm_max"] = prm_max
        item[f"{base_model_name}_prm_product"] = prm_product
        item[f"{base_model_name}_prm_min"] = prm_min
        item[f"{base_model_name}_prm_mean"] = prm_mean
        item[f"{base_model_name}_prm_geom_mean"] = prm_geom_mean

    # 保存结果到新的 json 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model used for PRM scoring.")
    args = parser.parse_args()
    cal_prm_n_scores_stats(args.filename, args.output_path, args.model_name)
    