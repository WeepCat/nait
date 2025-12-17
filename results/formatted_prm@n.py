# 读取 json 文件，并将 responses 中的每个 response 转化为一个 steps 列表，保存为新的 json 文件
# 下面是一个示例，你需要将 "responses" 中的每个 response 按照步骤拆分为 steps 列表
# {
#     "problem": "Given sets $M=\\{x|x+2\\geq 0\\},N=\\{x|x-1<0\\}$, find $M \\cap N$.",
#     "responses": [
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we need to determine the elements that satisfy both conditions defining \\( M \\) and \\( N \\).\n\nFirst, let's solve for the set \\( M \\):\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSubtract 2 from both sides of the inequality:\n\\[ x \\geq -2 \\]\nThus, the set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\n\nNext, let's solve for the set \\( N \\):\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nAdd 1 to both sides of the inequality:\n\\[ x < 1 \\]\nThus, the set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\n\nNow, we need to find the intersection \\( M \\cap N \\), which consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).\n\nGraphically, this can be represented on a number line:\n- The set \\( M \\) includes all points from \\(-2\\) to positive infinity.\n- The set \\( N \\) includes all points from negative infinity to just below \\(1\\).\n\nThe intersection \\( M \\cap N \\) will be the overlap of these two intervals:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq x < 1 \\} \\]\n\nTherefore, the final answer is:\n\\[\n\\boxed{[-2, 1)}\n\\]",
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we need to determine the elements that satisfy both conditions defining \\( M \\) and \\( N \\).\n\nFirst, let's solve for the set \\( M \\):\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSubtract 2 from both sides of the inequality:\n\\[ x \\geq -2 \\]\nThus, the set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\n\nNext, let's solve for the set \\( N \\):\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nAdd 1 to both sides of the inequality:\n\\[ x < 1 \\]\nThus, the set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\n\nNow, we need to find the intersection \\( M \\cap N \\), which consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).\n\nCombining these two conditions, we get:\n\\[ -2 \\leq x < 1 \\]\n\nTherefore, the intersection \\( M \\cap N \\) is:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq x < 1 \\} \\]\n\nThe final answer is:\n\\[\n\\boxed{[-2, 1)}\n\\]",
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we need to determine the elements that satisfy both conditions defining \\( M \\) and \\( N \\).\n\nFirst, let's solve for the set \\( M \\):\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSubtract 2 from both sides of the inequality:\n\\[ x \\geq -2 \\]\nThus, the set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\n\nNext, let's solve for the set \\( N \\):\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nAdd 1 to both sides of the inequality:\n\\[ x < 1 \\]\nThus, the set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\n\nNow, we need to find the intersection \\( M \\cap N \\), which consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).\n\nGraphically, this can be represented on a number line:\n- The set \\( M \\) includes all numbers from \\(-2\\) to positive infinity.\n- The set \\( N \\) includes all numbers from negative infinity to just below \\(1\\).\n\nThe intersection of these two sets is the range where both conditions are satisfied, which is from \\(-2\\) (inclusive) to \\(1\\) (exclusive).\n\nTherefore, the intersection \\( M \\cap N \\) is:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq!!",
#         "To find the intersection \\( M \\cap N \\) of the sets \\( M \\) and \\( N \\), we need to determine the elements that satisfy both conditions defining \\( M \\) and \\( N \\).\n\nFirst, let's solve for the set \\( M \\):\n\\[ M = \\{ x \\mid x + 2 \\geq 0 \\} \\]\nSubtract 2 from both sides of the inequality:\n\\[ x \\geq -2 \\]\nThus, the set \\( M \\) is:\n\\[ M = \\{ x \\mid x \\geq -2 \\} \\]\n\nNext, let's solve for the set \\( N \\):\n\\[ N = \\{ x \\mid x - 1 < 0 \\} \\]\nAdd 1 to both sides of the inequality:\n\\[ x < 1 \\]\nThus, the set \\( N \\) is:\n\\[ N = \\{ x \\mid x < 1 \\} \\]\n\nNow, we need to find the intersection \\( M \\cap N \\), which consists of all \\( x \\) that satisfy both \\( x \\geq -2 \\) and \\( x < 1 \\).\n\nCombining these two conditions, we get:\n\\[ -2 \\leq x < 1 \\]\n\nTherefore, the intersection \\( M \\cap N \\) is:\n\\[ M \\cap N = \\{ x \\mid -2 \\leq x < 1 \\} \\]\n\nThe final answer is:\n\\[\n\\boxed{[-2, 1)}\n\\]"
#     ],
#     "answer": "$\\{x|-2\\leq x < 1\\}$"
# }

import json
import os
import argparse
from tqdm import tqdm


def transform_format(input_file: str, output_file: str):
    """Transform the format of the JSON file by converting responses to steps lists."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformed_data = []
    for item in tqdm(data, desc="Transforming data"):
        transformed_item = {
            "problem": item["problem"],
            "responses": item["responses"],
            "answer": item["answer"],
            "steps": [response.split('\n\n') for response in item["responses"]]
        }
        transformed_data.append(transformed_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

    print(f"Transformed data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    transform_format(args.filename, args.output)