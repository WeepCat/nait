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

    for item in tqdm(data, desc="Evaluating maj@n"):
        labels = item["labels"]
        if sum(1 for label in labels if label == 1) >= len(labels) // 2:
            passed_problems += 1

    pass_n_rate = passed_problems / total_problems if total_problems > 0 else 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{dataset_name} maj@{k}: {pass_n_rate:.4f} ({passed_problems}/{total_problems})\n")

    print(f"{dataset_name} maj@{k}: {pass_n_rate:.4f} ({passed_problems}/{total_problems}) saved to {output_file}")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--k", type=int, default=8, help="Value of n for maj@n")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")
    args = parser.parse_args()
    run_eval_pass_k(args.filename, args.output, args.k)