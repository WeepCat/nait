import json
import os
import argparse
from tqdm import tqdm


def run_eval_greedy(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = os.path.basename(input_file).replace("_greedy_format.json", "")

    total_problems = len(data)
    passed_problems = 0

    for item in tqdm(data, desc="Evaluating pass@n"):
        label = item["greedy_label"]
        if label == 1:
            passed_problems += 1
    
    pass_n_rate = passed_problems / total_problems if total_problems > 0 else 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{dataset_name} greedy: {pass_n_rate:.4f} ({passed_problems}/{total_problems})\n")

    print(f"{dataset_name} greedy: {pass_n_rate:.4f} ({passed_problems}/{total_problems}) saved to {output_file}")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")
    args = parser.parse_args()
    run_eval_greedy(args.filename, args.output)