# qwen2.5-7b-instruct
# qwen2.5-math-7b-instruct
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
load_dotenv()

SYSTEM_PROMPT = """
Please reason step by step, and put your final answer within \\boxed{}.
"""


client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

async def _make_request(args, problem:  str) -> list[str]:
    """Make a single API request and return all generated responses."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    response = await client.chat.completions.create(
        model=args.model,
        messages=messages,
        n=args.n_responses,
        temperature=0.7,
        stop=None
    )
    # Return all n responses, not just the first one
    return [choice.message.content.strip() for choice in response.choices]


async def _retry_request(args, problem: str, retries:  int = 3, delay: int = 5) -> list[str]:
    """
    Attempt to make an API request with retries.

    Args:
        args: Command line arguments.
        problem (str): The input to process. 
        retries (int): The number of attempts to try.
        delay (int): Seconds to wait between attempts.

    Returns:
        list[str]: List of generated responses. 

    Raises:
        Exception:  The last exception caught if all retries fail.
    """
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            responses = await _make_request(args, problem)
            return responses
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt} for problem failed with error: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
    raise last_exception


async def process_batch(args, batch_data: list[dict]) -> list[dict]:
    """
    Process a batch of problems concurrently. 

    Args:
        args: Command line arguments.
        batch_data: List of dictionaries containing problem and answer.

    Returns:
        List of results with problem, responses, and answer.
    """
    tasks = [
        _retry_request(args, item["problem"])
        for item in batch_data
    ]
    
    responses_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for item, responses in zip(batch_data, responses_list):
        if isinstance(responses, Exception):
            print(f"Failed to process problem: {item['problem'][: 50]}...  Error: {responses}")
            results. append({
                "problem": item["problem"],
                "responses": [],
                "answer": item["answer"]
            })
        else:
            results.append({
                "problem": item["problem"],
                "responses": responses,
                "answer": item["answer"]
            })
    
    return results


async def run_async(args):
    """Main async function to process the dataset."""
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split)
    
    # Extract problems and answers from the dataset
    data = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]
    
    all_results = []
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size
    
    print(f"Total problems: {len(data)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Responses per problem: {args.n_responses}")
    print(f"Interval between batches: {args.interval} seconds")
    print("-" * 50)
    
    for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (problems {start_idx + 1}-{end_idx})")
        
        batch_results = await process_batch(args, batch_data)
        all_results.extend(batch_results)
        
        # Sleep between batches (except for the last batch)
        if batch_idx < total_batches - 1:
            print(f"Sleeping for {args.interval} seconds...")
            await asyncio.sleep(args.interval)
    
    # Save results to JSON file
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {len(all_results)} problems")
    print(f"Results saved to {args.output}")


def run(args):
    """Entry point that runs the async main function."""
    asyncio.run(run_async(args))


if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Generate LLM responses for math problems")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="HuggingFace dataset split")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for concurrent API calls")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of responses to generate per problem")
    parser.add_argument("--output", type=str, default="responses.json", help="Output JSON file path")
    parser.add_argument("--interval", type=int, default=5, help="Time interval between batches in seconds")
    args = parser.parse_args()
    run(args)

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=os.getenv("DASHSCOPE_BASE_URL"),
# )
# completion = client.chat.completions.create(
#     model="qwen2.5-math-7b-instruct",
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "Given sets $M=\{x|x+2\geq 0\},N=\{x|x-1<0\}$, find $M \cap N$."},
#     ],
#     n=4,
#     temperature=0.7,
# )

# response = completion.choices[0].message.content
# print("Response:", response)
# print(completion.model_dump_json())