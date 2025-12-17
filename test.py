import os
os.environ['HF_HOME'] = "/root/autodl-tmp/hf-mirror"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Initialize DeepSpeed inference
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=1,  # 2 GPUs for tensor parallelism
    dtype=torch.float16,  # FP16
    replace_with_kernel_inject=True,  # Optimized kernels
    # enable_cuda_graph=True  # CUDA Graph
)

# Prepare input
text = ["This movie is fantastic!"] * 100000
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Inference
with torch.no_grad():
    outputs = ds_engine(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1)
    print(pred)
    # print(f"Text: {text}, Prediction: {'Positive' if pred.item() == 1 else 'Negative'}")
