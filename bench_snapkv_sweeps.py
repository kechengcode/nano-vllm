import os
import time
import json
import torch
import gc
import random
import csv
from nanovllm import LLM, SamplingParams

# Constants
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B"
RESULTS_FILE = "snapkv_sweep_results.csv"
INPUT_LENGTHS = [2000, 4000, 6000, 8000, 10000]
COMPRESSION_RATES = [2, 4, 8, 16, 32, 64]
TOTAL_REQUESTS = 256
OUTPUT_LEN = 500

def run_benchmark(input_len, use_snapkv, compression_rate=None):
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"--- Running: Input={input_len}, SnapKV={use_snapkv}, Rate={compression_rate} ---")
    
    # Calculate configuration
    snapkv_limit = None
    if use_snapkv and compression_rate:
        snapkv_limit = max(1, int(input_len / compression_rate))
    
    # Initialize LLM
    try:
        # Increase max_model_len to accommodate the largest test case
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=12000, 
            enable_snapkv=use_snapkv,
            snapkv_limit=snapkv_limit,
            enforce_eager=True, # Align with previous tests for consistency
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # Prepare requests
    # Use random token IDs to avoid tokenizer overhead and ensure exact lengths
    # Vocab size for Qwen usually large, safe to use 0-10000
    prompt_token_ids = [
        [random.randint(0, 10000) for _ in range(input_len)]
        for _ in range(TOTAL_REQUESTS)
    ]
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=OUTPUT_LEN,
        ignore_eos=True # Enforce exact output length
    )
    
    # Warmup (optional, but good for stability, maybe skip to save time for large batch)
    # Using a tiny run? The engine usually warms up.
    
    # Measure performance
    start_time = time.perf_counter()
    
    # Run generation
    try:
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    except Exception as e:
        print(f"Generation failed: {e}")
        llm.exit()
        return None
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Stats
    total_output_tokens = sum(len(o['token_ids']) for o in outputs)
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_tokens = total_input_tokens + total_output_tokens
    
    throughput = total_tokens / duration if duration > 0 else 0
    
    # Cleanup
    llm.exit()
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "input_len": input_len,
        "output_len": OUTPUT_LEN,
        "snapkv": "ON" if use_snapkv else "OFF",
        "compression_rate": compression_rate if use_snapkv else 1,
        "snapkv_limit": snapkv_limit if use_snapkv else "N/A",
        "duration": round(duration, 4),
        "total_tokens": total_tokens,
        "throughput": round(throughput, 2)
    }

def main():
    # Setup CSV
    fieldnames = ["input_len", "output_len", "snapkv", "compression_rate", "snapkv_limit", "duration", "total_tokens", "throughput"]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    results = []

    # Loop over input lengths
    for input_len in INPUT_LENGTHS:
        print(f"\n\nStarting sweeps for Input Length: {input_len}")
        
        # 1. Baseline (SnapKV OFF)
        res = run_benchmark(input_len, use_snapkv=False)
        if res:
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(res)
            results.append(res)
            print(f"Baseline Result: {res['throughput']} tok/s")
        
        # 2. SnapKV Sweeps
        for rate in COMPRESSION_RATES:
            res = run_benchmark(input_len, use_snapkv=True, compression_rate=rate)
            if res:
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)
                results.append(res)
                print(f"SnapKV (x{rate}) Result: {res['throughput']} tok/s")

    print(f"\nBenchmark completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    torch.manual_seed(42)
    main()
