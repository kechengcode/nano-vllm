import os
import time
import sys
import torch
import gc
import random
import csv
from tqdm import tqdm
from nanovllm import LLM, SamplingParams

# Constants
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B"
RESULTS_FILE = "snapkv_max_batch_size.csv"
INPUT_LENGTHS = [2000, 4000, 6000, 8000, 10000]
# INPUT_LENGTHS = [2000] # For debug
COMPRESSION_RATES = [2, 4, 8, 16, 32, 64]
# COMPRESSION_RATES = [2] # For Debug

# Enough requests to saturate memory
# With 20GB and compressed 10k input (x64 -> 156 tokens), we can fit thousands.
# 20GB / (156 * 2B * 2 layers * ...) -> lots.
# Let's set a high limit.
TOTAL_REQUESTS_PROBE = 256
OUTPUT_LEN = 512 # Increased from 20 to 512 to ensure memory saturation before early tasks finish

def run_capacity_test(input_len, use_snapkv, compression_rate=None):
    torch.cuda.empty_cache()
    gc.collect()
    
    label = f"Input={input_len}, SnapKV={'ON' if use_snapkv else 'OFF'}"
    if use_snapkv:
        label += f", Rate={compression_rate}"
    tqdm.write(f"--- Probing Capacity: {label} ---")
    
    # Calculate configuration
    snapkv_limit = None
    if use_snapkv and compression_rate:
        snapkv_limit = max(1, int(input_len / compression_rate))
    
    # Initialize LLM with HIGH max_num_seqs to avoid software cap
    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=12000,
            max_num_seqs=4096, # High limit for capacity probing
            max_num_batched_tokens=32768, # Allow larger batches
            enable_snapkv=use_snapkv,
            snapkv_limit=snapkv_limit,
            enforce_eager=True,
        )
    except Exception as e:
        tqdm.write(f"Failed to load model: {e}")
        return None

    # Prepare requests
    prompt_token_ids = [
        [random.randint(0, 10000) for _ in range(input_len)]
        for _ in range(TOTAL_REQUESTS_PROBE)
    ]
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=OUTPUT_LEN,
        ignore_eos=True
    )
    
    for ids in prompt_token_ids:
        llm.add_request(ids, sampling_params)
        
    # Stats tracking
    max_running = 0
    stable_steps = 0
    
    # Custom stepping loop
    pbar = tqdm(total=TOTAL_REQUESTS_PROBE, desc="Step Progress (Running Seqs)", unit="seq", file=sys.stdout, leave=False)
    try:
        while not llm.is_finished():
            # Run a step
            llm.step()
            
            # Check concurrency
            # running is a deque in scheduler
            current_running = len(llm.scheduler.running)
            pbar.set_postfix({"Running": current_running, "Max": max_running, "StableSteps": stable_steps})
            
            # Since scheduler moves waiting -> running in batches, 
            # and running sequences stay in running list (unless preempted, which happens if OOM).
            # We want the peak steady state.
            
            if current_running > max_running:
                max_running = current_running
                stable_steps = 0
            else:
                stable_steps += 1
            
            # Optimization: If we see max_running stable for many steps and waiting queue is nonempty,
            # it means we hit the capacity limit (cannot schedule more).
            # Or if waiting queue is empty, we hit TOTAL_REQUESTS limit (need more probe requests).
            
            waiting_count = len(llm.scheduler.waiting)
            
            # if waiting_count > 0 and stable_steps > 20:
            #     tqdm.write(f"  Capacity Reached: {max_running} sequences (Waiting: {waiting_count})")
            #     break
            
            # if waiting_count == 0 and stable_steps > 20: 
            #     tqdm.write(f"  Warning: All {TOTAL_REQUESTS_PROBE} requests scheduled. Capacity might be higher.")
            #     break
                
    except Exception as e:
        tqdm.write(f"Run failed: {e}")
    finally:
        pbar.close()
        llm.exit()
        del llm
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "input_len": input_len,
        "snapkv": "ON" if use_snapkv else "OFF",
        "compression_rate": compression_rate if use_snapkv else 1,
        "snapkv_limit": snapkv_limit if use_snapkv else "N/A",
        "max_batch_size": max_running
    }

def main():
    fieldnames = ["input_len", "snapkv", "compression_rate", "snapkv_limit", "max_batch_size"]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # 1. Baseline
    for length in tqdm(INPUT_LENGTHS, desc="Baseline Sweeps", file=sys.stdout):
        res = run_capacity_test(length, use_snapkv=False)
        if res:
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(res)
            # print(f"  Result: {res['max_batch_size']}")

    # 2. SnapKV Sweeps
    total_snapkv_runs = len(INPUT_LENGTHS) * len(COMPRESSION_RATES)
    with tqdm(total=total_snapkv_runs, desc="SnapKV Sweeps", file=sys.stdout) as pbar:
        for length in INPUT_LENGTHS:
            for rate in COMPRESSION_RATES:
                res = run_capacity_test(length, use_snapkv=True, compression_rate=rate)
                if res:
                    with open(RESULTS_FILE, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(res)
                    # print(f"  Result: {res['max_batch_size']}")
                pbar.update(1)

    print(f"\nCapacity benchmark completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
