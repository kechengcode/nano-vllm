import os
import time
import torch
import gc
import random
import csv
import sys
from tqdm import tqdm
from nanovllm import LLM, SamplingParams

# Constants
MODEL_PATH = "/root/autodl-tmp/models/qwen3-8b"
RESULTS_FILE = "snapkv_throughput_and_maxbatch_v3.csv"
#INPUT_LENGTHS = [2000, 4000, 6000, 8000, 10000]
INPUT_LENGTHS = [20000]
COMPRESSION_RATES = [2, 4, 8, 16, 32, 64]
# Make sure we have enough requests to potentially hit the max batch size
# If TOTAL_REQUESTS is too small, the system might finish before ramping up fully.
# 256 might be enough for long inputs, but for short inputs/high compression, we might want more.
# However, the user asked to stick to the previous workload logic but record max batch size.
# Let's use 256 as in v1, but maybe the user implied using the v2 large number?
# "测试特定长度下TOTAL_REQUESTS个序列吞吐量的过程中" - "during the process of testing throughput of TOTAL_REQUESTS sequences".
# I will stick to the v1 constant of 256 for now, or maybe slightly larger to be safe?
# Let's stick to 256 to be consistent with the "v1" reference, unless it proves too small to hit max batch.
# Actually, for 10k input, 256 is plenty. For 2k x64 compression, 256 might run fast.
# Let's keep it 256 as per the v1 "TOTAL_REQUESTS" reference. 
TOTAL_REQUESTS = 512
OUTPUT_LEN = 512

def run_benchmark_v3(input_len, use_snapkv, compression_rate=None):
    torch.cuda.empty_cache()
    gc.collect()
    
    label = f"Input={input_len}, SnapKV={'ON' if use_snapkv else 'OFF'}"
    if use_snapkv:
        label += f", Rate={compression_rate}"
    tqdm.write(f"--- Running v3 Benchmark: {label} ---")
    
    # Calculate configuration
    snapkv_limit = None
    if use_snapkv and compression_rate:
        snapkv_limit = max(1, int(input_len / compression_rate))
    
    # Initialize LLM
    # We need a high max_num_seqs to ensure the software limit doesn't cap us before memory does
    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=12000, 
            max_num_seqs=4096,         # ALLOW HIGH CONCURRENCY
            max_num_batched_tokens=32768, # ALLOW LARGE BATCHES
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
        for _ in range(TOTAL_REQUESTS)
    ]
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=OUTPUT_LEN,
        ignore_eos=True
    )
    
    # Add all requests to the engine
    for ids in prompt_token_ids:
        llm.add_request(ids, sampling_params)
    
    # Measure performance variables
    start_time = time.perf_counter()
    outputs_map = {}
    
    # Stats tracking for Max Batch Size
    max_running = 0
    
    # Custom stepping loop (like v2) to monitor batch size, but collecting outputs (like v1)
    # We use tqdm for progress
    pbar = tqdm(total=TOTAL_REQUESTS, desc="Processing", unit="seq", leave=False, file=sys.stdout)
    
    try:
        while not llm.is_finished():
            # Run one step
            step_outputs, _ = llm.step()
            
            # 1. Update Max Batch Size
            # llm.scheduler.running is the deque of currently running sequences
            current_running = len(llm.scheduler.running)
            if current_running > max_running:
                max_running = current_running
            
            pbar.set_postfix({"Running": current_running, "MaxBS": max_running})
            
            # 2. Collect outputs (finished sequences)
            # step_outputs is a list of (seq_id, token_ids) for finished sequences? 
            # Wait, let's check llm_engine.py step() return value.
            # step() returns: outputs, num_tokens
            # and outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            # So step_outputs contains ONLY finished sequences.
            
            for seq_id, token_ids in step_outputs:
                outputs_map[seq_id] = token_ids
                pbar.update(1)

    except Exception as e:
        tqdm.write(f"Generation failed: {e}")
        try:
            llm.exit()
        except:
            pass
        return None
    
    pbar.close()
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Stats Calculation
    # We reconstruct outputs list based on input order if needed, but for stats we just need totals
    total_output_tokens = sum(len(ids) for ids in outputs_map.values())
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    # Note: total_input_tokens counts the full prompt length for every request.
    # Throughput usually = (Total Input Tokens + Total Output Tokens) / Total Time
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
        "throughput": round(throughput, 2),
        "max_batch_size": max_running
    }

def main():
    fieldnames = ["input_len", "output_len", "snapkv", "compression_rate", "snapkv_limit", "duration", "total_tokens", "throughput", "max_batch_size"]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    # Combined Sweeps: Loop through Lengths -> (Baseline + Rates)
    total_runs = len(INPUT_LENGTHS) * (1 + len(COMPRESSION_RATES))
    
    with tqdm(total=total_runs, desc="Benchmark Sweeps", file=sys.stdout) as pbar:
        for length in INPUT_LENGTHS:
            # 1. Baseline (SnapKV OFF)
            res = run_benchmark_v3(length, use_snapkv=False)
            if res:
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)
            pbar.update(1)

            # 2. SnapKV Sweeps
            for rate in COMPRESSION_RATES:
                res = run_benchmark_v3(length, use_snapkv=True, compression_rate=rate)
                if res:
                    with open(RESULTS_FILE, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(res)
                pbar.update(1)

    print(f"\nBenchmark v3 completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
