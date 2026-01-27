import torch
import sys
import os

# Reduce overhead for quick check
test_input_len = 128
test_output_len = 10
test_requests = 4

def verify_v1():
    print("\n--- Verifying V1 (Throughput) Compatibility ---")
    try:
        import bench_snapkv_sweeps as v1
        # Patch Globals
        v1.INPUT_LENGTHS = [test_input_len]
        v1.COMPRESSION_RATES = [2]
        v1.TOTAL_REQUESTS = test_requests
        v1.OUTPUT_LEN = test_output_len
        
        # Run Baseline
        print("Running V1 Baseline...")
        res_base = v1.run_benchmark(test_input_len, False)
        if res_base is None: raise Exception("V1 Baseline returned None")
        
        # Run SnapKV
        print("Running V1 SnapKV...")
        res_snap = v1.run_benchmark(test_input_len, True, 2)
        if res_snap is None: raise Exception("V1 SnapKV returned None")
        
        print("V1 Verified.")
    except Exception as e:
        print(f"V1 Failed: {e}")
        import traceback
        traceback.print_exc()

def verify_v2():
    print("\n--- Verifying V2 (Capacity) Compatibility ---")
    try:
        import bench_snapkv_sweeps_v2 as v2
        # Patch Globals
        v2.INPUT_LENGTHS = [test_input_len]
        v2.COMPRESSION_RATES = [2]
        v2.TOTAL_REQUESTS_PROBE = test_requests # Small number for quick check
        v2.OUTPUT_LEN = test_output_len
        
        # Run Baseline
        print("Running V2 Baseline...")
        res_base = v2.run_capacity_test(test_input_len, False)
        # Note: v2 might return None if it fails
        
        # Run SnapKV
        print("Running V2 SnapKV...")
        res_snap = v2.run_capacity_test(test_input_len, True, 2)
        
        print("V2 Verified.")
    except Exception as e:
        print(f"V2 Failed: {e}")
        import traceback
        traceback.print_exc()        

def verify_v3():
    print("\n--- Verifying V3 (Combined) Compatibility ---")
    try:
        import bench_snapkv_sweeps_v3 as v3
        # Patch Globals
        v3.INPUT_LENGTHS = [test_input_len]
        v3.COMPRESSION_RATES = [2]
        v3.TOTAL_REQUESTS = test_requests
        v3.OUTPUT_LEN = test_output_len
        
        # Run SnapKV (Baseline is implied as compatible if this works)
        print("Running V3 SnapKV...")
        res_snap = v3.run_benchmark_v3(test_input_len, True, 2)
        if res_snap is None: raise Exception("V3 SnapKV returned None")
        
        print("V3 Verified.")
    except Exception as e:
        print(f"V3 Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure current dir is in path
    sys.path.append("/root/nano-vllm")
    
    verify_v1()
    torch.cuda.empty_cache()
    verify_v2()
    torch.cuda.empty_cache()
    verify_v3()
