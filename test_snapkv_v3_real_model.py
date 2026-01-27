import torch
import random
import os
import gc
from nanovllm import LLM, SamplingParams

# 配置
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-0.6B"
SNAPKV_LIMIT = 128  # 设置一个较小的 Limit 以强制触发压缩
MAX_MODEL_LEN = 4096

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def test_real_model_v3():
    print(f"==================================================")
    print(f"Testing SnapKV V3 Integration with Real Model")
    print(f"Model: {MODEL_PATH}")
    print(f"SnapKV Limit: {SNAPKV_LIMIT}")
    print(f"==================================================\n")

    setup_seed(42)

    # 1. 初始化模型
    print(">>> Initializing LLM with SnapKV Enabled...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            enable_snapkv=True,
            snapkv_limit=SNAPKV_LIMIT,
            enforce_eager=True  # 建议在测试新逻辑时使用 eager 模式，方便定位错误
        )
    except Exception as e:
        print(f"!!! Failed to load model: {e}")
        return

    # 2. 构造测试数据
    # Case 1: 长度超过 Limit (1000 > 128)，应该触发 SnapKV 选择
    # Case 2: 长度小于 Limit (50 < 128)，应该全量保留
    prompts_token_ids = [
        [random.randint(0, 10000) for _ in range(1000)], # Case 1: Long
        [random.randint(0, 10000) for _ in range(50)],   # Case 2: Short
    ]
    
    sampling_params = SamplingParams(
        max_tokens=20, 
        ignore_eos=True,
        temperature=0.6 # Use valid temp
    )

    print(f">>> Created {len(prompts_token_ids)} test prompts.")
    print(f"    Prompt 0 Length: {len(prompts_token_ids[0])} (Should trigger Compression)")
    print(f"    Prompt 1 Length: {len(prompts_token_ids[1])} (Should keep full)")

    # 3. 运行生成
    print("\n>>> Starting Generation...")
    try:
        # 使用 generate 接口运行
        # 内部会调用 model_runner -> attention.forward -> run_snapkv_selection
        outputs = llm.generate(prompts_token_ids, sampling_params)
        
        print("\n>>> Generation Completed Successfully!")
        
        # 4. 打印结果摘要
        for i, output in enumerate(outputs):
            input_len = len(prompts_token_ids[i])
            output_len = len(output['token_ids'])
            print(f"    Request {i}: Input={input_len}, Output={output_len} tokens generated.")
            print(f"    First 5 output ids: {output['token_ids'][:5]}")

    except torch.cuda.OutOfMemoryError:
        print("!!! OOM Error occurred. Try reducing batch size or input length.")
    except Exception as e:
        print(f"!!! Runtime Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        llm.exit()
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        print("\n>>> Test Verified.")

if __name__ == "__main__":
    test_real_model_v3()
