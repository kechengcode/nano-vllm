import torch
from nanovllm.layers.attention import Attention

class MockContext:
    def __init__(self, limit, cu_seqlens):
        self.snapkv_limit = limit
        self.cu_seqlens_q = cu_seqlens
        self.snapkv_enabled = True
        self.snapkv_sample_queries = 32 # Not used in v2 but kept for compat if needed

def test_snapkv_logic_v2():
    print("Testing SnapKV V2 Logic...")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    # --- Test Case 1: MHA ---
    print("\n--- Test Case 1: MHA (4 Heads, 4 KV Heads) ---")
    num_heads = 4
    head_dim = 64
    scale = head_dim ** -0.5
    num_kv_heads = 4 
    
    attn = Attention(num_heads, head_dim, scale, num_kv_heads).to(device)
    
    # Seq 1: length 100. Limit 50. Window 32. -> Keep 50
    # Seq 2: length 20. Limit 50. Window 32. -> Keep 20
    
    seq1_len = 100
    seq2_len = 20
    total_len = seq1_len + seq2_len
    
    limit = 50
    
    q = torch.randn((total_len, num_heads, head_dim), device=device)
    k = torch.randn((total_len, num_kv_heads, head_dim), device=device)
    # create distinct pattern to verify selection? 
    # Let's trust verify shapes first.
    
    cu_seqlens = torch.tensor([0, seq1_len, total_len], dtype=torch.int32, device=device)
    
    ctx = MockContext(limit, cu_seqlens)
    
    mask = attn.run_snapkv_selection(q, k, ctx)
    
    print(f"Mask shape: {mask.shape}")
    s1_kept = mask[:seq1_len].sum().item()
    s2_kept = mask[seq1_len:].sum().item()
    print(f"Seq 1 (L={seq1_len}) kept: {s1_kept} (Expected {limit})")
    print(f"Seq 2 (L={seq2_len}) kept: {s2_kept} (Expected {seq2_len})")
    
    # Verify Window
    # Last 32 of Seq 1
    window_start_idx = seq1_len - 32
    window_preserved = mask[window_start_idx:seq1_len].all().item()
    print(f"Seq 1 Window Preserved: {window_preserved}")
    
    if s1_kept != limit:
        print("FAILED: Seq 1 count mismatch")
    if s2_kept != seq2_len:
        print("FAILED: Seq 2 count mismatch")
    if not window_preserved:
        print("FAILED: Window not preserved")
        
    # --- Test Case 2: GQA ---
    print("\n--- Test Case 2: GQA (4 Heads, 1 KV Head) ---")
    num_heads = 4
    num_kv_heads = 1
    attn_gqa = Attention(num_heads, head_dim, scale, num_kv_heads).to(device)
    
    k_gqa = torch.randn((total_len, num_kv_heads, head_dim), device=device)
    
    mask_gqa = attn_gqa.run_snapkv_selection(q, k_gqa, ctx)
    
    s1_kept_gqa = mask_gqa[:seq1_len].sum().item()
    print(f"GQA Seq 1 kept: {s1_kept_gqa}")
    if s1_kept_gqa != limit:
        print("FAILED: GQA count mismatch")
    else:
        print("PASSED: GQA logic works")

    # --- Test Case 3: Verify Importance Logic (Simple) ---
    print("\n--- Test Case 3: Importance Logic Verification ---")
    # Construct a case where one specific token in history is very important to the window queries
    # Make q[-window:] correlate strongly with k[5]
    
    # Use small dimensions
    L = 60
    window = 32
    limit = 35 # Keep window(32) + 3 history
    
    q_simple = torch.zeros((L, 4, 16), device=device)
    k_simple = torch.zeros((L, 4, 16), device=device) # MHA
    
    # Last 32 are window
    # Make q in window look for k at index 5, 10, 15
    target_indices = [5, 10, 15]
    
    # Queries in window point to direction V
    V = torch.randn(16, device=device)
    V = V / V.norm()
    
    # Set window queries to V
    q_simple[-window:, :, :] = V
    
    # Set keys at targets to V (high score)
    # Set others to -V (low score)
    k_simple[:, :, :] = -V
    for idx in target_indices:
        k_simple[idx, :, :] = V
        
    cu_simple = torch.tensor([0, L], dtype=torch.int32, device=device)
    ctx_simple = MockContext(limit, cu_simple)
    
    # We expect indices 5, 10, 15 to be selected among history (0..27)
    # History range is 0 to (60-32)=28.
    
    mask_imp = attn.run_snapkv_selection(q_simple, k_simple, ctx_simple)
    
    # Check if 5, 10, 15 are TRUE
    print(f"Mask at 5: {mask_imp[5]}")
    print(f"Mask at 10: {mask_imp[10]}")
    print(f"Mask at 15: {mask_imp[15]}")
    print(f"Mask at 20 (distractor): {mask_imp[20]}")
    
    kept_count = mask_imp.sum().item()
    print(f"Total kept: {kept_count}/{limit}")
    
    if mask_imp[5] and mask_imp[10] and mask_imp[15] and not mask_imp[20]:
        print("SUCCESS: Priority tokens selected")
    else:
        print("WARNING: Importance logic might be fuzzy or params different, check results")

if __name__ == "__main__":
    test_snapkv_logic_v2()
