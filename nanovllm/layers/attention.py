import torch
from torch import nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        use_snapkv = (
            context.is_prefill and context.snapkv_enabled and context.slot_mapping is not None
            and context.slot_mapping.numel() and context.block_tables is None
            and k_cache.numel() and v_cache.numel()
        )
        if use_snapkv:
            keep_mask = self.run_snapkv_selection(q, k, context)
            compact_slots = self.build_compact_slot_mapping(context, keep_mask)
            if compact_slots.numel():
                store_kvcache(k[keep_mask], v[keep_mask], k_cache, v_cache, compact_slots)
        else:
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            kv_k, kv_v = (k_cache, v_cache) if context.block_tables is not None else (k, v)
            o = flash_attn_varlen_func(q, kv_k, kv_v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def run_snapkv_selection(self, q: torch.Tensor, k: torch.Tensor, context) -> torch.Tensor:
        limit = context.snapkv_limit
        if limit is None or limit <= 0:
            return torch.ones(k.shape[0], dtype=torch.bool, device=k.device)
        
        cu = context.cu_seqlens_q
        device = k.device
        H = self.num_heads
        Hkv = self.num_kv_heads
        scale = self.scale
        
        # SnapKV Hyperparameters (aligned with reference defaults)
        window_size = 8
        kernel_size = 7
        pooling = 'avgpool'
        
        mask = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
        
        # Process each sequence independently
        for i in range(cu.numel() - 1):
            start = cu[i].item()
            end = cu[i + 1].item()
            L = end - start
            
            # If sequence fits in limit, keep everything
            if L <= limit:
                mask[start:end] = True
                continue
            
            # 1. Determine Observation Window
            valid_window = min(window_size, L)
            
            # Edge case: if limit is extremely small (smaller than window), just keep last 'limit'
            if limit <= valid_window:
                mask[end - limit:end] = True
                continue
                
            # 2. Compute Selection for History
            # We want to keep `limit - valid_window` tokens from the history part.
            capacity_history = limit - valid_window
            
            # Prepare Queries (Observation Window) -> [H, W, D]
            # Select last `valid_window` queries
            q_window = q[end - valid_window:end].transpose(0, 1)
            
            # Prepare Keys (All) -> [H, L, D]
            # Since nanovllm uses packed KV, we extract the sequence slice.
            k_seq = k[start:end]
            # Handle GQA/MQA by repeating keys to match query heads if needed
            if Hkv != H:
                rep = H // Hkv
                k_seq = k_seq.repeat_interleave(rep, dim=1)
            # Transpose to [H, D, L] for matmul
            k_seq_t = k_seq.transpose(0, 1).transpose(1, 2)
            
            # Compute Attention Scores [H, W, L]
            # Queries attend to all keys in the sequence
            attn_scores = torch.matmul(q_window, k_seq_t) * scale
            
            # Apply Causal Mask to the window-window interaction
            # The window queries are at absolute positions [L-W, ..., L-1]
            # The keys are at absolute positions [0, ..., L-1]
            # We need to mask where key_pos > query_pos
            # This only affects the last `valid_window` columns of the keys
            big_neg = torch.finfo(attn_scores.dtype).min
            window_causal_mask = torch.triu(torch.ones(valid_window, valid_window, device=device, dtype=torch.bool), diagonal=1)
            attn_scores[:, :, -valid_window:].masked_fill_(window_causal_mask.unsqueeze(0), big_neg)
            
            # Compute Softmax results [H, W, L]
            attn_probs = torch.softmax(attn_scores, dim=-1)
            
            # 3. Calculate Importance
            # We only care about the history part for selection (keys 0 to L-W-1)
            history_len = L - valid_window
            # Slice to get history probs [H, W, History_Len]
            history_probs = attn_probs[:, :, :history_len]
            
            # Sum over observation window queries -> [H, History_Len]
            importance_per_head = history_probs.sum(dim=1)
            
            # Aggregate across heads (since we store KV slots for all heads at once) -> [History_Len]
            importance_global = importance_per_head.sum(dim=0)
            
            # 4. Pooling (AvgPool or MaxPool) with kernel_size
            # Input needs to be [Minibatch, Channels, Length] -> [1, 1, History_Len]
            imp_input = importance_global.view(1, 1, -1)
            
            if pooling == 'avgpool':
                imp_pooled = F.avg_pool1d(imp_input, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
            elif pooling == 'maxpool':
                imp_pooled = F.max_pool1d(imp_input, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
            else:
                imp_pooled = imp_input # fallback
            
            imp_pooled = imp_pooled.squeeze() # [History_Len]
            
            # 5. Top-K Selection
            k_val = min(capacity_history, imp_pooled.numel())
            if k_val > 0:
                topk = torch.topk(imp_pooled, k=k_val, largest=True)
                keep_indices = topk.indices # Indices relative to start of sequence
                mask[start + keep_indices] = True
                
            # 6. Always keep the Observation Window
            mask[end - valid_window : end] = True
            
        return mask

    def build_compact_slot_mapping(self, context, keep_mask: torch.Tensor) -> torch.Tensor:
        cu = context.cu_seqlens_q
        slot_mapping = context.slot_mapping
        compact_slots = []
        for i in range(cu.numel() - 1):
            start = cu[i].item()
            end = cu[i + 1].item()
            num_kept = int(keep_mask[start:end].sum().item())
            if num_kept == 0:
                continue
            compact_slots.append(slot_mapping[start:start + num_kept])
        if compact_slots:
            return torch.cat(compact_slots)
        return torch.empty(0, device=slot_mapping.device, dtype=slot_mapping.dtype)
