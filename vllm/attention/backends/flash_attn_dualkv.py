# SPDX-License-Identifier: Apache-2.0
"""DualKV attention backend for flash attention.

This module provides a DualKV decode path that uses contiguous context + decoded
KV caches instead of vLLM's paged KV cache. The prefill path is unchanged.

Usage:
    Set VLLM_USE_DUALKV=1 to enable dualkv decode path.
    Requires the `flash_attn` pip package (dualkv fork) to be installed.

The DualKV approach splits the KV cache into:
  - context_kv: (1, ctx_len, nheads_k, hdim) — shared across all seqs in batch
  - decoded_kv: (bs, max_dec_len, nheads_k, hdim) — per-sequence decoded tokens

This saves memory when all sequences share a long common prompt (e.g., RL rollout).

Bypass mode: context KV is captured directly from the prefill key/value tensors,
bypassing the paged KV cache entirely. This avoids the paged→contiguous gather
and is simpler and more correct.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from flash_attn import flash_attn_with_kvcache

logger = logging.getLogger(__name__)

DUALKV_ENABLED = os.environ.get("VLLM_USE_DUALKV", "0") == "1"


class DualKVCacheState:
    """Per-layer state for DualKV contiguous KV caches."""

    def __init__(self):
        self.context_k: Optional[torch.Tensor] = None  # (1, ctx_len, nheads_k, hdim)
        self.context_v: Optional[torch.Tensor] = None
        self.decoded_k: Optional[torch.Tensor] = None  # (bs, max_dec_len, nheads_k, hdim)
        self.decoded_v: Optional[torch.Tensor] = None
        self.context_seqlen: int = 0
        self.decoded_seqlens: Optional[torch.Tensor] = None  # (bs,), int32
        self.initialized: bool = False

    def reset(self):
        self.context_k = None
        self.context_v = None
        self.decoded_k = None
        self.decoded_v = None
        self.context_seqlen = 0
        self.decoded_seqlens = None
        self.initialized = False


# Global registry: layer_id -> DualKVCacheState
_dualkv_states: Dict[int, DualKVCacheState] = {}


def get_dualkv_state(layer_id: int) -> DualKVCacheState:
    if layer_id not in _dualkv_states:
        _dualkv_states[layer_id] = DualKVCacheState()
    return _dualkv_states[layer_id]


def reset_all_dualkv_states():
    """Call this at the start of each generation batch."""
    for state in _dualkv_states.values():
        state.reset()


def capture_context_kv(
    key: torch.Tensor,      # (total_prefill_tokens, nheads_k, hdim) — flat
    value: torch.Tensor,    # (total_prefill_tokens, nheads_k, hdim) — flat
    first_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture the first sequence's prefill key/value as shared context.

    Since all sequences share the same prompt, we only need the first one.
    Returns (context_k, context_v) each of shape (1, first_seq_len, nheads_k, hdim).
    """
    context_k = key[:first_seq_len].unsqueeze(0).contiguous()
    context_v = value[:first_seq_len].unsqueeze(0).contiguous()
    return context_k, context_v


def allocate_decoded_kv(
    batch_size: int,
    max_decode_len: int,
    nheads_k: int,
    hdim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate contiguous decoded KV buffers."""
    decoded_k = torch.zeros(batch_size, max_decode_len, nheads_k, hdim,
                           device=device, dtype=dtype)
    decoded_v = torch.zeros(batch_size, max_decode_len, nheads_k, hdim,
                           device=device, dtype=dtype)
    return decoded_k, decoded_v


def dualkv_decode(
    query: torch.Tensor,       # (bs, nheads, hdim) — decode tokens
    new_key: torch.Tensor,     # (bs, nheads_k, hdim) — new KV from model
    new_value: torch.Tensor,   # (bs, nheads_k, hdim)
    state: DualKVCacheState,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Run dualkv decode attention.

    Args:
        query: (bs, nheads, hdim)
        new_key: (bs, nheads_k, hdim)
        new_value: (bs, nheads_k, hdim)
        state: DualKVCacheState with initialized context and decoded buffers

    Returns:
        output: (bs, nheads, hdim)
    """
    # Reshape for flash_attn_with_kvcache: expects (bs, seqlen_q, nheads, hdim)
    q = query.unsqueeze(1)          # (bs, 1, nheads, hdim)
    k_new = new_key.unsqueeze(1)    # (bs, 1, nheads_k, hdim)
    v_new = new_value.unsqueeze(1)  # (bs, 1, nheads_k, hdim)

    out = flash_attn_with_kvcache(
        q=q,
        k_cache_context=state.context_k,     # (1, ctx_len, nheads_k, hdim)
        v_cache_context=state.context_v,
        k_cache_decoded=state.decoded_k,      # (bs, max_dec_len, nheads_k, hdim)
        v_cache_decoded=state.decoded_v,
        k=k_new,
        v=v_new,
        cache_seqlens_context=state.context_seqlen,  # scalar -> broadcast
        cache_seqlens_decoded=state.decoded_seqlens,  # (bs,)
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        num_splits=1,
        use_dualkv_attention=True,
    )

    # Update decoded seqlens (kernel already appended k/v to decoded cache)
    state.decoded_seqlens += 1


    return out.squeeze(1)  # (bs, nheads, hdim)
