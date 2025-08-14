from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from .kv import KVCache, PastKeyValues
from transformers import DynamicCache

@torch.no_grad()
def prefill(model: nn.Module, input_ids: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
    """Run the full prompt once to build the KV cache.
    Returns last-position logits [B,V] and a KVCache holding past_key_values.
    """
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past: PastKeyValues = out.past_key_values  # tuple(layer) of (k,v)
    return logits, KVCache.from_past(past)

@torch.no_grad()
def decode_step(model: nn.Module, last_token: torch.Tensor, kv: KVCache) -> Tuple[torch.Tensor, KVCache]:
    """One decode step: feed only the last token, reuse cached KV.
    last_token: [B,1], kv: KVCache with same B.
    Returns last-position logits [B,V] and the updated cache.
    """
    out = model(input_ids=last_token, past_key_values=DynamicCache.from_legacy_cache(kv.to_past()), use_cache=True)
    logits = out.logits[:, -1, :]
    new_past: PastKeyValues = out.past_key_values
    return logits, KVCache.from_past(new_past)
