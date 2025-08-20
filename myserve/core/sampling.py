from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class SamplerCfg:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = 0

@torch.no_grad()
def apply_penalties(
    logits: torch.Tensor,              # [B, V]
    generated_ids: torch.Tensor,       # [B, T] (prefix + already sampled)
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    if presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits
    B, V = logits.shape
    # token counts per batch
    # build counts on CPU for simplicity; move back to device
    counts = torch.zeros((B, V), dtype=torch.int32)
    for b in range(B):
        ids = generated_ids[b].tolist()
        for t in ids:
            if 0 <= t < V:
                counts[b, t] += 1
    counts = counts.to(logits.device)
    # presence penalty subtracts a constant if token ever appeared
    presence_mask = (counts > 0).to(logits.dtype)
    logits = logits - presence_penalty * presence_mask
    # frequency penalty subtracts count * penalty
    logits = logits - frequency_penalty * counts.to(logits.dtype)
    return logits

@torch.no_grad()
def top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int], top_p: float) -> torch.Tensor:
    # logits: [B, V]
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        # shift right to always keep the first token above threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        # unsort
        unsorted = torch.full_like(sorted_logits, float('-inf'))
        unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted
    return logits

@torch.no_grad()
def sample_next(
    logits: torch.Tensor,              # [B, V] last‑step logits
    cfg: SamplerCfg,
    generated_ids: torch.Tensor,       # [B, T]
    gen: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # penalties first (operate in logits space)
    logits = apply_penalties(logits, generated_ids, cfg.presence_penalty, cfg.frequency_penalty)
    # temperature
    if cfg.temperature != 0.0:
        temperature = max(1e-5, float(cfg.temperature))
        logits = logits / temperature
        # filter and normalize
        logits = top_k_top_p_filter(logits, cfg.top_k, cfg.top_p)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)
        # sample
        next_ids = torch.multinomial(probs, num_samples=1, generator=gen)  # [B,1]
    else:
        logprobs = torch.full_like(logits, -float('inf'), dtype=logits.dtype)
        logprobs[torch.arange(logits.shape[0]), logits.argmax(dim=-1)] = 0.0
        next_ids = torch.argmax(logits, dim=-1).unsqueeze(-1) # greedy decoding
    chosen_logprobs = logprobs.gather(-1, next_ids)                    # [B,1]
    return next_ids.squeeze(-1), chosen_logprobs.squeeze(-1), logprobs
