from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from .sampling import SamplerCfg, sample_next

@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Extremely simple greedy loop: one token at a time, no KV cache.
    input_ids: [B, T] on the correct device.
    Returns: [B, T + new] tokens.
    """
    model.eval()
    bsz = input_ids.size(0)
    out = input_ids
    for _ in range(max_new_tokens):
        # full forward each step (slow!)
        logits = model(out).logits  # [B, T, V]
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]
        next_token = next_token.unsqueeze(-1)                # [B,1]
        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None:
            # Stop early only if *all* sequences ended
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break
    return out

@torch.no_grad()
def sample_generate(
    model: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    cfg: SamplerCfg,
    gen: Optional[torch.Generator] = None,
    collect_logprobs: bool = False,
):
    """Token‑by‑token sampling. Returns (all_ids, per_step) where per_step is a list of dicts
    with keys: {"id": int, "logprob": float, "top_logprobs": List[(id, logprob)]} when requested.
    """
    model.eval()
    B = input_ids.size(0)
    out = input_ids
    per_step = []
    for _ in range(max_new_tokens):
        logits = model(out).logits[:, -1, :]  # [B, V]
        next_ids, chosen_logprobs, logprobs = sample_next(logits, cfg, out, gen)
        out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)
        if collect_logprobs:
            k = int(cfg.top_logprobs or 0)
            step = []
            for b in range(B):
                item = {"id": int(next_ids[b]), "logprob": float(chosen_logprobs[b])}
                if k > 0:
                    topv, topi = torch.topk(logprobs[b], k)
                    item["top_logprobs"] = [(int(topi[j]), float(topv[j])) for j in range(topv.numel())]
                step.append(item)
            per_step.append(step)
        if eos_token_id is not None and torch.all(next_ids == eos_token_id):
            break
    return out, per_step
