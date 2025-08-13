from __future__ import annotations
from typing import List, Optional
import torch
from torch import nn

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
