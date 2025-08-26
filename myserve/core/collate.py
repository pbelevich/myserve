from __future__ import annotations
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence

# Types
Past = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

def pad_past(pasts: List[Past]) -> Tuple[Past, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of past_kv to a common T_max.
    Returns: (padded_past, lengths[B], attention_mask[B, T_max+1], position_ids[B,1])
    """
    L = len(pasts[0])
    B = len(pasts)
    lengths = torch.tensor([p[0][0].shape[2] for p in pasts], dtype=torch.long)  # T_i
    Tm = int(lengths.max().item())

    padded: List[Tuple[torch.Tensor, torch.Tensor]] = []
    device = pasts[0][0][0].device

    for layer in range(L):
        ks = []
        vs = []
        for b in range(B):
            k, v = pasts[b][layer]
            t = k.shape[2]
            if t == Tm:
                ks.append(k)
                vs.append(v)
            else:
                pad_k = torch.zeros((k.shape[0], k.shape[1], Tm - t, k.shape[3]), device=k.device, dtype=k.dtype)
                pad_v = torch.zeros_like(pad_k)
                ks.append(torch.cat([pad_k, k], dim=2))
                vs.append(torch.cat([pad_v, v], dim=2))
        K = torch.cat(ks, dim=0)  # [B,H,Tm,D]
        V = torch.cat(vs, dim=0)
        padded.append((K, V))

    # attention mask: ones for valid tokens incl. new token slot
    attn = torch.zeros((B, Tm + 1), dtype=torch.long, device=device)
    for b in range(B):
        # attn[b, : lengths[b] + 1] = 1
        attn[b, -lengths[b] - 1 : ] = 1
    # position ids: next index per request
    pos = lengths.view(B, 1).to(device)
    return tuple(padded), lengths.to(device), attn, pos


def split_past(padded: Past, lengths: torch.Tensor) -> List[Past]:
    """Split padded past into per-request past using lengths (after a decode step)."""
    B = int(lengths.numel())
    L = len(padded)
    out: List[Past] = []
    for b in range(B):
        layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(L):
            K, V = padded[l]
            t = int(lengths[b].item())  # after step
            k = K[b : b + 1, :, -t :, :].contiguous()
            v = V[b : b + 1, :, -t :, :].contiguous()
            layers.append((k, v))
        out.append(tuple(layers))
    return out

def pad_sequences(input_ids: List[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id, padding_side="left")
    lengths = torch.tensor([len(s) for s in input_ids], device=inputs_ids.device)  # [B]
    max_len = int(lengths.max())
    attention_mask = (torch.arange(max_len, device=inputs_ids.device) >= (max_len - lengths.unsqueeze(1))).long()  # [B, L]
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    assert position_ids.device == inputs_ids.device == attention_mask.device, f"{position_ids.device=} {inputs_ids.device=} {attention_mask.device=}"
    return inputs_ids, attention_mask, position_ids, lengths
