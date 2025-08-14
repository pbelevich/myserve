from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

KVPair = Tuple[torch.Tensor, torch.Tensor]  # (k, v)
PastKeyValues = Tuple[KVPair, ...]          # HF convention per layer

@dataclass
class KVCache:
    layers: PastKeyValues                   # tuple of (k: [B,H,T,D], v: [B,H,T,D]) per layer

    @staticmethod
    def empty_like(past: PastKeyValues) -> "KVCache":
        # create an empty cache with zero-length T on same device/dtype
        new_layers: List[KVPair] = []
        for k, v in past:
            B,H,T,D = k.shape
            device, dtype = k.device, k.dtype
            zk = torch.empty((B,H,0,D), device=device, dtype=dtype)
            zv = torch.empty((B,H,0,D), device=device, dtype=dtype)
            new_layers.append((zk, zv))
        return KVCache(tuple(new_layers))

    @staticmethod
    def from_past(past: PastKeyValues) -> "KVCache":
        return KVCache(tuple((k.contiguous(), v.contiguous()) for (k,v) in past))

    def to_past(self) -> PastKeyValues:
        return tuple((k, v) for (k, v) in self.layers)

    def append_step(self, step_kv: PastKeyValues) -> None:
        # Concatenate new time-step along T for each layer
        new_layers: List[KVPair] = []
        for (k, v), (nk, nv) in zip(self.layers, step_kv):
            new_layers.append((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2)))
        self.layers = tuple(new_layers)

    @property
    def length(self) -> int:
        # sequence length T (assumes non-empty)
        return 0 if len(self.layers) == 0 else int(self.layers[0][0].shape[2])
