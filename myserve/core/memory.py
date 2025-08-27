from __future__ import annotations
import os, torch
from dataclasses import dataclass, field
from typing import Dict

def _dtype_nbytes(dt: torch.dtype) -> int:
    return {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int8: 1, torch.uint8: 1, torch.float64: 8
    }.get(dt, 4)

def _cfg_heads(cfg) -> int:
    return int(getattr(cfg, "num_key_value_heads",
                       getattr(cfg, "n_kv_heads",
                       getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))))

def _cfg_layers(cfg) -> int:
    return int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))

def _cfg_hidden(cfg) -> int:
    return int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))

def _kv_bytes_per_token(cfg, dtype: torch.dtype) -> int:
    # per token per layer: 2 * kv_heads * head_dim * bytes
    kv_heads = _cfg_heads(cfg)
    head_dim = _cfg_hidden(cfg) // int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", kv_heads)))
    per_layer = 2 * kv_heads * head_dim * _dtype_nbytes(dtype)
    return _cfg_layers(cfg) * per_layer  # sum over layers

def _free_bytes(device: torch.device) -> int:
    forced = os.getenv("MYSERVE_MEM_FORCE_FREE_BYTES")
    if forced:  # for tests
        return int(forced)
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device.index or 0)
        return int(free_b)
    # CPU: treat as effectively unlimited
    return 1 << 62

@dataclass
class Reservation:
    req_id: str
    bytes: int

@dataclass
class MemManager:
    model_name: str
    device: torch.device
    cfg: object     # HF config
    dtype: torch.dtype
    mem_fraction: float = float(os.getenv("MYSERVE_MEM_FRACTION", "0.90"))
    reserve_tokens_cap: int = int(os.getenv("MYSERVE_RESERVE_TOKENS", "256"))
    workspace_mb: int = int(os.getenv("MYSERVE_WORKSPACE_MB", "64"))
    safety_mb: int = int(os.getenv("MYSERVE_SAFETY_MB", "256"))
    _reserved_total: int = 0
    _by_req: Dict[str, Reservation] = field(default_factory=dict)

    @property
    def kv_bytes_per_token(self) -> int:
        return _kv_bytes_per_token(self.cfg, self.dtype)

    def _budget(self) -> int:
        # use a fraction of current free memory; subtract a safety buffer
        free_now = _free_bytes(self.device)
        safety = self.safety_mb * 1024 * 1024
        return max(0, int(self.mem_fraction * free_now) - safety)

    def _estimate_bytes_for_request(self, prompt_len: int, max_new: int) -> int:
        runway = min(int(max_new), int(self.reserve_tokens_cap))
        kv_tokens = prompt_len + runway
        kv_bytes = kv_tokens * self.kv_bytes_per_token
        workspace = self.workspace_mb * 1024 * 1024
        return kv_bytes + workspace

    def can_reserve(self, req_id: str, prompt_len: int, max_new: int) -> bool:
        need = self._estimate_bytes_for_request(prompt_len, max_new)
        return (self._reserved_total + need) <= self._budget()

    def reserve(self, req_id: str, prompt_len: int, max_new: int) -> bool:
        need = self._estimate_bytes_for_request(prompt_len, max_new)
        if (self._reserved_total + need) > self._budget():
            return False
        self._reserved_total += need
        self._by_req[req_id] = Reservation(req_id, need)
        return True

    def release(self, req_id: str) -> None:
        r = self._by_req.pop(req_id, None)
        if r:
            self._reserved_total = max(0, self._reserved_total - r.bytes)

    def reserved_bytes(self) -> int:
        return self._reserved_total
