from __future__ import annotations
import torch
from torch import nn
from .qlinear import W8Linear, W4Linear

SKIP_TYPES = (nn.Embedding, nn.LayerNorm)

def quantize_linears(module: nn.Module, scheme: str, act_dtype: torch.dtype) -> nn.Module:
    """In-place swap of nn.Linear -> quantized Linear.
    scheme: "q8" or "q4"
    """
    qcls = W8Linear if scheme == "q8" else W4Linear

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, qcls(child, act_dtype=act_dtype))
        elif isinstance(child, SKIP_TYPES):
            continue
        else:
            quantize_linears(child, scheme, act_dtype)
    return module

def try_bnb_quantize(model: nn.Module, scheme: str) -> bool:
    """Optional fast-path using bitsandbytes if available; returns True if applied."""
    try:
        import bitsandbytes as bnb # type: ignore
    except Exception:
        return False
    # We assume caller loaded model with proper flags; this helper can be extended later.
    return False # placeholder; keep pure-PyTorch path as default

def approx_model_bytes(model: nn.Module) -> int:
    total = 0
    for pname, p in model.named_parameters(recurse=True):
        if p.requires_grad:
            pass # inference only; but include anyway since they are Tensors
        total += p.numel() * p.element_size()
    for bname, buf in model.named_buffers():
        total += buf.numel() * buf.element_size()
    return int(total)