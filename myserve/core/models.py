from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass(frozen=True)
class ModelBundle:
    tokenizer: any
    model: any
    device: torch.device
    dtype: torch.dtype

class ModelRegistry:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, str], ModelBundle] = {}

    def load(self, model_name: str, dtype: str = "auto", device: str = "auto") -> ModelBundle:
        key = (model_name, dtype, device)
        if key in self._cache:
            return self._cache[key]

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        
        # Resolve dtype
        if dtype == "auto":
            torch_dtype = None  # let HF pick
        elif dtype.lower() in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype.lower() in ("fp16", "float16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=os.environ.get("TRUST_REMOTE_CODE", "0") == "1",
        )
        model.to(dev)
        model.eval()

        bundle = ModelBundle(tokenizer=tok, model=model, device=dev, dtype=model.dtype)
        self._cache[key] = bundle
        return bundle

REGISTRY = ModelRegistry()
