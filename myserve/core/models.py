from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.quant.convert import quantize_linears
from myserve.core.quant.convert import approx_model_bytes
from myserve.metrics import MODEL_BYTES

DEFAULT_DTYPE = os.getenv("MYSERVE_DTYPE", "auto")
DEFAULT_DEVICE = os.getenv("MYSERVE_DEVICE", "auto")

@dataclass(frozen=True)
class ModelBundle:
    tokenizer: any
    model: any
    device: torch.device
    dtype: torch.dtype

class ModelRegistry:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, str], ModelBundle] = {}

    def load(self, model_name: str, dtype: str = DEFAULT_DTYPE, device: str = DEFAULT_DEVICE) -> ModelBundle:
        key = (model_name, dtype, device)
        if key in self._cache:
            return self._cache[key]

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        
        # Resolve dtype
        if dtype == "auto":
            torch_dtype = "auto"  # let HF pick
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

        # quantize (weight‑only)
        if dtype.lower() in ("q8", "q4"):
            act_dtype = model.dtype # torch.bfloat16 if torch_dtype in (torch.bfloat16, None) else torch.float16
            quantize_linears(model, dtype.lower(), act_dtype)
            print(f"Quantized model to {dtype.lower()} with act_dtype {act_dtype}")

        MODEL_BYTES.labels(model=model_name).set(approx_model_bytes(model))

        bundle = ModelBundle(tokenizer=tok, model=model, device=dev, dtype=model.dtype)
        self._cache[key] = bundle
        return bundle

REGISTRY = ModelRegistry()
