import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from myserve.core.generate import greedy_generate

MODEL = "sshleifer/tiny-gpt2"  # tiny, CPU‑friendly

@pytest.mark.parametrize("prompt", [
    "Hello world",
    "The capital of France is",
])
@torch.no_grad()
def test_next_token_parity(prompt):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    # our greedy (1 token)
    ours = greedy_generate(model, input_ids, max_new_tokens=1)

    # HF greedy (1 token)
    hf = model.generate(
        **enc,
        max_new_tokens=1,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    assert torch.equal(ours, hf), "our greedy tokens should match HF for 1 token"