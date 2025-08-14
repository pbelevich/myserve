import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import sample_generate

MODEL = "sshleifer/tiny-gpt2"

@pytest.mark.parametrize("prompt,temperature,top_p,top_k", [
    ("Hello", 1.0, 1.0, None),
    ("Hello", 0.7, 1.0, 50),
    ("Hello", 1.0, 0.9, None),
])
@torch.no_grad()
def test_one_token_sampling_parity(prompt, temperature, top_p, top_k):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    inp = enc["input_ids"]

    # our sample (1 token)
    cfg = SamplerCfg(temperature=temperature, top_p=top_p, top_k=top_k)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    ours, _ = sample_generate(model, inp, 1, tok.eos_token_id, cfg, gen, collect_logprobs=False)

    # HF sample (1 token)
    set_seed(1234)
    hf = model.generate(**enc, max_new_tokens=1, do_sample=True, temperature=temperature,
                        top_p=top_p, top_k=(top_k or 0), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    assert torch.equal(ours, hf)
