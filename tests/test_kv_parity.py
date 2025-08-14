import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import cached_generate
from myserve.core.generate import sample_generate  # from Post 3

MODEL = "sshleifer/tiny-gpt2"

@torch.no_grad()
def test_cached_equals_uncached_one_seed():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    cfg = SamplerCfg(temperature=1.0, top_p=1.0)
    gen = torch.Generator(device="cpu").manual_seed(123)

    ids_cached, _ = cached_generate(model, enc["input_ids"], 8, tok.eos_token_id, cfg, gen)

    # reinit generator for identical draw
    gen2 = torch.Generator(device="cpu").manual_seed(123)
    ids_uncached, _ = sample_generate(model, enc["input_ids"], 8, tok.eos_token_id, cfg, gen2, collect_logprobs=False)

    assert torch.equal(ids_cached, ids_uncached)
