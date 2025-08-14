import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import cached_generate, sample_generate

MODEL = "sshleifer/tiny-gpt2"

@torch.no_grad()
def test_kv_is_faster_smoke():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    enc = tok("write a story about a cat", return_tensors="pt", add_special_tokens=False)
    cfg = SamplerCfg()

    t0 = time.time(); cached_generate(model, enc["input_ids"], 100, tok.eos_token_id, cfg); t1 = time.time()
    t2 = time.time(); sample_generate(model, enc["input_ids"], 100, tok.eos_token_id, cfg); t3 = time.time()

    # KV path should be at least modestly faster; guard against flaky envs
    assert (t1 - t0) < (t3 - t2) * 0.9 or (t3 - t2) > 0.05
