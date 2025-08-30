import pytest
from transformers import AutoModelForCausalLM


@pytest.mark.parametrize("impl", ["sdpa", "eager"]) # FA2 requires wheels/GPU; keep CI robust
def test_model_loads_with_attn_impl(monkeypatch, impl):
    monkeypatch.setenv("MYSERVE_ATTN_IMPL", impl)
    m = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", attn_implementation=impl)
    assert m is not None
