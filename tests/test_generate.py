import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from myserve.core.generate import greedy_generate

@torch.no_grad()
def test_next_token_parity():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    messages = [
        {"role": "user", "content": "What is the capital of France? Answer with one word."}
    ]
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")  # present in Llama 3 chat models
    eos_ids = [tok.eos_token_id, eot_id] if eot_id is not None and eot_id != tok.eos_token_id else [tok.eos_token_id]

    messages = [
        {"role": "user", "content": "What is the capital of France? Answer with one word."},
        {"role": "assistant", "content": "Paris"}
    ]
    expected = tok.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    generated = greedy_generate(model, inputs, max_new_tokens=100, eos_token_id=eos_ids[0])
    assert torch.equal(generated, expected), f"{tok.decode(generated[0], skip_special_tokens=False)=}, {tok.decode(expected[0], skip_special_tokens=False)=}"