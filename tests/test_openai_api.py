import pytest
import httpx
import torch
from httpx import ASGITransport
from openai import AsyncOpenAI
from asgi_lifespan import LifespanManager
from myserve.main import app
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.tokenizer import render_messages

@pytest.fixture()
async def started_app():
    # Starts FastAPI lifespan (startup) before tests and runs shutdown after
    async with LifespanManager(app):
        yield app

@pytest.fixture()
async def openai_client(started_app):
    """
    Async OpenAI client that sends requests into the FastAPI app in-memory.
    Works with httpx>=0.28 where ASGITransport is async-only.
    """
    transport = ASGITransport(app=started_app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    client = AsyncOpenAI(
        base_url="http://test/v1",   # include /v1 if your routes live there
        api_key="test-key",
        http_client=http_client,
    )
    try:
        yield client
    finally:
        # Close both the client and transport cleanly
        await http_client.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_chat_completions_basic(openai_client: AsyncOpenAI):
    resp = await openai_client.chat.completions.create(
        model="gpt2",
        messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        temperature=0,
        max_tokens=10,
    )

    assert resp.id
    assert resp.object in {"chat.completion", "chat.completion.chunk"}
    assert resp.choices and resp.choices[0].message
    text = resp.choices[0].message.content
    assert isinstance(text, str) and len(text) > 0
    assert "Paris" in text


@pytest.mark.asyncio
async def test_chat_completions_stream(openai_client: AsyncOpenAI):
    stream = await openai_client.chat.completions.create(
        model="gpt2",
        messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        stream=True,
        temperature=0,
        max_tokens=10,
    )

    chunks = []
    try:
        async for event in stream:
            for choice in event.choices:
                if getattr(choice, "delta", None) and choice.delta.content:
                    chunks.append(choice.delta.content)
    finally:
        # close stream if the SDK exposes aclose (newer versions do)
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()

    out = "".join(chunks)
    assert "Paris" in out

@pytest.mark.asyncio
async def test_chat_completions_titanic(openai_client: AsyncOpenAI):
    model_id = "gpt2"
    messages = [{"role": "user", "content": "Tell me about Titanic"}]
    max_new_tokens = 100

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    prompt = render_messages(tokenizer, messages)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tid for tid in {tokenizer.eos_token_id, eot_id} if tid is not None]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,        # set False for deterministic (greedy) decoding
        eos_token_id=eos_ids,  # stop at chat turn end
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        output_ids = model.generate(inputs, **gen_kwargs)

    generated = output_ids[0, inputs.shape[-1]:]
    hf_generated = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    input_ids = inputs
    past_key_values = None

    B = input_ids.size(0)
    device = input_ids.device
    generated = torch.empty((B, 0), dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = out.past_key_values

            # Greedy next token for each batch item -> shape [B]
            next_ids = out.logits[:, -1, :].argmax(dim=-1)

            if int(next_ids[0]) in eos_ids:
                break

            # Append as [B, 1], not [1, 1]
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)

            # Feed only the last token back, shape [B, 1] (not [B])
            input_ids = next_ids.unsqueeze(1)

    # Decode per batch
    my_generated = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    assert my_generated == hf_generated, f"{my_generated=}\n{hf_generated=}"

    resp = await openai_client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    assert resp.id
    assert resp.object in {"chat.completion", "chat.completion.chunk"}
    assert resp.choices and resp.choices[0].message
    text = resp.choices[0].message.content
    assert isinstance(text, str) and len(text) > 0

    assert text == my_generated, f"{text=}\n\n\n{my_generated=}"
