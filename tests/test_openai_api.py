import pytest
import httpx
import torch
from httpx import ASGITransport
from openai import AsyncOpenAI
from asgi_lifespan import LifespanManager
from myserve.main import app
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        temperature=0,
        max_tokens=5,
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
        model="meta-llama/Llama-3.2-1B-Instruct",
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

titanic_expected = 'The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning of April 15, 1912, after colliding with an iceberg during her maiden voyage from Southampton to New York City. The tragedy resulted in the loss of over 1,500 lives and became one of the deadliest maritime disasters in history.\n\nHere are some key facts about the Titanic:\n\n**Design and Construction**\n\nThe Titanic was built by the Harland and Wolff shipyard in Belfast, Ireland, and was designed by Alexander Carlisle and William Pirrie. It was the largest ship in the world at the time, measuring over 882 feet (270 meters) long and 92 feet (28 meters) wide. The Titanic was powered by four high-pressure steam engines and had a top speed of around 21 knots (24 mph).\n\n**Maiden Voyage**\n\nThe Titanic began its maiden voyage from Southampton, England, on April 10, 1912, bound for New York City. On board were some of the most prominent people of the time, including millionaires, politicians, and royalty. The ship was considered unsinkable, with a double-bottom hull and 16 watertight compartments that could supposedly keep the ship afloat even if four of them were flooded.\n\n**Iceberg Collision**\n\nOn the night of April 14, 1912, the Titanic received several warnings of icebergs in the area, but the crew did not take adequate precautions. At 11:40 PM, the ship struck an iceberg on its starboard (right) side. The collision caused significant damage to the ship\'s hull, but it was not immediately apparent how severe the damage was.\n\n**Sinking**\n\nOver the next two hours, the Titanic took on water and began to list (tilt) to one side. The crew sent out distress signals, but they were not received in time to prevent the ship from sinking. At 2:20 AM on April 15, 1912, the Titanic finally slipped beneath the surface of the ocean, taking over 1,500 people with it.\n\n**Rescue Efforts**\n\nThe crew of the RMS Carpathia, which had arrived in the area the night before, received the Titanic\'s distress signals and began to rescue survivors. The Carpathia took on over 700 survivors and provided them with food, clothing, and medical care. The rescue efforts took several days, and many survivors were left stranded on the sinking ship.\n\n**Investigation and Legacy**\n\nThe sinking of the Titanic was investigated by a British inquiry, which concluded that a combination of factors contributed to the disaster, including:\n\n* Excessive speed in an area known to have icebergs\n* Insufficient lookout and warning systems\n* Inadequate lifeboat capacity and training\n* Design flaws in the ship\'s hull\n\nThe Titanic\'s legacy is still felt today, with many considering it a cautionary tale about the dangers of hubris and the importance of safety at sea. The ship\'s story has been immortalized in numerous books, films, and other works of art, and it continues to fascinate people around the world.\n\n**Interesting Facts**\n\n* The Titanic was on its maiden voyage when it set a new record for the fastest transatlantic crossing, traveling from Southampton to New York in just over 7 hours.\n* The ship\'s band played music for over 2 hours after the collision, including a rendition of "Nearer, My God, to Thee."\n* The Titanic\'s grand staircase was considered one of the most impressive in the world at the time.\n* The ship\'s lookouts were not adequately trained to spot icebergs, and the crew did not have access to binoculars or other tools to help them detect the danger.'

@pytest.mark.asyncio
async def test_chat_completions_titanic(openai_client: AsyncOpenAI):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    messages = [{"role": "user", "content": "Tell me about Titanic"}]
    max_new_tokens = 300

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

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

    # assert hf_generated == titanic_expected, f"{hf_generated=}\n\n\n{titanic_expected=}"

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
