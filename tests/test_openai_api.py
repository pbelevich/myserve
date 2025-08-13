import pytest
import httpx
from httpx import ASGITransport
from openai import AsyncOpenAI

from myserve.main import app


@pytest.fixture()
async def openai_client():
    """
    Async OpenAI client that sends requests into the FastAPI app in-memory.
    Works with httpx>=0.28 where ASGITransport is async-only.
    """
    transport = ASGITransport(app=app)
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
    assert "Paris" == text


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
    assert out == "Paris"
