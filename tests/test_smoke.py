import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from myserve.main import app

@pytest.fixture()
async def started_app():
    # Starts FastAPI lifespan (startup) before tests and runs shutdown after
    async with LifespanManager(app):
        yield app

@pytest.mark.asyncio
async def test_non_stream_basic(started_app):
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "model": "gpt2",
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a test."},
                {"role": "user", "content": "Hello world"},
            ],
        }
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)

@pytest.mark.asyncio
async def test_streaming_basic_sse(started_app):
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "model": "gpt2",
            "stream": True,
            "messages": [
                {"role": "user", "content": "stream me"}
            ],
        }
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        text = r.text
        # Should end with [DONE]
        assert text.strip().endswith("data: [DONE]")
        # First event should include role preamble
        assert '"object": "chat.completion.chunk"' in text
