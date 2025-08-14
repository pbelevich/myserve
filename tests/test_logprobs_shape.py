from httpx import AsyncClient, ASGITransport
import pytest
from myserve.main import app

@pytest.mark.asyncio
async def test_logprobs_shape():
    body = {
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": 3,
        "seed": 42,
        "stream": False,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert "logprobs" in data and "content" in data["logprobs"]
        assert len(data["logprobs"]["content"]) > 0
        first = data["logprobs"]["content"][0]
        assert "token" in first and "logprob" in first
        assert isinstance(first.get("top_logprobs", []), list)
