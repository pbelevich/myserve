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
async def test_q8_works_e2e(monkeypatch, started_app):
    monkeypatch.setenv("MYSERVE_DTYPE", "q8")
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "model": "sshleifer/tiny-gpt2",
            "stream": False,
            "messages": [{"role": "user", "content": "Quantized hello"}],
            "max_tokens": 8,
        }
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"]
