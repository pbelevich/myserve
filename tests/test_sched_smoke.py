import asyncio, pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from myserve.main import app

@pytest.fixture()
async def started_app():
    # Starts FastAPI lifespan (startup) before tests and runs shutdown after
    async with LifespanManager(app):
        yield app

@pytest.mark.asyncio
async def test_two_streams_concurrent(started_app):
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = lambda prompt: {
            "model": "sshleifer/tiny-gpt2",
            "stream": True,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": prompt}],
        }
        r1 = ac.post("/v1/chat/completions", json=body("alpha"))
        r2 = ac.post("/v1/chat/completions", json=body("beta"))
        a1, a2 = await asyncio.gather(r1, r2)
        # both should complete and contain [DONE]
        assert a1.status_code == 200 and a2.status_code == 200
        assert a1.text.strip().endswith("data: [DONE]")
        assert a2.text.strip().endswith("data: [DONE]")
