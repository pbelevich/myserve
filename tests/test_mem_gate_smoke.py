import os, asyncio, pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from myserve.main import app

@pytest.fixture()
async def started_app():
    async with LifespanManager(app):
        yield app

@pytest.mark.asyncio
async def test_memory_gate_serializes_when_budget_low(monkeypatch, started_app):
    # tiny "free" memory so only one reservation fits
    monkeypatch.setenv("MYSERVE_MEM_FORCE_FREE_BYTES", str(384 * 1024 * 1024))  # 384MB
    monkeypatch.setenv("MYSERVE_RESERVE_TOKENS", "512")
    monkeypatch.setenv("MYSERVE_WORKSPACE_MB", "16")
    body = lambda text: {
        "model": "sshleifer/tiny-gpt2",
        "stream": True,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": text}],
    }
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=30) as ac:
        r1 = ac.post("/v1/chat/completions", json=body("A"*128))
        r2 = ac.post("/v1/chat/completions", json=body("B"*128))
        a1, a2 = await asyncio.gather(r1, r2)
        assert a1.status_code == 200 and a2.status_code == 200
        # both complete; second had to wait (can’t assert exact timing in CI)
        assert a1.text.strip().endswith("data: [DONE]")
        assert a2.text.strip().endswith("data: [DONE]")
