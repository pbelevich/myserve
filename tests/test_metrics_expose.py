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
async def test_metrics_endpoint(started_app):
    transport = ASGITransport(app=started_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # a couple of new names should be present
        assert "myserve_queue_wait_seconds_bucket" in body
        assert "myserve_decode_batch_size_bucket" in body
        assert "myserve_inflight_requests" in body
