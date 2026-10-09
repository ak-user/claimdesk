from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimdesk.api import create_app
from claimdesk.ratelimit import RateLimiter, new_request_id


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_bucket_allows_a_burst_then_refills():
    clock = Clock()
    limiter = RateLimiter(per_minute=3, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.check("a") == pytest.approx(20.0)  # one token every 20 seconds
    assert limiter.check("b") == 0.0  # clients don't share a bucket
    clock.now += 20
    assert limiter.check("a") == 0.0
    assert limiter.check("a") > 0


def test_zero_disables_the_limit():
    limiter = RateLimiter(per_minute=0)
    assert not limiter.enabled
    assert all(limiter.check("a") == 0.0 for _ in range(100))


def test_request_ids():
    assert new_request_id("abc-123") == "abc-123"
    generated = new_request_id("bad id\nwith newline")
    assert len(generated) == 16 and generated.isalnum()
    assert new_request_id(None) != new_request_id(None)


@pytest.fixture(scope="module")
def client(settings):
    update = {"claimdesk_static_dir": Path("/nonexistent"), "claimdesk_rate_limit_per_minute": 2}
    return TestClient(create_app(settings.model_copy(update=update)))


def test_api_limits_model_routes_only(client):
    body = {"question": "How long do I have to submit a claim?"}
    assert client.post("/api/ask", json=body).status_code == 200
    assert client.post("/api/ask", json=body).status_code == 200
    limited = client.post("/api/ask", json=body)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    # Reads and examiner decisions never call the model, so they aren't limited.
    assert all(client.get("/api/health").status_code == 200 for _ in range(5))
    review = {"decision": "approved", "reviewer": "J. Okafor"}
    assert client.post("/api/claims/9999/review", json=review).status_code == 404


def test_api_echoes_request_ids(client):
    response = client.get("/api/health", headers={"X-Request-ID": "trace-42"})
    assert response.headers["X-Request-ID"] == "trace-42"
    assert len(client.get("/api/health").headers["X-Request-ID"]) == 16
