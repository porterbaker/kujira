import json

import httpx
import pytest

from cfb.api.client import CFBDClient, CFBDError
from cfb.config import ApiSettings

KEY = "test-key-abc"


def make_client(handler, **settings):
    sleeps: list[float] = []
    client = CFBDClient(
        KEY,
        ApiSettings(min_interval_s=0.0, backoff_min_s=0.01, backoff_max_s=0.05, **settings),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    return client, sleeps


def test_bearer_auth_and_params():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(200, json=[{"id": 1}])

    client, _ = make_client(handler)
    result = client.get("/plays", {"year": 2024, "week": 1, "team": None})
    assert result.ok and result.json() == [{"id": 1}]
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["url"] == "https://api.collegefootballdata.com/plays?year=2024&week=1"
    assert KEY not in seen["url"]
    assert result.attempts == 1 and client.calls_made == 1


def test_retries_429_honoring_retry_after_then_succeeds():
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "0.02"}),
        httpx.Response(503),
        httpx.Response(200, json=[]),
    ])
    client, sleeps = make_client(lambda request: next(responses))
    result = client.get("/games", {"year": 2024})
    assert result.ok
    assert result.attempts == 3 and client.calls_made == 3  # retries count as calls
    assert sleeps[0] == pytest.approx(0.02)


def test_gives_up_after_max_attempts_and_returns_last_error():
    client, _ = make_client(lambda request: httpx.Response(500, text="boom"), max_attempts=3)
    result = client.get("/games")
    assert result.status == 500 and result.attempts == 3


def test_client_errors_are_not_retried():
    client, _ = make_client(lambda request: httpx.Response(400, text="bad params"))
    result = client.get("/plays", {"year": 2024})
    assert result.status == 400 and result.attempts == 1


def test_transport_errors_retried_then_raise_without_key():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    client, _ = make_client(handler, max_attempts=2)
    with pytest.raises(CFBDError) as exc:
        client.get("/games")
    assert "2 attempts" in str(exc.value)
    assert KEY not in str(exc.value)
    assert client.calls_made == 2


def test_throttle_spaces_calls():
    clock = iter([0.0, 0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    sleeps: list[float] = []
    client = CFBDClient(
        KEY,
        ApiSettings(min_interval_s=0.5),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])),
        sleep=sleeps.append,
        clock=lambda: next(clock),
    )
    client.get("/a")
    client.get("/b")
    assert sleeps == [pytest.approx(0.4)]


def test_info_null_means_bad_key():
    client, _ = make_client(lambda request: httpx.Response(200, content=b"null"))
    with pytest.raises(CFBDError, match="not accepted"):
        client.info()


def test_body_kept_verbatim():
    body = json.dumps([{"ppa": 0.1234567890123}]).encode()
    client, _ = make_client(lambda request: httpx.Response(200, content=body))
    assert client.get("/plays").content == body
