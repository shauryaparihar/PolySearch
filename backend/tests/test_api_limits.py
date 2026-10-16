import threading
import time

import pytest
from api_support import SEARCH
from app.api.errors import ApiError
from app.api.guard import InferenceGuard
from app.api.ratelimit import SlidingWindowLimiter, client_id


def test_limiter_blocks_after_the_limit_and_recovers_as_the_window_slides():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60, clock=lambda: now[0])
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.check("a")
    assert not allowed and retry_after == 60
    now[0] = 59
    assert limiter.check("a") == (False, 1)
    now[0] = 60.5
    assert limiter.check("a")[0] is True  # the first hit left the window
    assert limiter.check("b")[0] is True  # other clients are independent


def test_limiter_memory_is_bounded():
    limiter = SlidingWindowLimiter(limit=1, max_keys=3)
    for i in range(10):
        limiter.check(f"client-{i}")
    assert len(limiter._hits) == 3


def test_forwarding_headers_are_trusted_only_from_configured_proxies():
    assert client_id("1.2.3.4", "9.9.9.9", []) == "1.2.3.4"  # untrusted peer: header ignored
    assert client_id("10.0.0.1", "9.9.9.9", ["10.0.0.1"]) == "9.9.9.9"
    assert (
        client_id("10.0.0.1", "spoofed, 9.9.9.9", ["10.0.0.1"]) == "9.9.9.9"
    )  # proxy-appended entry
    assert client_id(None, None, []) == "unknown"


def test_search_returns_429_with_retry_after_when_the_limit_is_exceeded(make_client):
    client = make_client(search_rate_limit_per_minute=3)
    codes = [client.post("/api/v1/search", json=SEARCH).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = client.post("/api/v1/search", json=SEARCH)
    assert int(r.headers["retry-after"]) >= 1
    error = r.json()["error"]
    assert error["code"] == "RATE_LIMITED" and error["request_id"] == r.headers["x-request-id"]
    assert client.get("/healthz").status_code == 200  # probes are not rate limited


def test_clients_behind_an_untrusted_proxy_share_a_bucket_but_trusted_ones_do_not(make_client):
    spoof = {"x-forwarded-for": "1.1.1.1"}
    untrusted = make_client(search_rate_limit_per_minute=1)
    assert untrusted.post("/api/v1/search", json=SEARCH, headers=spoof).status_code == 200
    spoof2 = {"x-forwarded-for": "2.2.2.2"}  # cannot dodge the limit by spoofing the header
    assert untrusted.post("/api/v1/search", json=SEARCH, headers=spoof2).status_code == 429

    trusted = make_client(search_rate_limit_per_minute=1, trusted_proxy_ips=["testclient"])
    assert trusted.post("/api/v1/search", json=SEARCH, headers=spoof).status_code == 200
    assert trusted.post("/api/v1/search", json=SEARCH, headers=spoof2).status_code == 200
    assert trusted.post("/api/v1/search", json=SEARCH, headers=spoof).status_code == 429


def test_inference_guard_never_exceeds_its_concurrency_limit():
    guard = InferenceGuard(max_concurrent=2, wait_timeout=5)
    active, peak, lock = [0], [0], threading.Lock()

    def work():
        with guard.slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 2


def test_inference_guard_rejects_instead_of_queueing_forever():
    guard = InferenceGuard(max_concurrent=1, wait_timeout=0.05)
    with guard.slot():
        with pytest.raises(ApiError) as exc, guard.slot():
            pass
    assert exc.value.status == 503 and exc.value.code == "BUSY"
    with guard.slot():  # the slot is released afterwards
        pass


def test_a_slow_encoder_cannot_pile_up_unbounded_work(make_client, api_bundle):
    from fastapi.testclient import TestClient

    client = make_client(inference_max_concurrency=1, inference_wait_timeout_seconds=0.05)
    real_search = api_bundle.search

    def slow_search(*args, **kwargs):
        time.sleep(0.4)
        return real_search(*args, **kwargs)

    api_bundle.search = slow_search
    results = []

    def call(query):
        with TestClient(client.app) as c:
            results.append(c.post("/api/v1/search", json={**SEARCH, "query": query}))

    threads = [threading.Thread(target=call, args=(f"volcano {i}",)) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    codes = sorted(r.status_code for r in results)
    assert codes.count(200) >= 1 and 503 in codes  # extra requests are shed, not queued
    busy = next(r for r in results if r.status_code == 503)
    assert busy.json()["error"]["code"] == "BUSY" and busy.headers["retry-after"] == "1"
