"""Deterministic tests for the load-check report: no real timing, no network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from scripts import load_check as lc
from scripts.load_check import Sample, percentile, summarise


def test_percentile_uses_nearest_rank():
    values = [float(n) for n in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 100) == 100.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) is None
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0  # input order does not matter
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


def test_median_of_even_and_odd_counts():
    assert lc.median([3.0, 1.0, 2.0]) == 2.0
    assert lc.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert lc.median([]) is None


def test_summary_aggregates_status_codes_latency_and_cache():
    samples = [
        Sample(200, 0.010, cache_hit=False),
        Sample(200, 0.020, cache_hit=False),
        Sample(200, 0.030, cache_hit=True),
        Sample(429, 0.001, retry_after=7),
        Sample(429, 0.001, retry_after=12),
        Sample(503, 0.040),
        Sample(None, 5.0, error="ReadTimeout"),
    ]
    s = summarise(samples, concurrency=4)
    assert s["requests"] == 7 and s["concurrency"] == 4
    assert s["status_counts"] == {"200": 3, "429": 2, "503": 1}
    assert s["transport_errors"] == {"ReadTimeout": 1}
    assert s["server_error_count"] == 1 and s["rate_limited"] == 2
    assert (s["cache_hits"], s["cache_misses"]) == (1, 2)
    assert s["retry_after_seconds_max"] == 12
    # the timed-out request never got a response, so it is excluded from latency
    assert s["median_ms"] == 15.0 and s["max_ms"] == 40.0  # median of 1, 1, 10, 20, 30, 40 ms
    assert s["p95_ms"] == 40.0


def test_throughput_is_requests_over_elapsed_time():
    samples = [Sample(200, 0.01, cache_hit=False)] * 7
    s = summarise(samples, concurrency=4, elapsed=2.0)
    assert s["elapsed_seconds"] == 2.0 and s["throughput_rps"] == 3.5
    assert summarise(samples, concurrency=4)["throughput_rps"] is None  # unknown, not guessed


def test_summary_of_nothing_is_empty_not_a_crash():
    s = summarise([], concurrency=1)
    assert s["requests"] == 0 and s["median_ms"] is None and s["p95_ms"] is None


def test_the_request_set_is_deterministic_distinct_and_non_sensitive():
    a, b = lc.request_set(40), lc.request_set(40)
    assert a == b and len({json.dumps(x, sort_keys=True) for x in a}) == 40
    known = {q for qs in lc.SEED_QUERIES.values() for q in qs}
    assert {x["query"] for x in a} <= known
    assert {x["mode"] for x in lc.request_set(200)} == set(lc.MODES)
    assert all(x["query_language"] not in x["target_languages"] or x["target_languages"] == [
        x["query_language"]] for x in a)
    with pytest.raises(ValueError):
        lc.request_set(10_000)


def test_every_request_body_is_accepted_by_the_real_api_schema():
    from app.api.schemas import SearchRequest

    for body in lc.request_set(200):
        SearchRequest(**body)


def _fake_api(limit: int | None = None, fail_after: int | None = None):
    """An in-process API: caches by body, optionally rate limits or fails."""
    seen: set[str] = set()
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"ready": True})
        if request.url.path == "/api/v1/stats":
            return httpx.Response(200, json={"index_version": "idx-test", "document_count": 5})
        state["calls"] += 1
        if fail_after is not None and state["calls"] > fail_after:
            return httpx.Response(500, json={"error": {"code": "INTERNAL"}})
        if limit is not None and state["calls"] > limit:
            return httpx.Response(429, headers={"Retry-After": "9"}, json={})
        key = request.content.decode()
        hit = key in seen
        seen.add(key)
        return httpx.Response(200, json={"cache_hit": hit, "results": []})

    return httpx.MockTransport(handler)


def _run(**overrides):
    args = dict(base_url="http://test", requests=12, concurrency=4, mode="load",
                burst_requests=30, timeout=5.0, transport=_fake_api())  # fmt: skip
    args.update(overrides)
    return asyncio.run(lc.run_check(**args))


def test_load_run_reports_uncached_then_warm_phases():
    report = _run()
    uncached, warm = report["phases"]["uncached"], report["phases"]["warm_cache"]
    assert (uncached["cache_hits"], uncached["cache_misses"]) == (0, 12)
    assert (warm["cache_hits"], warm["cache_misses"]) == (12, 0)
    assert uncached["status_counts"] == {"200": 12} and report["healthy"] is True
    assert uncached["elapsed_seconds"] > 0 and uncached["throughput_rps"] > 0
    assert report["server_before"]["index_version"] == "idx-test"
    assert report["context"]["limits"]["max_concurrency"] == 50
    json.dumps(report)  # machine-readable


def test_rate_limit_burst_is_bounded_when_the_server_answers_429():
    report = _run(mode="rate-limit", transport=_fake_api(limit=10))
    burst = report["phases"]["rate_limit"]
    assert burst["observed_429"] is True and burst["bounded"] is True
    assert burst["status_counts"] == {"200": 10, "429": 20}
    assert burst["retry_after_seconds_max"] == 9 and report["healthy"] is True


def test_a_burst_that_never_hits_the_limit_is_reported_as_not_observed():
    burst = _run(mode="rate-limit")["phases"]["rate_limit"]
    assert burst["observed_429"] is False and burst["bounded"] is True


def test_server_errors_make_the_run_unhealthy_and_the_burst_unbounded():
    report = _run(mode="all", transport=_fake_api(fail_after=15))
    assert report["healthy"] is False
    assert report["phases"]["rate_limit"]["bounded"] is False


def test_throttled_load_phases_do_not_count_as_healthy():
    report = _run(transport=_fake_api(limit=5))
    assert report["phases"]["uncached"]["rate_limited"] > 0 and report["healthy"] is False


def test_unreachable_server_is_reported_not_raised():
    def refuse(request):
        raise httpx.ConnectError("refused")

    report = _run(transport=httpx.MockTransport(refuse))
    assert report["server_before"] == {"unreachable": "ConnectError"}
    assert report["healthy"] is False
    assert report["phases"]["uncached"]["transport_errors"] == {"ConnectError": 12}


@pytest.mark.parametrize(
    "overrides",
    [{"concurrency": 51}, {"concurrency": 0}, {"requests": 0}, {"requests": 5000},
     {"burst_requests": 0}],  # fmt: skip
)
def test_limits_are_enforced(overrides):
    with pytest.raises(ValueError):
        _run(**overrides)


def test_cli_rejects_concurrency_above_the_cap(capsys):
    with pytest.raises(SystemExit) as exit_info:
        lc.main(["--concurrency", "51"])
    assert exit_info.value.code == 2
    assert "between 1 and 50" in capsys.readouterr().err


def test_console_summary_states_conditions_and_results():
    text = lc.console_summary(_run(mode="all", transport=_fake_api(limit=40)))
    for expected in ("uncached", "warm_cache", "rate_limit", "concurrency 4", "median", "p95"):
        assert expected in text
