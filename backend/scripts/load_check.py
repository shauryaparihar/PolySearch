"""A small, repeatable load check for a running PolySearch API (httpx + asyncio only).

    python backend/scripts/load_check.py --requests 60 --concurrency 20 --output load.json
    python backend/scripts/load_check.py --mode rate-limit --burst-requests 60

This is a bounded engineering check, not a capacity test and not a replacement for a proper
load-testing tool. It sends only the built-in, non-sensitive seed queries (common encyclopedia
topics) and never anything read from a file or from the server.

Phases (``--mode load`` runs the first two, ``--mode rate-limit`` the third, ``all`` runs all):

1. ``uncached``   every request is a distinct (query, mode, target languages) combination, so
                  each one misses the server-side result cache and exercises the encoder.
2. ``warm_cache`` the same requests again; they should now be answered from the cache.
3. ``rate_limit`` a rapid burst from this one client. With the server's default limit
                  (30 searches/minute/client) the extra requests must be answered with HTTP 429
                  and a Retry-After header - never with 5xx errors or timeouts.

The per-client limiter also throttles phases 1-2, so run them against an API started with a
high ``SEARCH_RATE_LIMIT_PER_MINUTE`` and run ``--mode rate-limit`` against the default limit.
``all`` is for a limit between the two; the report states what was observed, never assumed.

Latency is client-side wall time per request (connection reuse, JSON parsing excluded, queueing
behind ``--concurrency`` included). Percentiles use the nearest-rank method. Numbers are only
meaningful next to the machine, container and server settings recorded in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import httpx

MAX_CONCURRENCY = 50
MAX_REQUESTS = 2000
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Common encyclopedia topics in each language: public, non-sensitive, present in Wikipedia.
SEED_QUERIES: dict[str, list[str]] = {
    "en": ["mathematics", "physics", "chemistry", "biology", "music", "football", "paris",
           "medicine", "astronomy", "philosophy"],
    "es": ["matemáticas", "física", "química", "biología", "música", "fútbol", "parís",
           "medicina", "astronomía", "filosofía"],
    "fr": ["mathématiques", "physique", "chimie", "biologie", "musique", "football", "paris",
           "médecine", "astronomie", "philosophie"],
}  # fmt: skip
MODES = ("dense", "hybrid", "lexical")
LANGUAGES = ("en", "es", "fr")


@dataclass(frozen=True)
class Sample:
    """One finished request. ``status`` is None when no HTTP response arrived."""

    status: int | None
    seconds: float
    cache_hit: bool | None = None
    retry_after: int | None = None
    error: str | None = None


# ------------------------------------------------------------------ pure helpers
def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (the smallest value covering ``pct`` percent of the data)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 3)


def summarise(samples: list[Sample], concurrency: int, elapsed: float | None = None) -> dict:
    """Aggregate one phase. Latency statistics cover every request that got an HTTP response.

    ``elapsed`` is the wall-clock time of the whole phase; throughput is requests / elapsed."""
    answered = [s for s in samples if s.status is not None]
    seconds = [s.seconds for s in answered]
    statuses = Counter(str(s.status) for s in answered)
    errors = Counter(s.error for s in samples if s.error)
    cache = [s.cache_hit for s in answered if s.status == 200 and s.cache_hit is not None]
    retry = [s.retry_after for s in samples if s.retry_after is not None]
    server_errors = sum(n for code, n in statuses.items() if code.startswith("5"))
    return {
        "requests": len(samples),
        "concurrency": concurrency,
        "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
        "throughput_rps": round(len(samples) / elapsed, 2) if elapsed else None,
        "status_counts": dict(sorted(statuses.items())),
        "transport_errors": dict(sorted(errors.items())),
        "server_error_count": server_errors,
        "client_error_count": sum(
            n for code, n in statuses.items() if code.startswith("4") and code != "429"
        ),
        "median_ms": _ms(median(seconds)),
        "p95_ms": _ms(percentile(seconds, 95)),
        "max_ms": _ms(max(seconds)) if seconds else None,
        "cache_hits": sum(1 for hit in cache if hit),
        "cache_misses": sum(1 for hit in cache if not hit),
        "rate_limited": statuses.get("429", 0),
        "retry_after_seconds_max": max(retry) if retry else None,
    }


def request_set(count: int) -> list[dict]:
    """``count`` distinct, deterministic search bodies (distinct = distinct cache keys)."""
    combos = []
    for (language, queries), mode in itertools.product(SEED_QUERIES.items(), MODES):
        others = [t for t in LANGUAGES if t != language]
        for query in queries:
            for target in (others, others[:1], [language]):
                combos.append(
                    {"query": query, "query_language": language,
                     "target_languages": target, "mode": mode}  # fmt: skip
                )
    if count > len(combos):
        raise ValueError(f"only {len(combos)} distinct requests exist; asked for {count}")
    # A hash order is deterministic but mixes languages and modes, so short runs stay varied.
    combos.sort(key=lambda b: hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest())
    return combos[:count]


# ------------------------------------------------------------------ network phases
async def _one(client: httpx.AsyncClient, body: dict) -> Sample:
    start = time.perf_counter()
    try:
        response = await client.post("/api/v1/search", json=body)
    except httpx.HTTPError as exc:
        return Sample(None, time.perf_counter() - start, error=type(exc).__name__)
    elapsed = time.perf_counter() - start
    cache_hit = None
    retry_after = None
    if response.status_code == 200:
        try:
            cache_hit = bool(response.json().get("cache_hit"))
        except ValueError:
            return Sample(response.status_code, elapsed, error="InvalidJson")
    elif response.status_code == 429:
        header = response.headers.get("retry-after", "")
        retry_after = int(header) if header.isdigit() else None
    return Sample(response.status_code, elapsed, cache_hit, retry_after)


async def run_phase(
    client: httpx.AsyncClient, bodies: list[dict], concurrency: int
) -> tuple[list[Sample], float]:
    """Run every request (at most ``concurrency`` at once); return samples and elapsed seconds."""
    started = time.perf_counter()
    gate = asyncio.Semaphore(concurrency)

    async def bounded(body: dict) -> Sample:
        async with gate:
            return await _one(client, body)

    samples = list(await asyncio.gather(*(bounded(b) for b in bodies)))
    return samples, time.perf_counter() - started


async def _server_view(client: httpx.AsyncClient) -> dict:
    """What the server says about itself (no secrets): readiness, index version, mode."""
    view: dict = {}
    try:
        ready = await client.get("/readyz")
        view["readyz_status"] = ready.status_code
        stats = await client.get("/api/v1/stats")
        if stats.status_code == 200:
            data = stats.json()
            view["index_version"] = data.get("index_version")
            view["document_count"] = data.get("document_count")
    except httpx.HTTPError as exc:
        view["unreachable"] = type(exc).__name__
    return view


async def run_check(
    base_url: str,
    requests: int,
    concurrency: int,
    mode: str,
    burst_requests: int,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    if not 1 <= requests <= MAX_REQUESTS:
        raise ValueError(f"requests must be between 1 and {MAX_REQUESTS}")
    if not 1 <= burst_requests <= MAX_REQUESTS:
        raise ValueError(f"burst-requests must be between 1 and {MAX_REQUESTS}")
    started = datetime.now(UTC)
    clock = time.perf_counter()
    report: dict = {"phases": {}}
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        transport=transport,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        report["server_before"] = await _server_view(client)
        if mode in ("load", "all"):
            bodies = request_set(requests)
            for name in ("uncached", "warm_cache"):  # the second pass repeats the first
                samples, elapsed = await run_phase(client, bodies, concurrency)
                report["phases"][name] = summarise(samples, concurrency, elapsed)
        if mode in ("rate-limit", "all"):
            pool = request_set(min(burst_requests, 100))
            burst = [pool[i % len(pool)] for i in range(burst_requests)]
            burst_workers = min(concurrency, 10)
            samples, elapsed = await run_phase(client, burst, burst_workers)
            phase = summarise(samples, burst_workers, elapsed)
            phase["observed_429"] = phase["rate_limited"] > 0
            phase["bounded"] = (
                phase["server_error_count"] == 0
                and not phase["transport_errors"]
                and set(phase["status_counts"]) <= {"200", "429"}
            )
            report["phases"]["rate_limit"] = phase
        report["server_after"] = await _server_view(client)
    report["context"] = {
        "tool": "backend/scripts/load_check.py",
        "base_url": base_url,
        "mode": mode,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - clock, 3),
        "client_machine": (
            f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
        ),
        "latency_definition": "client-side wall time per request, queueing behind concurrency "
        "included; nearest-rank percentiles",
        "limits": {"max_concurrency": MAX_CONCURRENCY, "max_requests": MAX_REQUESTS},
    }  # fmt: skip
    report["healthy"] = is_healthy(report)
    return report


def is_healthy(report: dict) -> bool:
    """No 5xx, no transport errors and no unexpected 4xx in any phase, and the server answered."""
    if "unreachable" in report.get("server_before", {}):
        return False
    for name, phase in report["phases"].items():
        if phase["server_error_count"] or phase["transport_errors"]:
            return False
        if name != "rate_limit" and (phase["client_error_count"] or phase["rate_limited"]):
            return False  # load phases must not be throttled or rejected to count as healthy
    return True


def console_summary(report: dict) -> str:
    lines = [
        f"PolySearch load check: {report['context']['base_url']} "
        f"({report['context']['client_machine']})",
        f"server index: {report['server_before'].get('index_version')}, "
        f"readyz: {report['server_before'].get('readyz_status')}",
    ]
    for name, p in report["phases"].items():
        lines.append(
            f"- {name}: {p['requests']} requests @ concurrency {p['concurrency']} "
            f"({p['throughput_rps']} req/s): median {p['median_ms']} ms, p95 {p['p95_ms']} ms, "
            f"status {p['status_counts']}, "
            f"cache hit/miss {p['cache_hits']}/{p['cache_misses']}, 429s {p['rate_limited']}"
            + (f", transport errors {p['transport_errors']}" if p["transport_errors"] else "")
        )
    if "rate_limit" in report["phases"]:
        r = report["phases"]["rate_limit"]
        lines.append(
            f"rate limit: observed={r['observed_429']} bounded={r['bounded']} "
            f"retry-after max={r['retry_after_seconds_max']}"
        )
    lines.append(f"healthy: {report['healthy']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests", type=int, default=60, help="requests per load phase")
    parser.add_argument(
        "--concurrency", type=int, default=10, help=f"parallel requests, at most {MAX_CONCURRENCY}"
    )
    parser.add_argument("--mode", choices=["load", "rate-limit", "all"], default="load")
    parser.add_argument("--burst-requests", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", help="write the JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    try:
        report = asyncio.run(
            run_check(
                args.base_url, args.requests, args.concurrency, args.mode,
                args.burst_requests, args.timeout,
            )  # fmt: skip
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=asdict)
            handle.write("\n")
    print(console_summary(report))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
