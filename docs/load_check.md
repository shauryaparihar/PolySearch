# Load check

A small, repeatable check of a running API, **not** a capacity test. The script is
`backend/scripts/load_check.py` (httpx + asyncio, no extra service). The raw reports and the full
test conditions are in [`load_check_results.json`](load_check_results.json).

## Run it

```bash
# 1. load phases: distinct queries (cache misses), then the same queries again (cache hits).
#    Start the API with a high per-client limit, otherwise the limiter throttles the run:
SEARCH_RATE_LIMIT_PER_MINUTE=10000 docker compose up -d api
python backend/scripts/load_check.py --requests 100 --concurrency 20 --output load.json

# 2. the deliberate rate-limit burst, against the API's default limit (30 searches/minute/client):
docker compose up -d api
python backend/scripts/load_check.py --mode rate-limit --burst-requests 60 --concurrency 10
```

* `--concurrency` is capped at 50 and `--requests` at 2000; larger values are refused.
* Only built-in, non-sensitive seed queries are sent (common topics in en/es/fr).
* Latency is client-side wall time per request, including queueing behind `--concurrency`.
  Percentiles use the nearest-rank method. The exit code is non-zero if any 5xx, transport error
  or unexpected 4xx occurs in a load phase.

## What was measured (2026-10-16)

Conditions: Apple M4 (10 cores, 16 GiB), Docker Desktop VM with 10 CPUs and ~8.3 GB, the
Compose stack (PostgreSQL 16 and **one** uvicorn worker), MiniLM on CPU, index `idx-b10ebf967ff3`
(5,585 documents, built inside the stack), client on the same machine over localhost, API
restarted before every run so the cache started empty. 100 requests per phase (40 at
concurrency 1).

| Concurrency | Phase | Req/s | Median | p95 | Status codes | Cache hit / miss |
|---:|---|---:|---:|---:|---|---|
| 1 | uncached | 22.2 | 42.6 ms | 63.9 ms | 200 ×40 | 0 / 40 |
| 1 | warm cache | 82.3 | 10.2 ms | 18.1 ms | 200 ×40 | 40 / 0 |
| 20 | uncached | 32.3 | 571 ms | 754 ms | 200 ×100 | 0 / 100 |
| 20 | warm cache | 55.1 | 303 ms | 675 ms | 200 ×100 | 100 / 0 |
| 50 | uncached | 28.7 | 1,518 ms | 2,017 ms | 200 ×100 | 0 / 100 |
| 50 | warm cache | 50.1 | 849 ms | 1,479 ms | 200 ×100 | 100 / 0 |

Deliberate rate-limit burst (60 requests, concurrency 10, default limit 30/minute): 30 × `200`,
30 × `429`, `Retry-After: 60`, no 5xx, no timeouts. The limiter is checked before any encoding,
so rejected requests cost no encoder work.

## How to read it

* **The server stayed responsive:** no errors, timeouts or 5xx at up to 50 concurrent requests.
* **Throughput is flat (~30 req/s uncached, ~50-55 warm) while latency grows with concurrency.**
  That is the shape of a single process doing serialized work (one worker, the Python GIL, an
  encoder guard of 2), not of failures. More concurrent clients wait longer; they do not get
  more throughput.
* **A cache hit saves encoding and ranking but not the database write:** every search still
  stores its run and a pool of up to 100 result rows (this is what makes pagination stable and
  analytics possible), so warm requests are cheaper (10 ms vs 43 ms with one request in flight)
  but not free.
* Median and p95 mean nothing without the concurrency, the phase and the machine beside them.
  A 570 ms median at concurrency 20 is mostly queueing time, not the time one search needs.

## What it does not prove

* Not production capacity: one machine, client and server sharing CPUs, a Docker Desktop VM,
  100 requests per phase, a short run, one synthetic query mix, no network latency.
* Not multi-replica behaviour: the cache, rate limiter and index-job worker live in one
  process's memory. Scaling out needs a shared cache/limiter (for example Redis) and a job queue.
* Not a quality measurement: it does not look at result relevance (see the benchmark for that).
* Not a statement about any hosted deployment; none has been measured.
