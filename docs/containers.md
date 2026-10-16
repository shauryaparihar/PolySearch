# Containers, volumes and the CI commands

One Compose file (`docker-compose.yml`) runs everything locally:

| Service | Image | Purpose | Host port |
|---|---|---|---|
| `postgres` | `postgres:16` | data (documents, searches, clicks, jobs) | 127.0.0.1:5433 |
| `migrate` | built from `backend/Dockerfile` | one-shot `alembic upgrade head`, then exits | - |
| `api` | built from `backend/Dockerfile` | FastAPI on one uvicorn worker, non-root (uid 10001) | 127.0.0.1:8000 |
| `frontend` | built from `frontend/Dockerfile` | static React build on unprivileged nginx (uid 101) | 127.0.0.1:8080 |

Named volumes: `pgdata` (database), `artifacts` (index versions under `/app/artifacts/index`),
`hf_cache` (the downloaded model, so restarts do not download it again). `docker compose down`
keeps them; `docker compose down -v` deletes them. All published ports are bound to 127.0.0.1.

`docker compose up -d postgres` starts only the database (what the test suite needs).
All values have throwaway local defaults; override them in a gitignored `.env` (see
`.env.example`). The web UI is built with `VITE_API_URL=http://localhost:${API_HOST_PORT}` and the
API allows exactly that UI origin through CORS.

## First run of a new environment

```bash
docker compose up -d --build          # builds both images, migrates, starts API + UI
# 1. seed documents (the original corpus; mount it read-only, nothing is copied into the image)
docker compose run --rm --no-deps -v "$PWD/data/raw:/data/raw:ro" api \
    python -m app.ingestion.cli import-legacy --raw-dir /data/raw
# 2. build the first index FROM THAT DATABASE (downloads the pinned model, ~470 MB, once)
docker compose run --rm api python scripts/build_artifacts.py --from-database \
    --corpus /app/artifacts/corpus.jsonl --out /app/artifacts/index
# 3. start serving it (the path is printed by step 2; needed only for this first start)
ACTIVE_INDEX_PATH=/app/artifacts/index/idx-<version> docker compose up -d api
curl localhost:8000/readyz            # 200 once all seven checks pass
```

An index must be built from the database it serves: `/readyz` reports `document_mappings: false`
if the index contains documents the database does not have. After this first start, restarts
(`docker compose restart`, `down`/`up`) serve whichever version PostgreSQL marks active, so a
reindexed version survives a restart. In the verified run the first build of 5,585 documents took
108 s on CPU inside the container. The API image is about 2 GB because of PyTorch.

Without an index the stack still starts: `/healthz` is 200, `/readyz` is 503 with the failing
checks listed, and search answers a safe `503 INDEX_UNAVAILABLE`.

## The same commands as CI

```bash
# backend job (Python 3.11; PostgreSQL on 5433 with a *_test database)
pip install -r backend/requirements-dev.txt -c backend/constraints.txt
ruff check .
(cd backend && DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head)   # on an empty database
REQUIRE_DB=1 pytest
# frontend job
(cd frontend && npm ci && npm test && npm run build)
# containers job: builds both images, boots the stack from empty volumes, checks probes
scripts/stack_smoke.sh
```

## CI versus deployment testing

CI proves the code, the migrations and the container wiring on a clean machine. It never
downloads the transformer (tests use a deterministic fake encoder), so it says nothing about
real search quality, real latency or a hosted environment; those are measured separately (the
benchmark and `docs/load_check.md`) and only after a deployment exists could they be measured
there.

## Scope notes

* One process holds the result cache, the rate limiter and the index-job worker; running
  several API replicas would need a shared cache/limiter and a job queue.
* The nginx image sets basic security headers but no Content-Security-Policy yet.
* Dependencies are locked: `backend/constraints.txt` (Python) and `frontend/package-lock.json`.
