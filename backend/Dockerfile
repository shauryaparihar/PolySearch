# PolySearch API image. Build context is the repository root:
#   docker build -f backend/Dockerfile -t polysearch-api .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

# Exact dependency versions (backend/constraints.txt); CPU-only PyTorch wheels.
COPY backend/requirements.txt backend/constraints.txt /tmp/deps/
RUN pip install -r /tmp/deps/requirements.txt -c /tmp/deps/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /tmp/deps

# Unprivileged user. The directories below are created here so that named volumes mounted on
# them inherit this ownership on first use.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts/index /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

WORKDIR /app
COPY --chown=app:app backend/ /app/backend/
COPY --chown=app:app benchmark/v1/ /app/benchmark/v1/
RUN rm -rf /app/backend/tests

USER app
WORKDIR /app/backend
ENV BENCHMARK_DIR=/app/benchmark/v1 \
    ARTIFACT_ROOT=/app/artifacts/index

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# One process: the result cache, rate limiter and index-job worker live in memory (see docs).
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
