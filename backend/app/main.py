"""FastAPI application factory. Phase 0 only exposes a liveness probe."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="PolySearch", version="0.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        # Liveness only: no database, model or index checks (those belong in /readyz later).
        return {"status": "ok"}

    return app


app = create_app()
