"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory   (from the backend/ directory)
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from app.api import routes, service, suggest
from app.api.cache import QueryCache
from app.api.errors import install_error_handlers
from app.api.guard import InferenceGuard
from app.api.logging_config import configure_logging, log_event
from app.api.middleware import RequestContextMiddleware
from app.api.ratelimit import SlidingWindowLimiter
from app.api.state import AppState
from app.config import Settings, get_settings
from app.db.base import make_engine, make_session_factory
from app.search.artifacts import ArtifactError
from app.search.bundle import SearchBundle


def create_app(
    settings: Settings | None = None,
    *,
    bundle: SearchBundle | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    """Build the app. ``bundle``/``session_factory`` can be injected (tests); otherwise they
    are created from settings during startup."""
    settings = settings or get_settings()
    configure_logging()

    secret = (settings.cursor_secret or secrets.token_hex(32)).encode()
    state = AppState(
        settings=settings,
        session_factory=session_factory or make_session_factory(make_engine(settings.database_url)),
        bundle=bundle,
        cache=QueryCache(settings.cache_max_entries, settings.cache_ttl_seconds),
        guard=InferenceGuard(
            settings.inference_max_concurrency, settings.inference_wait_timeout_seconds
        ),
        search_limiter=SlidingWindowLimiter(settings.search_rate_limit_per_minute),
        suggest_limiter=SlidingWindowLimiter(settings.suggest_rate_limit_per_minute),
        write_limiter=SlidingWindowLimiter(settings.write_rate_limit_per_minute),
        admin_limiter=SlidingWindowLimiter(settings.admin_rate_limit_per_minute),
        cursor_secret=secret,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state.bundle is None and settings.active_index_path:
            try:
                state.bundle = SearchBundle.load(settings.active_index_path)
            except (ArtifactError, OSError):
                log_event("index_load_failed", level=40, error_code="ARTIFACTS_INVALID")
        if state.bundle is not None:
            try:
                with state.session_factory() as session:
                    state.index_version_id = service.register_index_version(
                        session, state.bundle
                    ).id
                    session.commit()
            except Exception:  # database not reachable yet: /readyz reports it, search retries
                log_event("index_registration_failed", level=40, error_code="DATABASE_UNAVAILABLE")
            try:
                suggest.get_suggester(state)  # build the autocomplete trie for this bundle
            except Exception:  # retried lazily on the first suggestion request
                log_event("suggester_build_failed", level=40, error_code="DATABASE_UNAVAILABLE")
        yield

    app = FastAPI(title="PolySearch", version="0.1.0", lifespan=lifespan)
    app.state.polysearch = state
    install_error_handlers(app)
    app.include_router(routes.probes)
    app.include_router(routes.v1)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_body_bytes)
    app.add_middleware(  # added last = outermost, so even 413/429 responses carry CORS headers
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    return app
