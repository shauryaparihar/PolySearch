"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory   (from the backend/ directory)
"""

from __future__ import annotations

import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
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
from app.db.models import IndexVersion
from app.jobs.manager import recover_stale_jobs
from app.jobs.worker import JobWorker
from app.search.artifacts import ArtifactError
from app.search.bundle import SearchBundle

VERSION_NAME = re.compile(r"idx-[0-9a-f]{12}")


def _database_active_index(settings: Settings, factory) -> Path | None:
    """Directory of the version the database marks active, if its files are still on disk.

    A reindex activates a new version in the database; the environment variable keeps naming
    the version the deployment started with. Restarting must not silently undo the activation.
    """
    configured = Path(settings.active_index_path) if settings.active_index_path else None
    root = Path(settings.artifact_root) if settings.artifact_root else (
        configured.parent if configured else None
    )
    if root is None:
        return None
    try:
        with factory() as session:
            version = session.scalar(
                select(IndexVersion.version).where(IndexVersion.status == "active")
            )
    except Exception:  # database not reachable yet: fall back to the configured path
        return None
    if not version or not VERSION_NAME.fullmatch(version):  # never build a path from odd text
        return None
    candidate = root / version
    return candidate if (candidate / "manifest.json").is_file() else None


def _load_startup_bundle(settings: Settings, factory) -> SearchBundle | None:
    candidates = [_database_active_index(settings, factory), settings.active_index_path]
    tried: list[Path] = []
    for path in candidates:
        if path is None or Path(path) in tried:
            continue
        tried.append(Path(path))
        try:
            return SearchBundle.load(path)
        except (ArtifactError, OSError):
            log_event("index_load_failed", level=40, error_code="ARTIFACTS_INVALID")
    return None


def create_app(
    settings: Settings | None = None,
    *,
    bundle: SearchBundle | None = None,
    session_factory: sessionmaker[Session] | None = None,
    synchronous_jobs: bool = False,
    source_factory=None,
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
        admin_failure_limiter=SlidingWindowLimiter(settings.admin_failed_attempts_per_minute),
        cursor_secret=secret,
        source_factory=source_factory,
    )
    state.worker = JobWorker(state, synchronous=synchronous_jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state.bundle is None:
            state.bundle = _load_startup_bundle(settings, state.session_factory)
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
        try:
            recovered = recover_stale_jobs(state.session_factory)
            if recovered:
                log_event("stale_jobs_recovered", level=30, error_code="RESTART_INTERRUPTED")
        except Exception:  # database unavailable: nothing to recover yet
            log_event("stale_job_recovery_failed", level=40, error_code="DATABASE_UNAVAILABLE")
        state.worker.start()
        try:
            yield
        finally:
            state.worker.stop(settings.job_shutdown_timeout_seconds)

    app = FastAPI(title="PolySearch", version="0.1.0", lifespan=lifespan)
    app.state.polysearch = state
    install_error_handlers(app)
    app.include_router(routes.probes)
    app.include_router(routes.v1)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_body_bytes)
    app.add_middleware(  # added last = outermost, so even 413/429 responses carry CORS headers
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST", "PUT"],  # PUT: relevance feedback
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    return app
