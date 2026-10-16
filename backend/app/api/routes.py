"""HTTP routes: public search API under /api/v1 plus liveness/readiness probes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from app.api import analytics, service, signals, suggest
from app.api.admin import require_admin
from app.api.errors import ApiError, request_id_of
from app.api.logging_config import log_event
from app.api.ratelimit import client_id
from app.api.readiness import run_readiness
from app.api.schemas import (
    ClickRequest,
    ClickResponse,
    DocumentResponse,
    FeedbackRequest,
    FeedbackResponse,
    Language,
    ResultPage,
    SearchRequest,
    SearchResponse,
    SuggestionItem,
    SuggestionResponse,
)
from app.api.state import AppState
from app.config import SUPPORTED_LANGUAGES
from app.db.models import ContentSource, Document

v1 = APIRouter(prefix="/api/v1")
probes = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.polysearch


def _enforce_rate_limit(request: Request, group: str = "search") -> None:
    state = _state(request)
    limiter = {
        "search": state.search_limiter,
        "suggest": state.suggest_limiter,
        "write": state.write_limiter,
    }[group]
    peer = request.client.host if request.client else None
    who = client_id(peer, request.headers.get("x-forwarded-for"), state.settings.trusted_proxy_ips)
    allowed, retry_after = limiter.check(who)
    if not allowed:
        state.metrics["rate_limited"][group] += 1
        log_event(
            "rate_limited", level=30, request_id=request_id_of(request),
            route=request.scope["route"].path, error_code="RATE_LIMITED", retry_after=retry_after,
        )  # fmt: skip
        raise ApiError(
            429,
            "RATE_LIMITED",
            "Too many requests. Please slow down.",
            details={"retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )


@v1.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    _enforce_rate_limit(request)
    return service.run_search(_state(request), body, request_id_of(request))


@v1.get("/searches/{search_id}/results", response_model=ResultPage)
def search_results(
    search_id: str,
    request: Request,
    cursor: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=10, ge=5, le=20),
) -> ResultPage:
    _enforce_rate_limit(request)
    return service.get_page(_state(request), search_id, cursor, limit)


@v1.get("/documents/{document_id}", response_model=DocumentResponse)
def document(document_id: str, request: Request) -> DocumentResponse:
    _enforce_rate_limit(request)
    return service.get_document(_state(request), document_id)


@v1.get("/suggestions", response_model=SuggestionResponse)
def suggestions(
    request: Request,
    prefix: str = Query(min_length=2, max_length=100),
    language: Language | None = None,
    limit: int = Query(default=8, ge=1, le=20),
) -> SuggestionResponse:
    """Title autocomplete from the in-memory trie of the active index."""
    _enforce_rate_limit(request, "suggest")
    state = _state(request)
    suggester = suggest.get_suggester(state)
    if suggester is None:
        raise ApiError(503, "INDEX_UNAVAILABLE", "No search index is currently available.")
    entries = suggester.suggest(prefix, language, min(limit, state.settings.max_suggestions))
    return SuggestionResponse(
        prefix=prefix.strip(),
        suggestions=[
            SuggestionItem(text=e.display, languages=sorted(e.languages)) for e in entries
        ],
    )


@v1.post("/searches/{search_id}/clicks", response_model=ClickResponse)
def click(
    search_id: str, body: ClickRequest, request: Request, response: Response
) -> ClickResponse:
    _enforce_rate_limit(request, "write")
    result, created = signals.record_click(
        _state(request), search_id, body.document_id, body.idempotency_key
    )
    response.status_code = 201 if created else 200
    return result


@v1.put("/searches/{search_id}/feedback/{document_id}", response_model=FeedbackResponse)
def feedback(
    search_id: str, document_id: uuid.UUID, body: FeedbackRequest, request: Request
) -> FeedbackResponse:
    _enforce_rate_limit(request, "write")
    return signals.set_feedback(_state(request), search_id, document_id, body.label)


@v1.get("/admin/analytics")
def admin_analytics(request: Request, days: int = Query(default=7, ge=1, le=90)) -> dict:
    require_admin(request)
    state = _state(request)
    with state.session_factory() as session:
        return analytics.build_report(
            session,
            days,
            state.metrics["rate_limited"],
            state.settings.analytics_store_query_text,
            state.settings.popular_query_min_count,
        )


@v1.get("/stats")
def stats(request: Request) -> dict:
    """Public corpus and index summary (no paths, hashes or configuration)."""
    state = _state(request)
    with state.session_factory() as session:
        counts = dict(
            session.execute(
                select(Document.language, func.count())
                .where(Document.is_active.is_(True))
                .group_by(Document.language)
            ).all()
        )
        sources = [
            {"name": s.name, "licence_name": s.licence_name, "licence_url": s.licence_url}
            for s in session.scalars(select(ContentSource).where(ContentSource.enabled.is_(True)))
        ]
    bundle = state.bundle
    return {
        "languages": list(SUPPORTED_LANGUAGES),
        "documents_per_language": {lang: counts.get(lang, 0) for lang in SUPPORTED_LANGUAGES},
        "sources": sources,
        "index_version": bundle.version if bundle else None,
        "encoder": bundle.manifest["encoder"]["name"] if bundle else None,
        "default_mode": service.default_mode(bundle) if bundle else None,
        "modes": ["dense", "lexical", "hybrid"],
    }


@probes.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only: the process is up. No database, model or index checks."""
    return {"status": "ok"}


@probes.get("/readyz")
def readyz(request: Request):
    outcome = run_readiness(_state(request))
    return JSONResponse(outcome, status_code=200 if outcome["ready"] else 503)
