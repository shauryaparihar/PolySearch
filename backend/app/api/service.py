"""Search service: cached candidate pools, stored result pages, cursors and attribution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, tuple_, update
from sqlalchemy.orm import Session

from app.api.cache import cache_key, canonical_query, query_sha256
from app.api.errors import ApiError
from app.api.logging_config import log_event
from app.api.schemas import (
    DocumentResponse,
    ResultItem,
    ResultPage,
    SearchRequest,
    SearchResponse,
    SourceAttribution,
)
from app.api.state import AppState
from app.db.models import ContentSource, Document, IndexVersion, SearchResult, SearchRun
from app.ingestion.text import bound_text, clean_text
from app.search.artifacts import ArtifactError

STAGE_FIELDS = ("encode_ms", "lexical_ms", "faiss_ms", "fusion_ms")
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")


def safe_text(text: str, max_chars: int | None = None) -> str:
    """Plain text for API output: markup removed, any leftover tag-start neutralised, bounded."""
    text = _TAG_START.sub("", clean_text(text))
    return bound_text(text, max_chars) if max_chars else text


# ----------------------------------------------------------------------- cursor
def encode_cursor(secret: bytes, search_id: str, next_rank: int) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"s": search_id, "r": next_rank}, separators=(",", ":")).encode()
    ).decode()
    signature = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{signature}"


def decode_cursor(secret: bytes, cursor: str, search_id: str, pool_size: int) -> int:
    """Return the next rank, or raise CURSOR_INVALID. Arbitrary offsets are never trusted."""
    invalid = ApiError(400, "CURSOR_INVALID", "The cursor is invalid.")
    if len(cursor) > 200 or "." not in cursor:
        raise invalid
    payload, signature = cursor.rsplit(".", 1)
    expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(signature, expected):
        raise invalid
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        rank = int(data["r"])
        owner = str(data["s"])
    except (ValueError, KeyError, TypeError):
        raise invalid from None
    if owner != search_id or not 1 <= rank <= pool_size:
        raise invalid
    return rank


# ------------------------------------------------------------------ attribution
def load_attribution(session: Session, names: set[str]) -> dict[str, SourceAttribution]:
    rows = session.scalars(select(ContentSource).where(ContentSource.name.in_(names)))
    return {
        r.name: SourceAttribution(
            name=r.name,
            licence_name=r.licence_name,
            licence_url=r.licence_url,
            attribution_text=r.attribution_text,
        )
        for r in rows
    }


def _item(
    rank, doc: Document, source_name: str, settings, score, dense_rank, lexical_rank
) -> ResultItem:
    return ResultItem(
        rank=rank,
        document_id=str(doc.id),
        title=safe_text(doc.title, 300),
        snippet=safe_text(doc.summary, settings.snippet_max_chars),
        language=doc.language,
        source=source_name,
        source_url=doc.source_url,
        source_revision=doc.source_revision,
        score=float(score),
        dense_rank=dense_rank,
        lexical_rank=lexical_rank,
    )


# ----------------------------------------------------------------------- search
def register_index_version(session: Session, bundle) -> IndexVersion:
    """Make the served bundle the single active index version (retiring any other)."""
    row = session.scalar(select(IndexVersion).where(IndexVersion.version == bundle.version))
    manifest = bundle.manifest
    if row is None:
        evaluation = manifest.get("evaluation") or {}
        default = manifest.get("default_mode") or {}
        row = IndexVersion(
            version=bundle.version,
            encoder_name=manifest["encoder"]["name"],
            encoder_revision=manifest["encoder"]["revision"],
            corpus_sha256=manifest["corpus"]["sha256"],
            manifest_sha256=hashlib.sha256(
                (bundle.path / "manifest.json").read_bytes()
            ).hexdigest(),
            file_hashes=manifest["files"],
            document_count=manifest["corpus"]["documents"],
            per_language_counts=manifest["corpus"]["per_language"],
            vector_dimension=manifest["encoder"]["dimension"],
            default_mode=default.get("default_mode"),
            benchmark_version=evaluation.get("benchmark_version"),
            metrics=evaluation.get("metrics"),
            status="validated",
        )
        session.add(row)
        session.flush()
    if row.status != "active":
        for other in session.scalars(select(IndexVersion).where(IndexVersion.status == "active")):
            other.status = "retired"
        session.flush()  # retire first: the partial unique index allows only one active row
        row.status, row.activated_at = "active", datetime.now(UTC)
        session.flush()
    return row


def default_mode(bundle) -> str:
    choice = (bundle.manifest.get("default_mode") or {}).get("default_mode")
    return choice if choice in ("dense", "lexical", "hybrid") else "dense"


def _require_ready(state: AppState):
    """Return a consistent ``(bundle, index_version_id)`` snapshot or raise 503."""
    bundle, version_id = state.snapshot()
    if bundle is None:
        raise ApiError(503, "INDEX_UNAVAILABLE", "No search index is currently available.")
    if version_id is None:
        with state.session_factory() as session:
            try:
                version_id = register_index_version(session, bundle).id
                session.commit()
            except Exception:
                session.rollback()
                raise ApiError(
                    503, "INDEX_UNAVAILABLE", "No search index is currently available."
                ) from None
        with state.lock:
            if state.bundle is bundle:  # only record it if the bundle was not swapped meanwhile
                state.index_version_id = version_id
    return bundle, version_id


def run_search(state: AppState, req: SearchRequest, request_id: str) -> SearchResponse:
    started = time.perf_counter()
    bundle, version_id = _require_ready(state)
    settings = state.settings
    mode = req.mode or default_mode(bundle)
    pool_size = settings.result_pool_size
    key = cache_key(
        req.query, req.query_language, req.target_languages, req.sources, mode, pool_size,
        bundle.version,
    )  # fmt: skip
    qhash = query_sha256(req.query)

    hits = state.cache.get(key)
    cache_hit = hits is not None
    timings: dict[str, float] = {}
    if not cache_hit:
        with state.guard.slot():
            try:
                result = bundle.search(
                    req.query, mode, req.target_languages, k=pool_size, pool=pool_size
                )
            except ArtifactError:
                raise ApiError(
                    503, "INDEX_UNAVAILABLE", "No search index is currently available."
                ) from None
        hits, timings = result.hits, result.timings_ms
        state.cache.set(key, hits)

    db_started = time.perf_counter()
    with state.session_factory() as session:
        pairs = [tuple(h.key.split(":", 1)) for h in hits]
        rows = (
            session.execute(
                select(Document, ContentSource.name)
                .join(ContentSource, ContentSource.id == Document.source_id)
                .where(
                    tuple_(Document.language, Document.external_id).in_(pairs),
                    Document.is_active.is_(True),
                    ContentSource.name.in_(req.sources),
                )
            ).all()
            if pairs
            else []
        )
        by_key = {f"{d.language}:{d.external_id}": (d, name) for d, name in rows}
        kept = [h for h in hits if h.key in by_key]
        dropped = len(hits) - len(kept)

        run = SearchRun(
            id=uuid.uuid4(),
            query_sha256=qhash,
            analytics_query_text=(
                canonical_query(req.query)[:500] if settings.analytics_store_query_text else None
            ),
            query_language=req.query_language,
            target_languages=sorted(req.target_languages),
            sources=sorted(req.sources),
            requested_mode=mode,
            index_version_id=version_id,
            result_pool_size=len(kept),
            encoding_ms=timings.get("encode_ms"),
            lexical_ms=timings.get("lexical_ms"),
            dense_ms=timings.get("faiss_ms"),
            fusion_ms=timings.get("fusion_ms"),
            cache_hit=cache_hit,
            status="ok",
        )
        session.add(run)
        session.flush()
        shown_at = datetime.now(UTC)
        session.add_all(
            SearchResult(
                search_run_id=run.id,
                document_id=by_key[h.key][0].id,
                rank=rank,
                shown_at=shown_at if rank <= req.page_size else None,  # first page = impressions
                score=h.score,
                dense_rank=h.dense_rank,
                dense_score=h.dense_score,
                lexical_rank=h.lexical_rank,
                lexical_score=h.lexical_score,
            )
            for rank, h in enumerate(kept, start=1)
        )
        db_ms = (time.perf_counter() - db_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        run.database_ms, run.total_ms = db_ms, total_ms
        session.flush()

        page = kept[: req.page_size]
        items = [
            _item(r, *by_key[h.key], settings, h.score, h.dense_rank, h.lexical_rank)
            for r, h in enumerate(page, start=1)
        ]
        attribution = load_attribution(session, {i.source for i in items} or set(req.sources))
        session.commit()
        search_id = str(run.id)

    next_cursor = (
        encode_cursor(state.cursor_secret, search_id, len(page) + 1)
        if len(kept) > len(page)
        else None
    )
    log_event(
        "search_completed",
        request_id=request_id, search_id=search_id, route="/api/v1/search", status=200,
        cache_hit=cache_hit, result_count=len(kept), languages=sorted(req.target_languages),
        sources=sorted(req.sources), mode=mode, index_version=bundle.version,
        query_sha256=qhash[:16], dropped_documents=dropped, db_ms=round(db_ms, 3),
        total_ms=round(total_ms, 3),
        **{k: round(v, 3) for k, v in timings.items() if k in STAGE_FIELDS},
    )  # fmt: skip
    return SearchResponse(
        search_id=search_id,
        results=items,
        next_cursor=next_cursor,
        total_results=len(kept),
        attribution=attribution,
        index_version=bundle.version,
        mode=mode,
        cache_hit=cache_hit,
        latency_ms=round(total_ms, 3),
    )


# ------------------------------------------------------------------ stored pages
def get_page(state: AppState, search_id: str, cursor: str | None, limit: int) -> ResultPage:
    try:
        run_uuid = uuid.UUID(search_id)
    except ValueError:
        raise ApiError(
            404, "SEARCH_NOT_FOUND", "The search was not found or has expired."
        ) from None
    with state.session_factory() as session:
        run = session.get(SearchRun, run_uuid)
        if run is None:
            raise ApiError(404, "SEARCH_NOT_FOUND", "The search was not found or has expired.")
        start = (
            decode_cursor(state.cursor_secret, cursor, str(run.id), run.result_pool_size)
            if cursor
            else 1
        )
        rows = session.execute(
            select(SearchResult, Document, ContentSource.name)
            .join(Document, Document.id == SearchResult.document_id)
            .join(ContentSource, ContentSource.id == Document.source_id)
            .where(SearchResult.search_run_id == run.id, SearchResult.rank >= start)
            .order_by(SearchResult.rank)
            .limit(limit)
        ).all()
        items = [
            _item(r.rank, d, name, state.settings, r.score, r.dense_rank, r.lexical_rank)
            for r, d, name in rows
        ]
        attribution = load_attribution(session, {i.source for i in items} or set(run.sources))
        total = run.result_pool_size
        if items:  # these results are being returned now: record the impressions
            session.execute(
                update(SearchResult)
                .where(
                    SearchResult.search_run_id == run.id,
                    SearchResult.rank.in_([i.rank for i in items]),
                    SearchResult.shown_at.is_(None),
                )
                .values(shown_at=datetime.now(UTC))
            )
            session.commit()
    last = items[-1].rank if items else start - 1
    next_cursor = encode_cursor(state.cursor_secret, search_id, last + 1) if last < total else None
    return ResultPage(
        search_id=search_id,
        results=items,
        next_cursor=next_cursor,
        total_results=total,
        attribution=attribution,
    )


def get_document(state: AppState, document_id: str) -> DocumentResponse:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ApiError(404, "DOCUMENT_NOT_FOUND", "The document was not found.") from None
    with state.session_factory() as session:
        row = session.execute(
            select(Document, ContentSource.name)
            .join(ContentSource, ContentSource.id == Document.source_id)
            .where(Document.id == doc_uuid, Document.is_active.is_(True))
        ).first()
        if row is None:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "The document was not found.")
        doc, name = row
        attribution = load_attribution(session, {name})[name]
        return DocumentResponse(
            document_id=str(doc.id),
            title=safe_text(doc.title, 300),
            summary=safe_text(doc.summary, state.settings.summary_max_chars),
            language=doc.language,
            source=name,
            source_url=doc.source_url,
            source_revision=doc.source_revision,
            attribution=attribution,
        )


def purge_expired_pools(session: Session, retention_days: int, now: datetime | None = None) -> int:
    """Delete stored search runs (and their result pools) older than the retention period.

    Run this from an operator script or scheduler once analytics needs are met; it is not
    called automatically. Returns the number of search runs removed.
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    result = session.execute(delete(SearchRun).where(SearchRun.created_at < cutoff))
    session.commit()
    return result.rowcount or 0
