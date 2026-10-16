"""Product signals: clicks and relevance feedback.

A click or label is accepted only for a result that this search actually *showed* (the
database also enforces that the document belongs to the search through a composite foreign
key). The rank is read from the stored pool, never trusted from the client.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.api.errors import ApiError
from app.api.schemas import ClickResponse, FeedbackResponse
from app.api.state import AppState
from app.db.models import ClickEvent, RelevanceFeedback, SearchResult, SearchRun


def _not_found_search() -> ApiError:
    return ApiError(404, "SEARCH_NOT_FOUND", "The search was not found or has expired.")


def _shown_result(session, search_id: str, document_id: uuid.UUID) -> SearchResult:
    try:
        run_id = uuid.UUID(search_id)
    except ValueError:
        raise _not_found_search() from None
    if session.get(SearchRun, run_id) is None:
        raise _not_found_search()
    result = session.scalar(
        select(SearchResult).where(
            SearchResult.search_run_id == run_id,
            SearchResult.document_id == document_id,
            SearchResult.shown_at.is_not(None),
        )
    )
    if result is None:
        raise ApiError(404, "RESULT_NOT_FOUND", "That document was not shown by this search.")
    return result


def record_click(
    state: AppState, search_id: str, document_id: uuid.UUID, idempotency_key: str | None
) -> tuple[ClickResponse, bool]:
    """Returns ``(response, created)``; a repeated idempotency key returns the first click."""
    with state.session_factory() as session:
        result = _shown_result(session, search_id, document_id)
        if idempotency_key:
            existing = session.scalar(
                select(ClickEvent).where(
                    ClickEvent.search_run_id == result.search_run_id,
                    ClickEvent.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                return ClickResponse(
                    click_id=str(existing.id), rank=existing.rank, duplicate=True
                ), False
        click = ClickEvent(
            search_run_id=result.search_run_id,
            document_id=document_id,
            rank=result.rank,
            idempotency_key=idempotency_key,
        )
        session.add(click)
        try:
            session.commit()
        except IntegrityError:  # two identical retries raced: the second becomes a duplicate
            session.rollback()
            existing = session.scalar(
                select(ClickEvent).where(
                    ClickEvent.search_run_id == result.search_run_id,
                    ClickEvent.idempotency_key == idempotency_key,
                )
            )
            if existing is None:
                raise
            return ClickResponse(
                click_id=str(existing.id), rank=existing.rank, duplicate=True
            ), False
        return ClickResponse(click_id=str(click.id), rank=click.rank, duplicate=False), True


def set_feedback(
    state: AppState, search_id: str, document_id: uuid.UUID, label: str
) -> FeedbackResponse:
    """Create or change the label for one result of one search (one label per result)."""
    with state.session_factory() as session:
        result = _shown_result(session, search_id, document_id)
        now = datetime.now(UTC)
        stmt = pg_insert(RelevanceFeedback).values(
            id=uuid.uuid4(),
            search_run_id=result.search_run_id,
            document_id=document_id,
            label=label,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_relevance_feedback_run_doc",
            set_={"label": label, "updated_at": now},
        )
        session.execute(stmt)
        session.commit()
        return FeedbackResponse(
            document_id=str(document_id), label=label, updated_at=now.isoformat()
        )
