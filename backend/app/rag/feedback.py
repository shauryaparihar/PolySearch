"""One helpful / not-helpful rating per answer (re-rating replaces it)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.api.answer_schemas import AnswerFeedbackResponse, FeedbackBody
from app.api.errors import ApiError
from app.api.state import AppState
from app.db.models import GenerationFeedback, GenerationRun


def record_feedback(state: AppState, run_id: str, body: FeedbackBody) -> AnswerFeedbackResponse:
    not_found = ApiError(404, "ANSWER_NOT_FOUND", "The answer was not found.")
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise not_found from None
    with state.session_factory() as session:
        if session.get(GenerationRun, run_uuid) is None:
            raise not_found
        now = datetime.now(UTC)
        stmt = pg_insert(GenerationFeedback).values(
            id=uuid.uuid4(),
            generation_run_id=run_uuid,
            rating=body.rating,
            reason_code=body.reason,
            created_at=now,
            updated_at=now,
        )
        session.execute(
            stmt.on_conflict_do_update(
                constraint="uq_generation_feedback_run",
                set_={"rating": body.rating, "reason_code": body.reason, "updated_at": now},
            )
        )
        session.commit()
    return AnswerFeedbackResponse(
        run_id=run_id, rating=body.rating, reason=body.reason, updated_at=now.isoformat()
    )
