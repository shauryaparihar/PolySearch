"""Creating, reading and recovering jobs (the database is the source of truth)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import ContentSource, IndexJob, IngestionJob
from app.jobs.states import ACTIVE, check_transition

RESTART_CODE = "RESTART_INTERRUPTED"
RESTART_MESSAGE = "The server restarted before this job finished; run it again."


class JobConflict(Exception):
    """Another job is already queued or running (the single worker runs one at a time)."""


class SourceNotInitialised(Exception):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def active_job_count(session: Session) -> int:
    return sum(
        session.scalar(select(func.count()).select_from(model).where(model.status.in_(ACTIVE)))
        for model in (IndexJob, IngestionJob)
    )


def create_index_job(factory: sessionmaker[Session]) -> uuid.UUID:
    with factory() as session:
        if active_job_count(session):
            raise JobConflict
        job = IndexJob(status="queued", step="queued")
        session.add(job)
        try:
            session.commit()  # the partial unique index also rejects a racing second job
        except IntegrityError:
            session.rollback()
            raise JobConflict from None
        return job.id


def create_ingestion_job(factory: sessionmaker[Session], mode: str, language: str) -> uuid.UUID:
    with factory() as session:
        source_id = session.scalar(
            select(ContentSource.id).where(ContentSource.name == "wikipedia")
        )
        if source_id is None:
            raise SourceNotInitialised
        if active_job_count(session):
            raise JobConflict
        job = IngestionJob(
            source_id=source_id,
            mode=mode,
            language=language,
            status="queued",
            fetched_count=0,
            created_count=0,
            updated_count=0,
            unchanged_count=0,
            failed_count=0,
            failed_items=[],
        )
        session.add(job)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise JobConflict from None
        return job.id


def transition(session: Session, job, new_status: str, **fields) -> None:
    """Move a job to ``new_status`` (validated) and set fields such as error_code."""
    check_transition(job.status, new_status)
    job.status = new_status
    if new_status == "running":
        job.started_at = _now()
    if new_status in ("completed", "partial", "failed"):
        job.finished_at = _now()
    for key, value in fields.items():
        setattr(job, key, value)


def recover_stale_jobs(factory: sessionmaker[Session]) -> int:
    """At startup no worker can be mid-job, so any queued/running job is a leftover of a
    previous process. Mark it failed so it never appears to be running forever. An in-process
    worker does not survive restarts (a durable queue would be needed for that)."""
    recovered = 0
    with factory() as session:
        for model in (IndexJob, IngestionJob):
            for job in session.scalars(select(model).where(model.status.in_(ACTIVE))):
                transition(
                    session, job, "failed", error_code=RESTART_CODE, error_message=RESTART_MESSAGE
                )
                recovered += 1
        session.commit()
    return recovered
