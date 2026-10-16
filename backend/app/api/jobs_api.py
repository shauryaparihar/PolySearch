"""HTTP-facing job operations (admin only): start a job, read its status."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.api.errors import ApiError
from app.api.schemas import IngestionJobRequest, JobResponse
from app.api.state import AppState
from app.db.models import IndexJob, IndexVersion, IngestionJob
from app.jobs import manager
from app.jobs.manager import JobConflict, SourceNotInitialised

CONFLICT = ApiError(
    409, "JOB_CONFLICT", "Another sync or re-index job is already queued or running."
)


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _response(job, kind: str, session) -> JobResponse:
    common = dict(
        job_id=str(job.id),
        kind=kind,
        status=job.status,
        created_at=_iso(job.created_at),
        started_at=_iso(job.started_at),
        finished_at=_iso(job.finished_at),
        error_code=job.error_code,
        error_message=job.error_message,
    )
    if kind == "ingestion":
        return JobResponse(
            **common,
            language=job.language,
            mode=job.mode,
            counters={
                "fetched": job.fetched_count,
                "created": job.created_count,
                "updated": job.updated_count,
                "unchanged": job.unchanged_count,
                "failed": job.failed_count,
            },
            failed_items=(job.failed_items or [])[:10],
        )
    version = session.get(IndexVersion, job.index_version_id) if job.index_version_id else None
    return JobResponse(
        **common,
        step=job.step,
        documents_total=job.documents_total,
        documents_encoded=job.documents_encoded,
        staged_version=job.staged_version,  # a version name, never a filesystem path
        index_version=version.version if version else None,
    )


def _worker(state: AppState):
    if state.worker is None:
        raise ApiError(503, "JOBS_UNAVAILABLE", "Background jobs are not running.")
    return state.worker


def start_ingestion_job(state: AppState, body: IngestionJobRequest) -> JobResponse:
    worker = _worker(state)
    try:
        job_id = manager.create_ingestion_job(state.session_factory, body.mode, body.language)
    except JobConflict:
        raise CONFLICT from None
    except SourceNotInitialised:
        raise ApiError(
            409, "SOURCE_NOT_INITIALISED",
            "Import the corpus first (command line) so the content source exists.",
        ) from None  # fmt: skip
    max_pages = min(
        body.max_pages or state.settings.max_pages_per_job, state.settings.max_pages_per_job
    )
    params = {
        "language": body.language, "mode": body.mode, "titles": body.titles,
        "max_pages": max_pages, "resume": body.resume,
    }  # fmt: skip
    try:
        worker.submit("ingestion", job_id, params)
    except JobConflict:
        _abandon(state, IngestionJob, job_id)
        raise CONFLICT from None
    return _read(state, "ingestion", job_id)


def start_index_job(state: AppState) -> JobResponse:
    worker = _worker(state)
    try:
        job_id = manager.create_index_job(state.session_factory)
    except JobConflict:
        raise CONFLICT from None
    try:
        worker.submit("index", job_id, {})
    except JobConflict:
        _abandon(state, IndexJob, job_id)
        raise CONFLICT from None
    return _read(state, "index", job_id)


def _abandon(state: AppState, model, job_id: uuid.UUID) -> None:
    """The worker could not take the job: close the row so it does not block later jobs."""
    with state.session_factory() as session:
        job = session.get(model, job_id)
        if job is not None and job.status == "queued":
            manager.transition(
                session,
                job,
                "failed",
                error_code="WORKER_BUSY",
                error_message="The worker was busy.",
            )
            session.commit()


def _read(state: AppState, kind: str, job_id: uuid.UUID) -> JobResponse:
    model = IndexJob if kind == "index" else IngestionJob
    with state.session_factory() as session:
        return _response(session.get(model, job_id), kind, session)


def get_job(state: AppState, kind: str, job_id: str) -> JobResponse:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        raise ApiError(404, "JOB_NOT_FOUND", "The job was not found.") from None
    model = IndexJob if kind == "index" else IngestionJob
    with state.session_factory() as session:
        job = session.scalar(select(model).where(model.id == parsed))
        if job is None:
            raise ApiError(404, "JOB_NOT_FOUND", "The job was not found.")
        return _response(job, kind, session)
