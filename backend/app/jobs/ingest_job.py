"""Run a queued ingestion job (Wikipedia sync) on the worker."""

from __future__ import annotations

import uuid
from typing import Any

from app.db.models import IngestionJob
from app.ingestion import service
from app.ingestion.sources import SourceError
from app.ingestion.wikipedia import WikipediaSource
from app.jobs.manager import transition
from app.jobs.states import ACTIVE


def default_source_factory(state):
    return WikipediaSource(state.settings)


def run_ingestion_job(
    state, job_id: uuid.UUID, params: dict[str, Any], stop, cancelled_exc
) -> None:
    """``params``: language, mode ('seed'|'incremental'), titles, max_pages, resume."""
    factory = state.session_factory
    source = (state.source_factory or (lambda: default_source_factory(state)))()
    try:
        language, mode = params["language"], params["mode"]
        if mode == "incremental":
            service.sync_incremental(
                factory, source, language, params.get("max_pages"), job_id=job_id
            )
        elif params.get("titles"):
            service.sync_titles(factory, source, language, params["titles"], job_id=job_id)
        else:
            token = service.last_continuation(factory, language) if params.get("resume") else None
            service.sync_enumerate(
                factory, source, language, params.get("max_pages"), token, job_id=job_id
            )
    except SourceError as exc:
        # Normally the sync service already recorded the failure. If it failed before the job
        # started (for example while fetching licence metadata) the row is still queued: close it.
        with factory() as session:
            job = session.get(IngestionJob, job_id)
            if job is not None and job.status in ACTIVE:
                transition(
                    session, job, "failed", error_code=exc.code, error_message=exc.public_message
                )
                session.commit()
    finally:
        source.close()
