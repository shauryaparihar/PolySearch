"""One application-owned worker thread that runs sync and re-index jobs, one at a time.

The queue holds at most one waiting job, and the database allows at most one queued/running
job, so work can never pile up. A job running when the process stops is interrupted
cooperatively where possible and otherwise recovered as failed at the next startup.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Callable
from typing import Any

from app.api.logging_config import log_event
from app.jobs import ingest_job, reindex
from app.jobs.manager import JobConflict, transition
from app.jobs.states import ACTIVE

_STOP = object()


class JobCancelled(Exception):
    """Raised inside a job when the worker is asked to stop."""


class JobWorker:
    def __init__(self, state, synchronous: bool = False):
        """``synchronous=True`` runs jobs inline in the caller's thread (used by tests)."""
        self.state = state
        self.synchronous = synchronous
        self.stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.synchronous or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="polysearch-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float) -> None:
        """Ask the worker to stop and wait up to ``timeout`` seconds for the current job."""
        self.stop_event.set()
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._thread = None

    # ------------------------------------------------------------------- work
    def submit(self, kind: str, job_id: uuid.UUID, params: dict[str, Any] | None = None) -> None:
        item = (kind, job_id, params or {})
        if self.synchronous:
            self._execute(*item)
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            raise JobConflict from None

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            item = self._queue.get()
            if item is _STOP:
                break
            self._execute(*item)
        # Anything still waiting never ran: fail it now rather than leave it queued forever.
        self._fail_unstarted()

    def _execute(self, kind: str, job_id: uuid.UUID, params: dict[str, Any]) -> None:
        handler: Callable = {
            "index": reindex.run_index_job,
            "ingestion": ingest_job.run_ingestion_job,
        }[kind]
        try:
            handler(self.state, job_id, params, self.stop_event, JobCancelled)
        except JobCancelled:
            self._fail(kind, job_id, "SHUTDOWN", "The job was interrupted by a server shutdown.")
        except Exception as exc:  # never let one bad job kill the worker thread
            log_event(
                "job_crashed",
                level=logging.ERROR,
                job_id=str(job_id),
                error_code=type(exc).__name__,
            )
            self._fail(kind, job_id, "INTERNAL_ERROR", "The job failed unexpectedly.")

    def _fail(self, kind: str, job_id: uuid.UUID, code: str, message: str) -> None:
        from app.db.models import IndexJob, IngestionJob

        model = IndexJob if kind == "index" else IngestionJob
        with self.state.session_factory() as session:
            job = session.get(model, job_id)
            if job is not None and job.status in ACTIVE:
                transition(session, job, "failed", error_code=code, error_message=message)
                session.commit()

    def _fail_unstarted(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                self._fail(
                    item[0], item[1], "SHUTDOWN", "The server stopped before the job started."
                )
