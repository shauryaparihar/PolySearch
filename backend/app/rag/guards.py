"""Bounded concurrency for provider calls, shared by every API process.

A process-local semaphore keeps waiting requests from each holding a database connection; the
real limit is a set of PostgreSQL advisory locks (one per slot) held on a dedicated connection
for the duration of the call. Advisory locks are released automatically if the process or its
connection dies, so a crash cannot leave a slot taken. Without an engine (unit tests of the
guard itself) only the local semaphore is used.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Engine

LOCK_NAMESPACE = 0x504F4C59  # "POLY": the first key of the two-key advisory lock


class GenerationBusy(Exception):
    """No generation slot became free within the wait timeout."""


class GenerationGuard:
    """At most ``max_concurrent`` provider calls at once across all processes; others wait
    briefly, then the request falls back to search instead of queueing without bound."""

    def __init__(
        self,
        max_concurrent: int,
        wait_timeout: float,
        engine: Engine | None = None,
        poll_interval: float = 0.02,
    ):
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.wait_timeout = wait_timeout
        self.max_concurrent = max_concurrent
        self._engine = engine
        self._poll = poll_interval

    @contextmanager
    def slot(self):
        deadline = time.monotonic() + self.wait_timeout
        if not self._sem.acquire(timeout=self.wait_timeout):
            raise GenerationBusy
        try:
            if self._engine is None:
                yield
                return
            with self._engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                slot_no = self._take_database_slot(conn, deadline)
                try:
                    yield
                finally:
                    try:
                        conn.execute(
                            text("select pg_advisory_unlock(:ns, :slot)"),
                            {"ns": LOCK_NAMESPACE, "slot": slot_no},
                        )
                    except Exception:  # noqa: BLE001  (a broken connection frees its locks anyway)
                        conn.invalidate()
        finally:
            self._sem.release()

    def _take_database_slot(self, conn, deadline: float) -> int:
        while True:
            for slot_no in range(self.max_concurrent):
                got = conn.execute(
                    text("select pg_try_advisory_lock(:ns, :slot)"),
                    {"ns": LOCK_NAMESPACE, "slot": slot_no},
                ).scalar_one()
                if got:
                    return slot_no
            if time.monotonic() >= deadline:
                raise GenerationBusy
            time.sleep(self._poll)
