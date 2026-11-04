"""Bounded concurrency for provider calls (process-local)."""

from __future__ import annotations

import threading
from contextlib import contextmanager


class GenerationBusy(Exception):
    """No generation slot became free within the wait timeout."""


class GenerationGuard:
    """At most ``max_concurrent`` provider calls at once; others wait briefly, then the request
    falls back to search instead of queueing without bound."""

    def __init__(self, max_concurrent: int, wait_timeout: float):
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.wait_timeout = wait_timeout
        self.max_concurrent = max_concurrent

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self.wait_timeout):
            raise GenerationBusy
        try:
            yield
        finally:
            self._sem.release()
