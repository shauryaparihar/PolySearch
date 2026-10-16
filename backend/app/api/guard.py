"""Bounded concurrency for model inference."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from app.api.errors import ApiError


class InferenceGuard:
    """At most ``max_concurrent`` searches run the encoder at once; others wait briefly, then
    receive 503 instead of queueing without bound (so memory and latency stay bounded)."""

    def __init__(self, max_concurrent: int, wait_timeout: float):
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.wait_timeout = wait_timeout
        self.max_concurrent = max_concurrent

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self.wait_timeout):
            raise ApiError(
                503,
                "BUSY",
                "The search service is busy. Please retry shortly.",
                headers={"Retry-After": "1"},
            )
        try:
            yield
        finally:
            self._sem.release()
