"""Small in-process sliding-window rate limiter.

State lives in one process, so with several replicas each enforces its own limit (a shared
store such as Redis becomes justified only then). Memory is bounded by ``max_keys``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        self.limit, self.window, self.clock, self.max_keys = limit, window_seconds, clock, max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _window(self, key: str) -> deque[float]:
        """The recent hits for ``key`` with expired ones removed (caller holds the lock)."""
        now = self.clock()
        hits = self._hits.get(key)
        if hits is None:
            hits = self._hits[key] = deque()
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)  # forget the least recently added client
        else:
            self._hits.move_to_end(key)
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits

    def _retry_after(self, hits: deque[float]) -> int:
        return max(1, math.ceil(self.window - (self.clock() - hits[0])))

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns ``(allowed, retry_after_seconds)``."""
        with self._lock:
            hits = self._window(key)
            if len(hits) >= self.limit:
                return False, self._retry_after(hits)
            hits.append(self.clock())
            return True, 0

    def blocked(self, key: str) -> tuple[bool, int]:
        """Is ``key`` at its limit? Does not record anything (used to lock out failures)."""
        with self._lock:
            hits = self._window(key)
            return (True, self._retry_after(hits)) if len(hits) >= self.limit else (False, 0)

    def record(self, key: str) -> None:
        """Count one event for ``key`` without checking the limit."""
        with self._lock:
            self._window(key).append(self.clock())


def client_id(peer: str | None, forwarded_for: str | None, trusted_proxies: list[str]) -> str:
    """Identify the caller. ``X-Forwarded-For`` is trusted only when the direct peer is a
    configured proxy; then the entry the proxy appended (the last one) is used."""
    peer = peer or "unknown"
    if forwarded_for and peer in trusted_proxies:
        last = forwarded_for.split(",")[-1].strip()
        if last:
            return last[:64]
    return peer
