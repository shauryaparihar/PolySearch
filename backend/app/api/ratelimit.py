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

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns ``(allowed, retry_after_seconds)``."""
        now = self.clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
                while len(self._hits) > self.max_keys:
                    self._hits.popitem(last=False)  # forget the least recently added client
            else:
                self._hits.move_to_end(key)
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            return True, 0


def client_id(peer: str | None, forwarded_for: str | None, trusted_proxies: list[str]) -> str:
    """Identify the caller. ``X-Forwarded-For`` is trusted only when the direct peer is a
    configured proxy; then the entry the proxy appended (the last one) is used."""
    peer = peer or "unknown"
    if forwarded_for and peer in trusted_proxies:
        last = forwarded_for.split(",")[-1].strip()
        if last:
            return last[:64]
    return peer
