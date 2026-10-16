"""Everything a request handler needs, assembled once at startup."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.api.cache import QueryCache
from app.api.guard import InferenceGuard
from app.api.ratelimit import SlidingWindowLimiter
from app.config import Settings
from app.search.bundle import SearchBundle
from app.search.trie import Suggester


@dataclass
class AppState:
    settings: Settings
    session_factory: sessionmaker[Session]
    bundle: SearchBundle | None
    cache: QueryCache
    guard: InferenceGuard
    search_limiter: SlidingWindowLimiter
    suggest_limiter: SlidingWindowLimiter
    write_limiter: SlidingWindowLimiter
    admin_limiter: SlidingWindowLimiter
    admin_failure_limiter: SlidingWindowLimiter
    cursor_secret: bytes
    index_version_id: uuid.UUID | None = None
    clock: Callable[[], float] = time.monotonic
    readiness: dict = field(default_factory=dict)
    suggester: Suggester | None = None
    # Process-lifetime counters (rate-limited requests are not stored in the database).
    metrics: dict = field(
        default_factory=lambda: {
            "rate_limited": {"search": 0, "suggest": 0, "write": 0, "admin": 0}
        }
    )
    # ---- background jobs (Phase 6)
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: Any = None  # app.jobs.worker.JobWorker, started in the lifespan
    source_factory: Callable[[], Any] | None = None  # builds the ContentSource for ingestion jobs

    def snapshot(self) -> tuple[SearchBundle | None, uuid.UUID | None]:
        """The served bundle and its database version id, read together.

        A search takes one snapshot and uses it for the whole request, so it can never pair
        the old bundle with the new version id while an activation is in progress.
        """
        with self.lock:
            return self.bundle, self.index_version_id

    def activate(
        self, bundle: SearchBundle, version_id: uuid.UUID, suggester: Suggester | None
    ) -> None:
        """Atomically replace the served index. In-flight searches keep their old snapshot."""
        with self.lock:
            self.bundle, self.index_version_id, self.suggester = bundle, version_id, suggester
            self.readiness.clear()
