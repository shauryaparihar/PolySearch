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
            "rate_limited": {"search": 0, "suggest": 0, "write": 0, "admin": 0, "answer": 0}
        }
    )
    # ---- background jobs (Phase 6)
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: Any = None  # app.jobs.worker.JobWorker, started in the lifespan
    source_factory: Callable[[], Any] | None = None  # builds the ContentSource for ingestion jobs
    # ---- optional answer mode: passages of the served index, or why there are none
    passages: Any = None  # app.rag.passage_store.PassageStore
    passage_error: str | None = None
    llm: Any = None  # app.rag.llm.LLMClient, or None when answers are not configured
    answer_limiter: Any = None  # SlidingWindowLimiter for POST /answers

    def snapshot(self) -> tuple[SearchBundle | None, uuid.UUID | None]:
        """The served bundle and its database version id, read together.

        A search takes one snapshot and uses it for the whole request, so it can never pair
        the old bundle with the new version id while an activation is in progress.
        """
        with self.lock:
            return self.bundle, self.index_version_id

    def snapshot_with_passages(self) -> tuple[SearchBundle | None, uuid.UUID | None, Any]:
        """Bundle, version id and passage store read together, so answers never pair the new
        index with the old passages (or the reverse) while an activation is in progress."""
        with self.lock:
            return self.bundle, self.index_version_id, self.passages

    def activate(
        self,
        bundle: SearchBundle,
        version_id: uuid.UUID,
        suggester: Suggester | None,
        passages: Any = None,
        passage_error: str | None = "PASSAGES_PENDING",
    ) -> None:
        """Atomically replace the served index (and its passages, which belong to exactly one
        index version). In-flight searches keep their old snapshot."""
        with self.lock:
            self.bundle, self.index_version_id, self.suggester = bundle, version_id, suggester
            self.passages = passages
            self.passage_error = None if passages is not None else passage_error
            self.readiness.clear()
