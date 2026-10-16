"""Everything a request handler needs, assembled once at startup."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

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
