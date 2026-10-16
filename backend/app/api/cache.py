"""Canonical query fingerprints and the version-aware result cache."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Sequence
from typing import Any

from cachetools import TTLCache

_WS = re.compile(r"\s+")


def canonical_query(query: str) -> str:
    return _WS.sub(" ", query).strip()


def query_sha256(query: str) -> str:
    """Fingerprint of the whitespace-normalised query text (used for analytics and logs)."""
    return hashlib.sha256(canonical_query(query).encode("utf-8")).hexdigest()


def cache_key(
    query: str,
    query_language: str,
    target_languages: Sequence[str],
    sources: Sequence[str],
    mode: str,
    pool_size: int,
    index_version: str,
) -> str:
    """SHA-256 over every input that can change the ranked pool. Including the index version
    makes entries from an older index unreachable after activation, so nothing is purged."""
    canonical = json.dumps(
        {
            "q": canonical_query(query),
            "ql": query_language,
            "tl": sorted(target_languages),
            "src": sorted(sources),
            "mode": mode,
            "pool": pool_size,
            "index": index_version,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QueryCache:
    """Bounded TTL cache of ranked candidate pools (never of click or feedback writes)."""

    def __init__(self, max_entries: int, ttl_seconds: float, timer=None):
        kwargs = {"timer": timer} if timer else {}
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=max_entries, ttl=ttl_seconds, **kwargs)
        self._lock = threading.Lock()  # TTLCache is not thread-safe

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
