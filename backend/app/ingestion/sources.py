"""Source-independent ingestion types and the ContentSource interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

_CODE_RE = re.compile(r"[^a-z0-9_\-]")


class SourceError(Exception):
    """A sanitized source failure. ``code`` is safe to store and show; ``detail`` is not."""

    def __init__(
        self, code: str, message: str = "Source request failed", *, transient: bool = False
    ):
        code = _CODE_RE.sub("_", code.lower())[:64] or "unknown"
        super().__init__(f"{code}: {message}")
        self.code = code
        self.public_message = message[:200]
        self.transient = transient
        self.retry_after: float | None = None


@dataclass(frozen=True)
class PageStub:
    """Lightweight page identity + revision (no text), used to avoid refetching."""

    language: str
    title: str
    revision: str | None
    updated_at: datetime | None


@dataclass(frozen=True)
class RawRecord:
    language: str
    title: str
    extract: str
    url: str | None
    revision: str | None
    updated_at: datetime | None


@dataclass(frozen=True)
class NormalizedRecord:
    external_id: str
    language: str
    title: str
    summary: str
    source_url: str
    source_revision: str | None
    source_updated_at: datetime | None
    content_sha256: str


class ContentSource(Protocol):
    name: str

    def fetch_page(self, external_id: str, language: str) -> RawRecord | None: ...

    def fetch_updates(
        self, language: str, continuation_token: dict | None, limit: int
    ) -> tuple[list[PageStub], dict | None]: ...

    def normalise_record(self, raw: RawRecord) -> NormalizedRecord: ...
