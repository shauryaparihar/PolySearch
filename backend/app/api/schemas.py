"""Typed request/response schemas for /api/v1."""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Language = Literal["en", "es", "fr"]
Mode = Literal["dense", "lexical", "hybrid"]
Source = Literal["wikipedia"]
_WS = re.compile(r"\s+")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_language: Language
    target_languages: list[Language] = Field(min_length=1, max_length=3)
    sources: list[Source] = Field(default=["wikipedia"], min_length=1, max_length=1)
    mode: Mode | None = None  # None -> the active index's evidence-selected default
    page_size: int = Field(default=10, ge=5, le=20)

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in value):
            raise ValueError("control characters are not allowed")
        value = _WS.sub(" ", value).strip()
        if not 2 <= len(value) <= 500:
            raise ValueError("query must contain between 2 and 500 characters")
        return value

    @field_validator("target_languages", "sources")
    @classmethod
    def _no_duplicates(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate filter values")
        return values


class SourceAttribution(BaseModel):
    name: str
    licence_name: str
    licence_url: str
    attribution_text: str


class ResultItem(BaseModel):
    rank: int
    document_id: str
    title: str
    snippet: str  # bounded plain text; render as text, never as HTML
    language: Language
    source: str
    source_url: str
    source_revision: str | None
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class ResultPage(BaseModel):
    search_id: str
    results: list[ResultItem]
    next_cursor: str | None
    total_results: int
    attribution: dict[str, SourceAttribution]


class SearchResponse(ResultPage):
    index_version: str
    mode: Mode
    cache_hit: bool
    latency_ms: float


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    summary: str
    language: Language
    source: str
    source_url: str
    source_revision: str | None
    attribution: SourceAttribution


# ----------------------------------------------------------------- Phase 5 schemas
class ClickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    # Optional client-generated id so a retried browser event is not counted twice.
    idempotency_key: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{8,64}$")


class ClickResponse(BaseModel):
    click_id: str
    rank: int
    duplicate: bool


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["relevant", "not_relevant"]


class FeedbackResponse(BaseModel):
    document_id: str
    label: Literal["relevant", "not_relevant"]
    updated_at: str


class SuggestionItem(BaseModel):
    text: str
    languages: list[Language]


class SuggestionResponse(BaseModel):
    prefix: str
    suggestions: list[SuggestionItem]


# ----------------------------------------------------------------- Phase 6 schemas
class IngestionJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Language
    mode: Literal["seed", "incremental"]
    titles: list[Annotated[str, Field(min_length=1, max_length=255)]] | None = Field(
        default=None, min_length=1, max_length=50
    )
    max_pages: int | None = Field(default=None, ge=1, le=1000)
    resume: bool = False

    @field_validator("titles")
    @classmethod
    def _titles(cls, titles: list[str] | None) -> list[str] | None:
        if titles is None:
            return None
        cleaned = [t.strip() for t in titles]
        if any(not t or "|" in t or any(ord(c) < 32 for c in t) for t in cleaned):
            raise ValueError("invalid page title")
        return cleaned

    @model_validator(mode="after")
    def _combination(self):
        if self.mode == "incremental" and (self.titles or self.resume):
            raise ValueError("titles and resume apply to seed jobs only")
        return self


class IndexJobRequest(BaseModel):
    """Rebuilding takes no parameters: it indexes exactly the active documents in the database."""

    model_config = ConfigDict(extra="forbid")


class JobResponse(BaseModel):
    job_id: str
    kind: Literal["ingestion", "index"]
    status: Literal["queued", "running", "completed", "partial", "failed"]
    created_at: str
    started_at: str | None
    finished_at: str | None
    error_code: str | None
    error_message: str | None
    # ingestion jobs
    language: str | None = None
    mode: str | None = None
    counters: dict[str, int] | None = None
    failed_items: list[dict[str, str]] | None = None
    # index jobs
    step: str | None = None
    documents_total: int | None = None
    documents_encoded: int | None = None
    staged_version: str | None = None
    index_version: str | None = None
