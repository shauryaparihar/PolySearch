"""Typed request/response schemas for /api/v1."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
