"""Typed schemas for the optional answer mode (``POST /api/v1/answers``).

Two groups: the public request/response, and ``ProviderAnswer``, the structure a language model
must return. A model's output is never trusted until it parses as ``ProviderAnswer`` and every
citation label it uses is checked against the evidence that was actually supplied.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.api.schemas import Language, Mode, ResultItem
from app.rag.types import AbstentionReason, CacheStatus, FallbackReason

_WS = re.compile(r"\s+")
CITATION_LABEL = re.compile(r"^P[1-9][0-9]?$")  # local labels only: P1, P2, ... never real ids
MAX_QUESTION_CHARS = 500  # hard schema cap; the configured limit (default 300) is stricter
MAX_CLAIMS = 8
MAX_CITATIONS_PER_CLAIM = 4
MAX_CLAIM_CHARS = 500
MAX_ANSWER_CHARS = 1500


def _no_controls(value: str) -> str:
    """Reject control characters and collapse whitespace (same rule as search queries)."""
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in value):
        raise ValueError("control characters are not allowed")
    return _WS.sub(" ", value).strip()


# --------------------------------------------------------------------------- request
class AnswerRequest(BaseModel):
    """Deliberately small: no model, prompt, temperature or token controls reach clients."""

    model_config = ConfigDict(extra="forbid")

    question: str
    languages: list[Language] = Field(default=["en", "es", "fr"], min_length=1, max_length=3)
    retrieval_mode: Mode | None = None  # None -> the active index's evidence-selected default
    include_search_results: bool = True

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        value = _no_controls(value)
        if not 3 <= len(value) <= MAX_QUESTION_CHARS:
            raise ValueError(f"question must contain between 3 and {MAX_QUESTION_CHARS} characters")
        return value

    @field_validator("languages")
    @classmethod
    def _no_duplicate_languages(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate language filters")
        return values


# ------------------------------------------------------------- what a model must return
class ProviderClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, Field(min_length=1, max_length=MAX_CLAIM_CHARS)]
    citation_ids: Annotated[
        list[str], Field(max_length=MAX_CITATIONS_PER_CLAIM)
    ]  # may be empty here; the validator rejects uncited claims with a precise reason

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        value = _no_controls(value)
        if not value:
            raise ValueError("claim text is empty")
        return value

    @field_validator("citation_ids")
    @classmethod
    def _labels(cls, labels: list[str]) -> list[str]:
        if any(not CITATION_LABEL.match(label) for label in labels):
            raise ValueError("citation labels must look like P1, P2, ...")
        return list(dict.fromkeys(labels))  # drop repeats, keep order


class ProviderAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Annotated[str, Field(max_length=MAX_ANSWER_CHARS)] | None
    claims: Annotated[list[ProviderClaim], Field(max_length=MAX_CLAIMS)]
    abstained: bool
    abstention_reason: Annotated[str, Field(max_length=300)] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ProviderAnswer:
        if self.abstained:
            if self.answer or self.claims:
                raise ValueError("an abstention must not contain an answer or claims")
        else:
            if not self.answer or not self.answer.strip():
                raise ValueError("an answer is required when not abstaining")
            if not self.claims:
                raise ValueError("an answer needs at least one claim")
        return self


# --------------------------------------------------------------------------- response
class Citation(BaseModel):
    """Resolved by the server from stored passages, never copied from model output."""

    label: str  # the local label the claim used, e.g. "P1"
    passage_id: str
    document_id: str
    title: str
    language: Language
    source_url: str  # a stored canonical http(s) URL
    excerpt: str  # the exact stored passage text, plain text

    @field_validator("source_url")
    @classmethod
    def _http_only(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be http or https")
        return value


class AnswerClaim(BaseModel):
    text: str
    citation_labels: list[str]


class AnswerVersions(BaseModel):
    index_version: str | None
    passage_artifact_version: str | None
    prompt_version: str
    context_builder_version: str
    provider: str | None
    model: str | None


class AnswerLatency(BaseModel):
    retrieval_ms: float | None = None
    provider_ms: float | None = None
    validation_ms: float | None = None
    total_ms: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AnswerResponse(BaseModel):
    run_id: str
    answer: str | None
    claims: list[AnswerClaim]
    citations: list[Citation]
    abstained: bool
    abstention_reason: AbstentionReason | None
    fallback_used: bool
    fallback_reason: FallbackReason | None
    search_results: list[ResultItem] | None  # present on fallback, or when requested
    versions: AnswerVersions
    latency: AnswerLatency
    token_usage: TokenUsage | None
    cache_status: CacheStatus

    @model_validator(mode="after")
    def _abstention_and_fallback_differ(self) -> AnswerResponse:
        if self.abstained and self.fallback_used:
            raise ValueError("a response is an abstention or a fallback, never both")
        if self.abstained != (self.abstention_reason is not None):
            raise ValueError("abstention_reason must be set exactly when abstained")
        if self.fallback_used != (self.fallback_reason is not None):
            raise ValueError("fallback_reason must be set exactly when fallback_used")
        if (self.abstained or self.fallback_used) and (self.answer or self.claims):
            raise ValueError("no answer text may accompany an abstention or fallback")
        return self


class FeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["helpful", "not_helpful"]
    reason: Literal["inaccurate", "missing_citation", "unclear", "other"] | None = None

    @model_validator(mode="after")
    def _reason_only_with_a_complaint(self) -> FeedbackBody:
        if self.rating == "helpful" and self.reason is not None:
            raise ValueError("a reason can only accompany a not_helpful rating")
        return self


class AnswerFeedbackResponse(BaseModel):
    run_id: str
    rating: Literal["helpful", "not_helpful"]
    reason: str | None
    updated_at: str
