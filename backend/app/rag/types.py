"""Stable vocabulary for the optional answer mode: statuses, reason codes and version labels.

These strings are stored in PostgreSQL, returned by the API and used in logs, so they are an
interface. Add new members; do not rename existing ones.
"""

from __future__ import annotations

from enum import StrEnum

PROMPT_VERSION = "answer-prompt-v1"
CONTEXT_BUILDER_VERSION = "context-v1"
PASSAGE_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    ANSWERED = "answered"  # a validated, cited answer was produced
    ABSTAINED = "abstained"  # the system worked, but the evidence was not enough
    FALLBACK = "fallback"  # an answer could not be produced or trusted: search results only


class AbstentionReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # retrieved evidence too weak or too short
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MODEL_ABSTAINED = "MODEL_ABSTAINED"  # the model itself declined (validated, so still safe)


class FallbackReason(StrEnum):
    RAG_DISABLED = "RAG_DISABLED"
    LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
    RAG_UNAVAILABLE = "RAG_UNAVAILABLE"  # no usable passage artifact for the active index
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"  # malformed structure, even after the one repair
    INVALID_CITATION = "INVALID_CITATION"  # a citation not in the supplied evidence, or uncited
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BUSY = "BUSY"  # no generation slot free within the wait timeout


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"  # not cacheable (fallbacks and errors are never cached)


class FeedbackRating(StrEnum):
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


class FeedbackReason(StrEnum):
    INACCURATE = "inaccurate"
    MISSING_CITATION = "missing_citation"
    UNCLEAR = "unclear"
    OTHER = "other"


def values(enum_cls: type[StrEnum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_cls)
