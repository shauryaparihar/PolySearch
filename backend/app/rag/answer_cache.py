"""Version-aware cache of *validated* answers.

Only a cited answer that passed every check, or a stable abstention, is cached. Fallbacks and
provider errors never are. The key contains everything that can change the result, so a change
to any of them is automatically a miss; nothing is purged by hand.

A hit skips the provider call (no cost, no tokens). The citations are still resolved from
PostgreSQL on every hit, so stored source data is never served stale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from app.api.cache import canonical_query
from app.rag.context import EvidenceContext
from app.rag.types import CONTEXT_BUILDER_VERSION, PROMPT_VERSION
from app.rag.validation import ValidatedAnswer


@dataclass(frozen=True)
class CachedAnswer:
    kind: Literal["answered", "abstained"]
    validated: ValidatedAnswer


def passage_set_fingerprint(context: EvidenceContext) -> str:
    """Which exact passages, in which order, under which labels, the model was shown."""
    parts = [
        f"{item.label}:{item.passage.passage_id}:{hashlib.sha256(item.text.encode()).hexdigest()}"
        for item in context.items
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def answer_cache_key(
    *,
    question: str,
    languages: list[str],
    mode: str,
    index_version: str,
    passage_artifact_version: str,
    context: EvidenceContext,
    provider: str,
    model: str,
    max_output_tokens: int,
    json_mode: bool,
) -> str:
    canonical = json.dumps(
        {
            "q": hashlib.sha256(canonical_query(question).casefold().encode()).hexdigest(),
            "langs": sorted(languages),
            "mode": mode,
            "index": index_version,
            "passages_artifact": passage_artifact_version,
            "passage_set": passage_set_fingerprint(context),
            "prompt": PROMPT_VERSION,
            "context_builder": CONTEXT_BUILDER_VERSION,
            "provider": provider,
            "model": model,
            "max_output_tokens": max_output_tokens,
            "json_mode": json_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
