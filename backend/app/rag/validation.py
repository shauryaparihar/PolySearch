"""Citation *verification*: nothing a model returns is shown until it passes these checks.

A model can generate citation-shaped text, but only the server can say whether a citation is
real. Parsing and checking happen here, against the evidence that was actually supplied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.api.answer_schemas import ProviderAnswer
from app.rag.context import EvidenceContext, clean_for_prompt
from app.rag.types import FallbackReason

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class AnswerRejected(Exception):
    """The model output cannot be trusted. ``reason`` is a stable fallback code."""

    def __init__(self, reason: FallbackReason):
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class ValidClaim:
    ordinal: int
    text: str
    labels: list[str]  # all verified to exist in the supplied evidence


@dataclass(frozen=True)
class ValidatedAnswer:
    abstained: bool
    answer: str | None
    claims: list[ValidClaim]


def _label_order(label: str) -> int:
    return int(label[1:])


def parse_provider_output(raw: str) -> ProviderAnswer:
    """Strict JSON -> ``ProviderAnswer``. Anything else is INVALID_OUTPUT (repair-eligible)."""
    text = (raw or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return ProviderAnswer.model_validate(json.loads(text))
    except (ValueError, ValidationError):  # JSONDecodeError is a ValueError
        raise AnswerRejected(FallbackReason.INVALID_OUTPUT) from None


def validate_answer(parsed: ProviderAnswer, context: EvidenceContext) -> ValidatedAnswer:
    """Every claim needs at least one citation, and every cited label must be in the context."""
    if parsed.abstained:
        return ValidatedAnswer(abstained=True, answer=None, claims=[])
    known = context.labels
    claims: list[ValidClaim] = []
    for ordinal, claim in enumerate(parsed.claims):
        if not claim.citation_ids or any(label not in known for label in claim.citation_ids):
            raise AnswerRejected(FallbackReason.INVALID_CITATION)
        claims.append(
            ValidClaim(
                ordinal=ordinal,
                text=clean_for_prompt(claim.text),
                labels=sorted(claim.citation_ids, key=_label_order),
            )
        )
    # The displayed answer is built from the verified claims only. The model's free-text
    # ``answer`` is ignored, so no sentence can reach the user without a checked citation.
    answer = " ".join(c.text for c in claims).strip()
    if not answer or any(not c.text for c in claims):
        raise AnswerRejected(FallbackReason.INVALID_OUTPUT)
    return ValidatedAnswer(abstained=False, answer=answer, claims=claims)
