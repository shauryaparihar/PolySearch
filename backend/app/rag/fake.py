"""Deterministic stand-in for a provider. Used by tests and CI; never calls the network."""

from __future__ import annotations

import json
from collections.abc import Sequence

from app.rag.llm import (
    LLMError,
    LLMProviderError,
    LLMRateLimited,
    LLMRequest,
    LLMResponse,
    LLMTimeout,
    LLMUsage,
)

VALID_ANSWER = {
    "answer": "A black hole is a region of spacetime from which nothing can escape.",
    "claims": [
        {
            "text": "A black hole is a region of spacetime from which nothing can escape.",
            "citation_ids": ["P1"],
        }
    ],
    "abstained": False,
    "abstention_reason": None,
}
ABSTENTION = {
    "answer": None,
    "claims": [],
    "abstained": True,
    "abstention_reason": "The supplied evidence does not answer the question.",
}

DEFAULT_USAGE = LLMUsage(input_tokens=120, output_tokens=40)
MODES = ("success", "abstain", "malformed", "invalid_citation", "timeout", "rate_limited", "error")


class FakeLLMClient:
    """``mode`` picks one behaviour for every call; ``script`` (a list of response texts or
    ``LLMError`` instances, consumed in order) lets a test drive a repair attempt."""

    provider = "fake"

    def __init__(
        self,
        mode: str = "success",
        script: Sequence[str | LLMError] | None = None,
        model: str = "fake-model",
        usage: LLMUsage | None = DEFAULT_USAGE,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown fake mode {mode!r}")
        self.mode, self.model, self.usage = mode, model, usage
        self._script = list(script) if script is not None else None
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._script is not None:
            item = self._script[min(len(self.calls), len(self._script)) - 1]
            if isinstance(item, LLMError):
                raise item
            return LLMResponse(text=item, usage=self.usage, model=self.model)
        if self.mode == "timeout":
            raise LLMTimeout
        if self.mode == "rate_limited":
            raise LLMRateLimited
        if self.mode == "error":
            raise LLMProviderError
        if self.mode == "malformed":
            text = '{"answer": "unterminated'
        elif self.mode == "abstain":
            text = json.dumps(ABSTENTION)
        elif self.mode == "invalid_citation":
            bad = {**VALID_ANSWER, "claims": [{"text": "x", "citation_ids": ["P99"]}]}
            text = json.dumps(bad)
        else:
            text = json.dumps(VALID_ANSWER)
        return LLMResponse(text=text, usage=self.usage, model=self.model)
