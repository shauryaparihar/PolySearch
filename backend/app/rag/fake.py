"""Deterministic stand-in for a provider. Used by tests and CI; never calls the network."""

from __future__ import annotations

import json
import re
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


# ------------------------------------------------------------------------------------------
# A deterministic *extractive* stand-in used by the offline evaluation. It is not a language
# model: it copies the sentence of the supplied evidence that shares the most words with the
# question and cites it. Results obtained with it must always be labelled "mock".
_BLOCK = re.compile(r"\[(P\d+)\] title:[^\n]*\n<<<BEGIN \1>>>\n(.*?)\n<<<END \1>>>", re.DOTALL)
_QUESTION = re.compile(r"<<<QUESTION>>>\n(.*?)\n<<<END QUESTION>>>", re.DOTALL)
_WORD = re.compile(r"[^\W_]{4,}", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveFakeLLM:
    provider = "mock-extractive"
    model = "mock-extractive-v1"

    def __init__(self, usage: LLMUsage | None = DEFAULT_USAGE):
        self.usage = usage
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        blocks = _BLOCK.findall(request.user)
        question = _QUESTION.search(request.user)
        if not blocks or question is None:
            if "<<<QUESTION>>>" in request.user:  # the answer pipeline found no evidence
                return LLMResponse(text=json.dumps(ABSTENTION), usage=self.usage, model=self.model)
            # the no-retrieval comparison: the model only ever sees the bare question
            return LLMResponse(
                text="I cannot answer without sources.", usage=self.usage, model=self.model
            )
        wanted = {w.casefold() for w in _WORD.findall(question.group(1))}
        best: tuple[int, int, str, str] | None = None
        for order, (label, body) in enumerate(blocks):
            for sentence in _SENTENCE.split(body):
                overlap = len(wanted & {w.casefold() for w in _WORD.findall(sentence)})
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, order, label, sentence.strip())
        if best is None:
            return LLMResponse(text=json.dumps(ABSTENTION), usage=self.usage, model=self.model)
        sentence = best[3][:480]
        payload = {
            "answer": sentence,
            "claims": [{"text": sentence, "citation_ids": [best[2]]}],
            "abstained": False,
            "abstention_reason": None,
        }
        return LLMResponse(text=json.dumps(payload), usage=self.usage, model=self.model)
