"""The only seam between PolySearch and a language-model provider.

Everything provider-specific (HTTP, authentication, response shapes) lives behind
``LLMClient``. Nothing outside this boundary sees the API key, and errors leaving it carry only
a stable code, never provider payloads, URLs or message text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.types import FallbackReason


@dataclass(frozen=True)
class LLMRequest:
    system: str  # fixed instructions (versioned prompt)
    user: str  # the question plus delimited, untrusted evidence
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str  # raw model output; the answer validator decides whether it can be trusted
    usage: LLMUsage | None  # provider-reported, None if the provider did not report it
    model: str


class LLMError(Exception):
    """A provider failure reduced to a stable ``FallbackReason`` code.

    The message is intentionally fixed so no provider text can leak into logs or responses.
    """

    def __init__(self, reason: FallbackReason):
        super().__init__(reason.value)
        self.reason = reason


class LLMTimeout(LLMError):
    def __init__(self) -> None:
        super().__init__(FallbackReason.PROVIDER_TIMEOUT)


class LLMRateLimited(LLMError):
    def __init__(self) -> None:
        super().__init__(FallbackReason.PROVIDER_RATE_LIMITED)


class LLMProviderError(LLMError):
    def __init__(self) -> None:
        super().__init__(FallbackReason.PROVIDER_ERROR)


@runtime_checkable
class LLMClient(Protocol):
    provider: str  # short label stored with each run, e.g. "openai-compatible" or "fake"
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        """One bounded generation call. Raises ``LLMError`` subclasses and nothing else."""
        ...
