"""OpenAI-compatible chat-completions client behind the ``LLMClient`` protocol.

Only this module talks HTTP to a provider. It never follows redirects (so the key cannot be
forwarded elsewhere), never logs or returns provider text, and turns every failure into a
fixed ``LLMError`` code.
"""

from __future__ import annotations

import httpx

from app.rag.llm import (
    LLMError,
    LLMProviderError,
    LLMRateLimited,
    LLMRequest,
    LLMResponse,
    LLMTimeout,
    LLMUsage,
)

PROVIDER = "openai-compatible"


class OpenAICompatibleClient:
    provider = PROVIDER

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        json_mode: bool = True,
        transport: httpx.BaseTransport | None = None,
    ):
        self._json_mode = json_mode
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._transport = transport

    def generate(self, request: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(
                transport=self._transport,
                timeout=request.timeout_seconds,
                follow_redirects=False,
            ) as http:
                response = http.post(self._url, headers=self._headers, json=body)
        except httpx.TimeoutException:
            raise LLMTimeout from None
        except httpx.HTTPError:
            raise LLMProviderError from None
        if response.status_code == 429:
            raise LLMRateLimited
        if response.status_code != 200:
            raise LLMProviderError
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            tokens = (
                LLMUsage(int(usage["prompt_tokens"]), int(usage["completion_tokens"]))
                if "prompt_tokens" in usage and "completion_tokens" in usage
                else None
            )
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMProviderError from None
        if not isinstance(text, str):
            raise LLMProviderError
        return LLMResponse(text=text, usage=tokens, model=self.model)


def build_llm_client(settings) -> OpenAICompatibleClient | None:
    """A real client only when answers are enabled and fully configured; otherwise None."""
    if not settings.rag_configured:
        return None
    return OpenAICompatibleClient(
        settings.llm_base_url,
        settings.llm_api_key.get_secret_value(),
        settings.llm_model,
        json_mode=settings.llm_json_mode,
    )


__all__ = ["OpenAICompatibleClient", "build_llm_client", "LLMError"]
