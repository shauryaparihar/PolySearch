"""The OpenAI-compatible client: request shape, error mapping, nothing leaks."""

from __future__ import annotations

import json

import httpx
import pytest
from app.config import Settings
from app.rag.llm import LLMProviderError, LLMRateLimited, LLMRequest, LLMTimeout
from app.rag.openai_client import OpenAICompatibleClient, build_llm_client

KEY = "sk-very-secret-key"
REQUEST = LLMRequest(system="SYS", user="USER", max_output_tokens=77, timeout_seconds=3.0)


def client_for(handler) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        "https://api.invalid/v1/", KEY, "model-x", transport=httpx.MockTransport(handler)
    )


def ok(content="{}", usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_the_request_has_the_expected_shape_and_no_extra_controls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok('{"a": 1}', {"prompt_tokens": 10, "completion_tokens": 4})

    response = client_for(handler).generate(REQUEST)
    assert seen["url"] == "https://api.invalid/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "model-x" and body["max_tokens"] == 77 and body["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert "tools" not in body and "functions" not in body  # no tool calls, ever
    assert response.text == '{"a": 1}' and response.model == "model-x"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 4)


def test_missing_usage_is_reported_as_none_not_invented():
    assert client_for(lambda r: ok("{}")).generate(REQUEST).usage is None


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda r: httpx.Response(429, text=f"slow down {KEY}"), LLMRateLimited),
        (lambda r: httpx.Response(500, text=f"boom {KEY}"), LLMProviderError),
        (lambda r: httpx.Response(401, text="bad key"), LLMProviderError),
        (
            lambda r: httpx.Response(302, headers={"location": "https://evil.example/"}),
            LLMProviderError,
        ),
        (lambda r: httpx.Response(200, text="not json"), LLMProviderError),
        (lambda r: httpx.Response(200, json={"choices": []}), LLMProviderError),
        (
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
            LLMProviderError,
        ),
    ],
)
def test_failures_become_fixed_codes_without_provider_text(handler, error):
    with pytest.raises(error) as info:
        client_for(handler).generate(REQUEST)
    assert KEY not in str(info.value) and "boom" not in str(info.value)


def test_timeouts_and_network_errors_are_mapped():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def broken(request):
        raise httpx.ConnectError(f"cannot reach {KEY}", request=request)

    with pytest.raises(LLMTimeout):
        client_for(timeout).generate(REQUEST)
    with pytest.raises(LLMProviderError) as info:
        client_for(broken).generate(REQUEST)
    assert KEY not in str(info.value) and info.value.__cause__ is None


def test_a_client_is_built_only_when_fully_configured():
    base = dict(
        _env_file=None, rag_enabled=True, llm_base_url="https://h.invalid/v1", llm_model="m"
    )
    assert build_llm_client(Settings(**base)) is None  # no key
    assert build_llm_client(Settings(_env_file=None, llm_api_key="k")) is None  # disabled
    built = build_llm_client(Settings(**base, llm_api_key="k"))
    assert built is not None and built.model == "m" and built.provider == "openai-compatible"


def test_json_mode_can_be_switched_off_for_servers_that_reject_it():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok("{}")

    on = OpenAICompatibleClient(
        "https://a.invalid/v1", KEY, "m", transport=httpx.MockTransport(handler)
    )
    off = OpenAICompatibleClient(
        "https://a.invalid/v1", KEY, "m", json_mode=False, transport=httpx.MockTransport(handler)
    )
    on.generate(REQUEST)
    off.generate(REQUEST)
    assert (
        seen[0]["response_format"] == {"type": "json_object"} and "response_format" not in seen[1]
    )


def test_the_setting_reaches_the_built_client(monkeypatch):
    base = dict(
        _env_file=None, rag_enabled=True, llm_base_url="https://h.invalid/v1",
        llm_model="m", llm_api_key="k",
    )  # fmt: skip
    assert build_llm_client(Settings(**base))._json_mode is True
    assert build_llm_client(Settings(**base, llm_json_mode=False))._json_mode is False
