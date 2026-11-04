"""The deterministic fake provider and the provider boundary's error contract."""

from __future__ import annotations

import json

import pytest
from app.api.answer_schemas import ProviderAnswer
from app.rag.fake import MODES, FakeLLMClient
from app.rag.llm import (
    LLMClient,
    LLMError,
    LLMProviderError,
    LLMRateLimited,
    LLMRequest,
    LLMTimeout,
)
from app.rag.types import FallbackReason

REQUEST = LLMRequest(system="s", user="u", max_output_tokens=100, timeout_seconds=5.0)


def test_the_fake_satisfies_the_client_protocol():
    assert isinstance(FakeLLMClient(), LLMClient)


def test_success_returns_a_valid_structured_answer_with_usage():
    response = FakeLLMClient("success").generate(REQUEST)
    parsed = ProviderAnswer.model_validate_json(response.text)
    assert parsed.claims[0].citation_ids == ["P1"] and not parsed.abstained
    assert response.usage.total_tokens == 160 and response.model == "fake-model"


def test_abstain_malformed_and_invalid_citation_modes():
    abstained = FakeLLMClient("abstain").generate(REQUEST).text
    assert ProviderAnswer.model_validate_json(abstained).abstained
    with pytest.raises(json.JSONDecodeError):
        json.loads(FakeLLMClient("malformed").generate(REQUEST).text)
    text = FakeLLMClient("invalid_citation").generate(REQUEST).text
    assert ProviderAnswer.model_validate_json(text).claims[0].citation_ids == ["P99"]


@pytest.mark.parametrize(
    ("mode", "error", "reason"),
    [
        ("timeout", LLMTimeout, FallbackReason.PROVIDER_TIMEOUT),
        ("rate_limited", LLMRateLimited, FallbackReason.PROVIDER_RATE_LIMITED),
        ("error", LLMProviderError, FallbackReason.PROVIDER_ERROR),
    ],
)
def test_failure_modes_raise_sanitised_errors(mode, error, reason):
    with pytest.raises(error) as caught:
        FakeLLMClient(mode).generate(REQUEST)
    assert caught.value.reason is reason
    assert str(caught.value) == reason.value  # a fixed code, never provider text


def test_a_script_drives_a_repair_sequence_and_records_calls():
    client = FakeLLMClient(script=["not json", json.dumps({"answer": None, "claims": [],
                                                            "abstained": True})])  # fmt: skip
    assert client.generate(REQUEST).text == "not json"
    assert ProviderAnswer.model_validate_json(client.generate(REQUEST).text).abstained
    assert len(client.calls) == 2 and client.calls[0] is REQUEST
    scripted_error = FakeLLMClient(script=[LLMTimeout()])
    with pytest.raises(LLMError):
        scripted_error.generate(REQUEST)


def test_an_unknown_mode_is_a_programming_error():
    assert "success" in MODES
    with pytest.raises(ValueError):
        FakeLLMClient("surprise")


def test_the_fake_never_touches_the_network(monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    for mode in ("success", "abstain", "malformed", "invalid_citation"):
        FakeLLMClient(mode).generate(REQUEST)
