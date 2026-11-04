"""Answer-mode settings: safe disabled defaults, secret hygiene, and strict bounds."""

from __future__ import annotations

import pytest
from app.config import Settings
from pydantic import ValidationError

SECRET = "sk-test-0123456789-DO-NOT-LEAK"


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_answers_are_off_and_unconfigured_by_default():
    s = make()
    assert s.rag_enabled is False and s.rag_configured is False
    assert s.llm_base_url is None and s.llm_api_key is None and s.llm_model is None
    assert s.rag_store_question_text is False  # raw questions are never stored by default
    assert s.rag_max_repair_attempts == 1  # at most one repair call
    assert s.llm_input_price_per_mtok is None  # no price is ever assumed


def test_configured_requires_enabled_url_key_and_model():
    full = dict(
        rag_enabled=True, llm_base_url="https://llm.example/v1", llm_api_key=SECRET, llm_model="m"
    )
    assert make(**full).rag_configured is True
    for missing in ("llm_base_url", "llm_api_key", "llm_model"):
        assert make(**{**full, missing: None}).rag_configured is False
    assert make(**{**full, "rag_enabled": False}).rag_configured is False  # key alone is not on


def test_empty_environment_strings_mean_unset(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(name, "")  # what Docker Compose passes for an unset variable
    s = make()
    assert (s.llm_base_url, s.llm_api_key, s.llm_model) == (None, None, None)


def test_the_api_key_is_never_printed():
    s = make(rag_enabled=True, llm_api_key=SECRET, llm_base_url="https://x/v1", llm_model="m")
    for rendered in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        assert SECRET not in rendered
    assert s.llm_api_key.get_secret_value() == SECRET  # still usable by the provider client


def test_a_bad_setting_never_echoes_the_key_in_its_error():
    with pytest.raises(ValidationError) as caught:
        make(llm_api_key=SECRET, llm_base_url="ftp://wrong", rag_candidate_documents=0)
    assert SECRET not in str(caught.value)


def test_base_url_must_be_http_and_loses_a_trailing_slash():
    assert make(llm_base_url="https://llm.example/v1/").llm_base_url == "https://llm.example/v1"
    with pytest.raises(ValidationError):
        make(llm_base_url="llm.example/v1")


def test_a_placeholder_key_is_refused_like_other_secrets():
    with pytest.raises(ValidationError) as caught:
        make(llm_api_key="CHANGE_ME")
    assert (
        "LLM_API_KEY" in str(caught.value)
        and "CHANGE_ME" not in str(caught.value).split("LLM_API_KEY")[0]
    )


def test_production_requires_an_https_provider():
    prod = dict(
        environment="production", cursor_secret="x" * 40, artifact_root="/tmp",
        database_url="postgresql+psycopg://u:strong-password-123@db/prod",
        cors_allowed_origins=["https://app.example"],
    )  # fmt: skip
    assert make(**prod, llm_base_url="https://llm.example/v1").llm_base_url
    with pytest.raises(ValidationError) as caught:
        make(**prod, llm_base_url="http://llm.example/v1")
    assert "https" in str(caught.value)


@pytest.mark.parametrize(
    "bad",
    [
        {"rag_max_repair_attempts": 2},
        {"generation_max_concurrency": 0},
        {"answer_rate_limit_per_minute": 0},
        {"llm_timeout_seconds": 0},
        {"llm_max_output_tokens": 10},
        {"rag_final_passages": 0},
        {"rag_daily_token_budget": -1},
        {"answer_max_question_chars": 5000},
        {"llm_input_price_per_mtok": -1.0},
    ],
)
def test_limits_are_bounded(bad):
    with pytest.raises(ValidationError):
        make(**bad)


def test_a_zero_daily_budget_is_allowed_and_means_no_calls():
    assert make(rag_daily_token_budget=0).rag_daily_token_budget == 0
