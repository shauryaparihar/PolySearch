"""Search must behave exactly as before when no answer-mode setting exists at all."""

from __future__ import annotations

import pytest
from api_support import SEARCH
from app.api.logging_config import ALLOWED_FIELDS, log_event

LLM_VARIABLES = (
    "RAG_ENABLED", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_OUTPUT_TOKENS", "RAG_DAILY_TOKEN_BUDGET",
)  # fmt: skip


@pytest.fixture
def no_llm_environment(monkeypatch):
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_search_stats_and_readiness_work_without_any_llm_setting(make_client, no_llm_environment):
    client = make_client()
    body = client.post("/api/v1/search", json=SEARCH).json()
    assert body["results"] and body["mode"] in ("dense", "lexical", "hybrid")
    assert client.get("/api/v1/stats").status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert set(ready.json()["checks"]) == {
        "database", "schema", "active_index", "artifacts", "encoder", "suggester",
        "document_mappings",
    }  # answer mode adds no readiness check, so it can never make search unready


def test_the_existing_routes_are_unchanged_and_no_answer_route_exists_yet(
    make_client, no_llm_environment
):
    client = make_client()
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/api/v1/search", "/api/v1/suggestions", "/healthz", "/readyz"} <= paths
    assert "/api/v1/answers" not in paths  # R0 adds the contract only, not the endpoint


def test_turning_answers_on_without_a_provider_does_not_affect_search(make_client):
    client = make_client(rag_enabled=True)  # enabled but no URL, key or model
    assert client.app.state.polysearch.settings.rag_configured is False
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200


def test_answer_log_fields_are_identifiers_and_numbers_only():
    for forbidden in ("question", "prompt", "answer", "passage_text", "api_key", "response"):
        assert forbidden not in ALLOWED_FIELDS
    assert {"run_id", "question_sha256", "fallback_reason", "total_tokens"} <= ALLOWED_FIELDS
    log_event("answer_completed", run_id="r1", rag_status="fallback", fallback_reason="X")
    with pytest.raises(ValueError):
        log_event("answer_completed", question="what is a volcano")  # raw text is refused
