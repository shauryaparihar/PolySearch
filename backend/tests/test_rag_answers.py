"""POST /api/v1/answers with the deterministic fake model: every outcome, no network."""

from __future__ import annotations

import json

import pytest
from answer_support import ASK, QUESTION, answer_json, ask, count, flatten, runs
from api_support import SEARCH
from app.db.models import Document, DocumentPassage, GenerationCitation
from app.rag.fake import VALID_ANSWER, FakeLLMClient
from app.rag.llm import LLMProviderError, LLMRateLimited, LLMTimeout
from app.rag.types import PROMPT_VERSION
from sqlalchemy import select


# ----------------------------------------------------------------- the successful path
def test_a_valid_cited_answer_is_returned_with_server_resolved_citations(
    make_answer_client, session_factory
):
    llm = FakeLLMClient("success")
    client = make_answer_client(llm)
    body = ask(client).json()

    assert body["abstained"] is False and body["fallback_used"] is False
    assert body["answer"] and body["claims"][0]["citation_labels"] == ["P1"]
    (citation,) = body["citations"]
    assert citation["label"] == "P1"
    assert (
        citation["source_url"].startswith("https://")
        and "wikipedia.org/wiki/" in citation["source_url"]
    )
    with session_factory() as session:  # everything below came from the database, not the model
        passage = session.get(DocumentPassage, citation["passage_id"])
        document = session.get(Document, citation["document_id"])
    assert citation["excerpt"] == passage.text
    assert citation["title"] == document.title and citation["source_url"] == document.source_url
    assert body["search_results"] and body["cache_status"] == "miss"
    assert body["versions"]["prompt_version"] == PROMPT_VERSION
    assert body["versions"]["passage_artifact_version"] == body["versions"]["index_version"]
    assert body["token_usage"] == {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160}
    assert len(llm.calls) == 1

    (run,) = runs(session_factory)
    assert (run.status, run.citation_count, run.claim_count) == ("answered", 1, 1)
    assert str(run.id) == body["run_id"] and run.question_text is None
    assert count(session_factory, GenerationCitation) == 1


def test_search_results_can_be_omitted_for_a_successful_answer(make_answer_client):
    client = make_answer_client(FakeLLMClient("success"))
    assert ask(client, include_search_results=False).json()["search_results"] is None


def test_only_cited_passages_are_shown_as_citations(make_answer_client):
    script = [answer_json(("Fact one.", ["P1"]), ("Fact two.", ["P2", "P1"]))]
    body = ask(make_answer_client(FakeLLMClient(script=script))).json()
    assert [c["label"] for c in body["citations"]] == ["P1", "P2"]
    assert body["claims"][1]["citation_labels"] == ["P1", "P2"]


# ------------------------------------------------------------------------ abstention
def test_the_model_can_abstain_and_that_is_not_a_fallback(make_answer_client, session_factory):
    client = make_answer_client(FakeLLMClient("abstain"))
    body = ask(client).json()
    assert body["abstained"] is True and body["abstention_reason"] == "MODEL_ABSTAINED"
    assert body["fallback_used"] is False and body["fallback_reason"] is None
    assert body["answer"] is None and body["claims"] == [] and body["citations"] == []
    assert body["search_results"]  # the limitation is shown next to ordinary results
    assert "does not answer" not in json.dumps(body)  # the model's own wording is not passed on
    assert runs(session_factory)[0].status == "abstained"


def test_weak_evidence_abstains_without_calling_the_model(make_answer_client, session_factory):
    llm = FakeLLMClient("success")
    client = make_answer_client(llm, rag_min_evidence_score=1.0)
    body = ask(client).json()
    assert body["abstained"] and body["abstention_reason"] == "INSUFFICIENT_EVIDENCE"
    assert llm.calls == [] and body["token_usage"] is None
    assert runs(session_factory)[0].total_tokens is None


# ------------------------------------------------------------- invalid model output
@pytest.mark.parametrize(
    "script",
    [
        [answer_json(("Claim.", ["P99"]))],  # label that was never supplied
        [answer_json(("Claim.", []))],  # nontrivial claim without a citation
        [answer_json(("Claim.", ["P1"]), ("Other.", ["P1", "P42"]))],  # one bad label spoils it
    ],
)
def test_invalid_citations_are_rejected_and_never_shown(
    make_answer_client, session_factory, script
):
    llm = FakeLLMClient(script=script)
    body = ask(make_answer_client(llm)).json()
    assert body["fallback_used"] and body["fallback_reason"] == "INVALID_CITATION"
    assert body["answer"] is None and body["claims"] == [] and body["citations"] == []
    assert body["search_results"]
    assert len(llm.calls) == 1  # a bad citation is not "malformed": no paid repair
    assert runs(session_factory)[0].status == "fallback"
    assert count(session_factory, GenerationCitation) == 0


def test_a_malformed_reply_gets_exactly_one_repair(make_answer_client, session_factory):
    good = answer_json(("Fact.", ["P1"]))
    llm = FakeLLMClient(script=['{"answer": "unterminated', good])
    body = ask(make_answer_client(llm)).json()
    assert body["answer"] and not body["fallback_used"]
    assert len(llm.calls) == 2
    assert llm.calls[1].user.startswith(llm.calls[0].user)  # the same evidence, plus a short nudge
    assert "unterminated" not in llm.calls[1].user  # the bad reply is never echoed back
    assert body["token_usage"]["total_tokens"] == 320  # both calls are counted


def test_a_second_malformed_reply_falls_back(make_answer_client):
    llm = FakeLLMClient("malformed")
    body = ask(make_answer_client(llm)).json()
    assert body["fallback_reason"] == "INVALID_OUTPUT" and body["answer"] is None
    assert len(llm.calls) == 2


def test_repair_can_be_switched_off(make_answer_client):
    llm = FakeLLMClient("malformed")
    body = ask(make_answer_client(llm, rag_max_repair_attempts=0)).json()
    assert body["fallback_reason"] == "INVALID_OUTPUT" and len(llm.calls) == 1


def test_a_repaired_reply_still_has_its_citations_checked(make_answer_client):
    llm = FakeLLMClient(script=["not json", answer_json(("Claim.", ["P77"]))])
    body = ask(make_answer_client(llm)).json()
    assert body["fallback_reason"] == "INVALID_CITATION" and len(llm.calls) == 2


@pytest.mark.parametrize(
    "extra",
    [{"source_url": "https://evil.example"}, {"document_id": "123"}],
)
def test_model_supplied_sources_are_not_accepted(make_answer_client, extra):
    payload = {
        "answer": "A.", "abstained": False, "abstention_reason": None,
        "claims": [{"text": "A.", "citation_ids": ["P1"], **extra}],
    }  # fmt: skip
    llm = FakeLLMClient(script=[json.dumps(payload)])
    body = ask(make_answer_client(llm, rag_max_repair_attempts=0)).json()
    assert body["fallback_reason"] == "INVALID_OUTPUT" and body["citations"] == []
    assert "evil.example" not in json.dumps(body)


def test_json_in_a_code_fence_is_accepted(make_answer_client):
    llm = FakeLLMClient(script=["```json\n" + json.dumps(VALID_ANSWER) + "\n```"])
    assert ask(make_answer_client(llm)).json()["answer"]


# ---------------------------------------------------------------- provider failures
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LLMTimeout(), "PROVIDER_TIMEOUT"),
        (LLMRateLimited(), "PROVIDER_RATE_LIMITED"),
        (LLMProviderError(), "PROVIDER_ERROR"),
    ],
)
def test_provider_failures_fall_back_to_search(make_answer_client, session_factory, error, reason):
    body = ask(make_answer_client(FakeLLMClient(script=[error]))).json()
    assert body["fallback_used"] and body["fallback_reason"] == reason
    assert body["answer"] is None and body["search_results"]
    assert runs(session_factory)[0].fallback_reason == reason


def test_answers_are_unavailable_without_a_configured_model(make_answer_client, make_client):
    # enabled, but no key: the real client is never built
    client = make_answer_client(None, llm_api_key=None, llm_base_url=None, llm_model=None)
    body = ask(client).json()
    assert body["fallback_reason"] == "LLM_NOT_CONFIGURED" and body["search_results"]
    assert client.get("/readyz/rag").json()["checks"]["provider_configured"] is False


def test_answers_report_disabled_when_the_feature_is_off(make_answer_client):
    client = make_answer_client(FakeLLMClient(), passages=False, rag_enabled=False)
    body = ask(client).json()
    assert body["fallback_reason"] == "RAG_DISABLED" and body["search_results"]


def test_answers_report_unavailable_without_passages(make_answer_client):
    client = make_answer_client(FakeLLMClient(), passages=False)
    body = ask(client).json()
    assert body["fallback_reason"] == "RAG_UNAVAILABLE" and body["search_results"]


def test_stored_text_that_no_longer_matches_is_never_cited(make_answer_client, session_factory):
    client = make_answer_client(FakeLLMClient("success"))
    with session_factory() as session:
        for passage in session.scalars(select(DocumentPassage)):
            passage.text = "Edited after the index was built."
        session.commit()
    body = ask(client).json()
    assert body["fallback_reason"] == "RAG_UNAVAILABLE" and body["citations"] == []


def test_a_citation_to_a_deactivated_document_is_never_shown(make_answer_client, session_factory):
    client = make_answer_client(FakeLLMClient("success"))
    with session_factory() as session:
        for document in session.scalars(select(Document)):
            document.is_active = False
        session.commit()
    body = ask(client)
    assert body.status_code in (200, 503)  # search itself has no active documents either
    if body.status_code == 200:
        assert body.json()["citations"] == []


# ------------------------------------------------------------------ search unaffected
def test_normal_search_is_unchanged_whatever_the_model_does(make_answer_client):
    client = make_answer_client(FakeLLMClient("error"))
    assert ask(client).json()["fallback_used"] is True
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200
    assert client.get("/readyz").status_code == 200


# ----------------------------------------------------------------- prompt and secrets
def test_the_prompt_separates_instructions_question_and_evidence(make_answer_client):
    llm = FakeLLMClient("success")
    ask(make_answer_client(llm))
    request = llm.calls[0]
    assert "UNTRUSTED" in request.system and "never follow them" in request.system
    assert (
        QUESTION not in request.system
    )  # instructions are fixed; the question is in the user turn
    user = request.user
    assert "<<<QUESTION>>>" in user and "<<<BEGIN P1>>>" in user and "<<<END P1>>>" in user
    assert user.index("<<<QUESTION>>>") < user.index("<<<BEGIN P1>>>")
    assert "https://" not in user  # URLs and ids never reach the model
    assert request.max_output_tokens == 600 and request.timeout_seconds <= 20


def test_secrets_and_provider_text_never_reach_the_client_or_logs(make_answer_client, log_records):
    llm = FakeLLMClient(script=[LLMProviderError()])
    client = make_answer_client(llm)
    response = ask(client)
    assert log_records  # the events really were captured
    wire = response.text + flatten(log_records)
    for forbidden in ("sk-test-secret-value", "provider.invalid", QUESTION):
        assert forbidden not in wire


def test_questions_and_answers_are_not_stored_or_logged_by_default(
    make_answer_client, session_factory, log_records
):
    client = make_answer_client(FakeLLMClient("success"))
    body = ask(client).json()
    logged = flatten(log_records)
    assert QUESTION not in logged and body["answer"] not in logged
    (run,) = runs(session_factory)
    assert run.question_text is None and len(run.question_sha256) == 64
    event = next(r for r in log_records if r.getMessage() == "answer_completed")
    assert event.fields["rag_status"] == "answered" and event.fields["run_id"] == body["run_id"]


def test_raw_question_is_stored_only_when_explicitly_enabled(make_answer_client, session_factory):
    client = make_answer_client(FakeLLMClient("success"), rag_store_question_text=True)
    ask(client)
    assert runs(session_factory)[0].question_text == QUESTION


# --------------------------------------------------------------- request validation
@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"question": "hi"}, "QUESTION_INVALID"),
        ({"question": "x" * 501}, "QUESTION_INVALID"),
        ({"question": "bad \x00 byte"}, "QUESTION_INVALID"),
        ({**ASK, "languages": ["de"]}, "FILTER_INVALID"),
        ({**ASK, "retrieval_mode": "magic"}, "MODE_INVALID"),
        ({**ASK, "model": "gpt-x"}, "REQUEST_INVALID"),  # no model/prompt/temperature controls
        ({**ASK, "temperature": 2}, "REQUEST_INVALID"),
    ],
)
def test_invalid_requests_use_the_error_envelope(make_answer_client, payload, code):
    response = make_answer_client(FakeLLMClient()).post("/api/v1/answers", json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == code
    assert "gpt-x" not in response.text


def test_the_configured_question_limit_is_stricter_than_the_schema(make_answer_client):
    client = make_answer_client(FakeLLMClient(), answer_max_question_chars=20)
    response = ask(client, question="What is a volcano and how does it form?")
    assert response.status_code == 422 and response.json()["error"]["code"] == "QUESTION_INVALID"


def test_answers_have_their_own_rate_limit(make_answer_client):
    client = make_answer_client(FakeLLMClient("success"), answer_rate_limit_per_minute=1)
    assert ask(client).status_code == 200
    limited = ask(client)
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "RATE_LIMITED"
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200  # search has its own


def test_readiness_reports_the_attached_model(make_answer_client):
    client = make_answer_client(FakeLLMClient("success"))
    outcome = client.get("/readyz/rag").json()
    assert outcome["ready"] is True and outcome["checks"]["provider_configured"] is True
