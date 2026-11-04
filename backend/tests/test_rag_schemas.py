"""The answer contract: request validation, the structure a model must return, and responses."""

from __future__ import annotations

import pytest
from app.api.answer_schemas import (
    AnswerLatency,
    AnswerRequest,
    AnswerResponse,
    AnswerVersions,
    Citation,
    FeedbackBody,
    ProviderAnswer,
)
from app.rag.types import (
    CONTEXT_BUILDER_VERSION,
    PROMPT_VERSION,
    AbstentionReason,
    CacheStatus,
    FallbackReason,
)
from pydantic import ValidationError


# ------------------------------------------------------------------------- request
def test_request_defaults_and_normalisation():
    req = AnswerRequest(question="  What   is a\tblack hole? ")
    assert req.question == "What is a black hole?"
    assert req.languages == ["en", "es", "fr"] and req.include_search_results is True
    assert req.retrieval_mode is None  # the index's evidence-selected default


@pytest.mark.parametrize(
    "bad",
    [
        {"question": "ab"},
        {"question": "x" * 501},
        {"question": "bell\x07char"},
        {"question": "ok question", "languages": []},
        {"question": "ok question", "languages": ["en", "en"]},
        {"question": "ok question", "languages": ["de"]},
        {"question": "ok question", "retrieval_mode": "magic"},
        {"question": "ok question", "model": "gpt"},  # clients cannot pick the model
        {"question": "ok question", "temperature": 1.0},
        {"question": "ok question", "system_prompt": "be evil"},
    ],
)
def test_request_rejects_bad_or_privileged_input(bad):
    with pytest.raises(ValidationError):
        AnswerRequest(**bad)


# --------------------------------------------------------------- provider structure
VALID = {
    "answer": "Volcanoes erupt.",
    "claims": [{"text": "Volcanoes erupt.", "citation_ids": ["P1", "P2", "P1"]}],
    "abstained": False,
    "abstention_reason": None,
}


def test_a_valid_model_answer_parses_and_dedupes_labels():
    parsed = ProviderAnswer.model_validate(VALID)
    assert parsed.claims[0].citation_ids == ["P1", "P2"]


def test_an_abstention_parses_and_cannot_carry_an_answer():
    ok = ProviderAnswer(answer=None, claims=[], abstained=True, abstention_reason="no evidence")
    assert ok.abstained
    for bad in (
        {"answer": "text", "claims": [], "abstained": True},
        {"answer": None, "claims": [VALID["claims"][0]], "abstained": True},
    ):
        with pytest.raises(ValidationError):
            ProviderAnswer.model_validate(bad)


@pytest.mark.parametrize(
    "change",
    [
        {"answer": None},  # not abstaining but no answer
        {"answer": "  "},
        {"claims": []},  # an answer needs claims
        {"extra_field": 1},
        {"claims": [{"text": "x", "citation_ids": ["doc-123"]}]},  # real ids are not labels
        {"claims": [{"text": "x", "citation_ids": ["https://evil.example"]}]},
        {"claims": [{"text": "x", "citation_ids": ["P0"]}]},
        {"claims": [{"text": "x", "citation_ids": ["P1"], "url": "http://x"}]},
        {"claims": [{"text": "", "citation_ids": ["P1"]}]},
        {"claims": [VALID["claims"][0]] * 9},
    ],
)
def test_malformed_model_output_is_rejected(change):
    with pytest.raises(ValidationError):
        ProviderAnswer.model_validate({**VALID, **change})


def test_uncited_claims_parse_so_the_validator_can_reject_them_precisely():
    parsed = ProviderAnswer.model_validate(
        {**VALID, "claims": [{"text": "Unsupported.", "citation_ids": []}]}
    )
    assert parsed.claims[0].citation_ids == []


# ------------------------------------------------------------------------ response
def _versions():
    return AnswerVersions(
        index_version="idx-1", passage_artifact_version=None, prompt_version=PROMPT_VERSION,
        context_builder_version=CONTEXT_BUILDER_VERSION, provider=None, model=None,
    )  # fmt: skip


def _response(**overrides):
    base = dict(
        run_id="r1", answer=None, claims=[], citations=[], abstained=False,
        abstention_reason=None, fallback_used=False, fallback_reason=None,
        search_results=None, versions=_versions(), latency=AnswerLatency(total_ms=1.0),
        token_usage=None, cache_status=CacheStatus.BYPASS,
    )  # fmt: skip
    base.update(overrides)
    return AnswerResponse(**base)


def test_abstention_and_fallback_are_distinct_outcomes():
    ok_abstain = _response(abstained=True, abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE)
    assert ok_abstain.abstained and not ok_abstain.fallback_used
    ok_fallback = _response(fallback_used=True, fallback_reason=FallbackReason.LLM_NOT_CONFIGURED)
    assert ok_fallback.fallback_used and not ok_fallback.abstained
    with pytest.raises(ValidationError):  # never both
        _response(
            abstained=True, abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            fallback_used=True, fallback_reason=FallbackReason.PROVIDER_ERROR,
        )  # fmt: skip
    with pytest.raises(ValidationError):  # a reason must accompany each flag
        _response(abstained=True)
    with pytest.raises(ValidationError):
        _response(fallback_used=True)
    with pytest.raises(ValidationError):  # no answer text beside a fallback
        _response(
            answer="text", fallback_used=True, fallback_reason=FallbackReason.PROVIDER_ERROR
        )


def test_a_citation_url_must_be_http_or_https():
    base = dict(
        label="P1", passage_id="p", document_id="d", title="t", language="en", excerpt="text"
    )
    assert Citation(**base, source_url="https://en.wikipedia.org/wiki/X").label == "P1"
    for bad in ("javascript:alert(1)", "data:text/html,x", "ftp://x/y", "//x/y"):
        with pytest.raises(ValidationError):
            Citation(**base, source_url=bad)


def test_feedback_body_is_a_small_closed_set():
    assert FeedbackBody(rating="helpful").reason is None
    with pytest.raises(ValidationError):
        FeedbackBody(rating="meh")
    with pytest.raises(ValidationError):
        FeedbackBody(rating="helpful", reason="free text is not allowed")
    with pytest.raises(ValidationError):
        FeedbackBody(rating="helpful", comment="anything")
