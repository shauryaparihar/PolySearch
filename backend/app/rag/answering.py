"""One bounded, synchronous answer request: retrieve, build context, call the model once
(plus at most one repair), verify every citation on the server, then answer, abstain or fall
back to ordinary search results.

Citation *generation* is the model's job; citation *verification* is ours. A single bounded
call with no tools, no loops and no URL fetching keeps cost, latency and the effect of hostile
evidence limited. Search itself never depends on any of this.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.api import service
from app.api.answer_schemas import (
    AnswerClaim,
    AnswerLatency,
    AnswerRequest,
    AnswerResponse,
    AnswerVersions,
    Citation,
    TokenUsage,
)
from app.api.cache import query_sha256
from app.api.errors import ApiError
from app.api.logging_config import log_event
from app.api.schemas import ResultItem, SearchRequest
from app.api.state import AppState
from app.db.models import Document, DocumentPassage, GenerationCitation, GenerationRun
from app.rag.context import REPAIR_SUFFIX, SYSTEM_PROMPT, EvidenceContext, build_context
from app.rag.llm import LLMError, LLMRequest, LLMUsage
from app.rag.passage_store import PassageArtifactError
from app.rag.retrieval import Evidence, RetrievalConfig, retrieve_evidence
from app.rag.types import (
    CONTEXT_BUILDER_VERSION,
    PROMPT_VERSION,
    AbstentionReason,
    CacheStatus,
    FallbackReason,
    RunStatus,
)
from app.rag.validation import (
    AnswerRejected,
    ValidatedAnswer,
    parse_provider_output,
    validate_answer,
)

MIN_REPAIR_SECONDS = 1.0  # do not start a repair call with less time than this left


class _Fallback(Exception):
    def __init__(self, reason: FallbackReason, error_code: str | None = None):
        super().__init__(reason.value)
        self.reason = reason
        self.error_code = error_code or reason.value


def _sum_usage(usages: list[LLMUsage | None]) -> LLMUsage | None:
    reported = [u for u in usages if u is not None]
    if not reported:
        return None
    return LLMUsage(sum(u.input_tokens for u in reported), sum(u.output_tokens for u in reported))


def _estimated_cost(settings, usage: LLMUsage | None) -> float | None:
    """From configured prices only; never a built-in vendor price."""
    if (
        usage is None
        or settings.llm_input_price_per_mtok is None
        or settings.llm_output_price_per_mtok is None
    ):
        return None
    return (
        usage.input_tokens * settings.llm_input_price_per_mtok
        + usage.output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000


def _generate(
    state: AppState, context: EvidenceContext, timings: dict, usages: list[LLMUsage | None]
) -> ValidatedAnswer:
    """Call the model, verify the result, and repair once if the *structure* was unusable."""
    settings, llm = state.settings, state.llm
    deadline = time.perf_counter() + settings.llm_timeout_seconds * 2  # whole request budget
    user = context.user_message
    repairs_left = settings.rag_max_repair_attempts
    while True:
        started = time.perf_counter()
        try:
            response = llm.generate(
                LLMRequest(
                    system=SYSTEM_PROMPT,
                    user=user,
                    max_output_tokens=settings.llm_max_output_tokens,
                    timeout_seconds=min(settings.llm_timeout_seconds, max(deadline - started, 0.1)),
                )
            )
        except LLMError as exc:
            timings["provider_ms"] += (time.perf_counter() - started) * 1000
            raise _Fallback(exc.reason) from None
        timings["provider_ms"] += (time.perf_counter() - started) * 1000
        usages.append(response.usage)
        checked = time.perf_counter()
        try:
            return validate_answer(parse_provider_output(response.text), context)
        except AnswerRejected as exc:
            reason = exc.reason
        finally:
            timings["validation_ms"] += (time.perf_counter() - checked) * 1000
        repairable = (
            reason is FallbackReason.INVALID_OUTPUT
            and repairs_left > 0
            and deadline - time.perf_counter() >= MIN_REPAIR_SECONDS
        )
        if not repairable:
            raise _Fallback(reason)
        repairs_left -= 1
        user = context.user_message + REPAIR_SUFFIX


def _label_number(label: str) -> int:
    return int(label[1:])


def _resolve_citations(
    state: AppState, context: EvidenceContext, validated: ValidatedAnswer
) -> dict[str, Citation]:
    """Build each cited citation from *stored* data (never from model output)."""
    cited = sorted({label for c in validated.claims for label in c.labels}, key=_label_number)
    items = {label: context.item(label) for label in cited}
    ids = [uuid.UUID(items[label].passage.passage_id) for label in cited]
    with state.session_factory() as session:
        rows = {
            passage.id: (passage, document)
            for passage, document in session.execute(
                select(DocumentPassage, Document)
                .join(Document, Document.id == DocumentPassage.document_id)
                .where(DocumentPassage.id.in_(ids), Document.is_active.is_(True))
            )
        }
    out: dict[str, Citation] = {}
    for label in cited:
        item = items[label]
        found = rows.get(uuid.UUID(item.passage.passage_id))
        if found is None or found[0].text != item.passage.text:
            raise _Fallback(FallbackReason.RAG_UNAVAILABLE, "CITATION_SOURCE_MISSING")
        passage, document = found
        try:
            out[label] = Citation(
                label=label,
                passage_id=str(passage.id),
                document_id=str(document.id),
                title=service.safe_text(document.title, 300),
                language=document.language,
                source_url=document.source_url,
                excerpt=service.safe_text(passage.text),
            )
        except ValueError:  # e.g. a stored URL that is not http(s)
            raise _Fallback(FallbackReason.RAG_UNAVAILABLE, "CITATION_SOURCE_INVALID") from None
    return out


def _search_results(
    state: AppState, req: AnswerRequest, mode: str | None, request_id: str
) -> list[ResultItem]:
    query_language = "en" if "en" in req.languages else req.languages[0]
    response = service.run_search(
        state,
        SearchRequest(
            query=req.question, query_language=query_language,
            target_languages=req.languages, mode=mode,
        ),
        request_id,
    )  # fmt: skip
    return response.results


def answer_question(state: AppState, req: AnswerRequest, request_id: str) -> AnswerResponse:
    started = time.perf_counter()
    settings = state.settings
    if len(req.question) > settings.answer_max_question_chars:
        raise ApiError(
            422,
            "QUESTION_INVALID",
            f"question must contain at most {settings.answer_max_question_chars} characters.",
        )
    run_id = uuid.uuid4()
    qhash = query_sha256(req.question)
    timings = {"retrieval_ms": 0.0, "provider_ms": 0.0, "validation_ms": 0.0}
    usages: list[LLMUsage | None] = []
    bundle, version_id, store = state.snapshot_with_passages()
    mode = req.retrieval_mode or (service.default_mode(bundle) if bundle else None)

    status = RunStatus.FALLBACK
    abstention: AbstentionReason | None = None
    fallback: FallbackReason | None = None
    error_code: str | None = None
    validated: ValidatedAnswer | None = None
    citations: dict[str, Citation] = {}
    evidence: Evidence | None = None
    context: EvidenceContext | None = None
    try:
        if not settings.rag_enabled:
            raise _Fallback(FallbackReason.RAG_DISABLED)
        if state.llm is None:
            raise _Fallback(FallbackReason.LLM_NOT_CONFIGURED)
        if bundle is None or store is None or store.index_version != bundle.version:
            raise _Fallback(FallbackReason.RAG_UNAVAILABLE, state.passage_error)

        stage = time.perf_counter()
        try:
            with state.guard.slot():
                evidence = retrieve_evidence(
                    bundle, store, req.question, mode, req.languages,
                    RetrievalConfig.from_settings(settings),
                )  # fmt: skip
        except PassageArtifactError:
            raise _Fallback(FallbackReason.RAG_UNAVAILABLE, "PASSAGES_INVALID") from None
        timings["retrieval_ms"] = (time.perf_counter() - stage) * 1000

        context = build_context(
            req.question, evidence.passages,
            max_passage_tokens=settings.rag_max_passage_tokens,
            max_context_tokens=settings.rag_max_context_tokens,
        )  # fmt: skip
        best = max((p.score for p in evidence.passages), default=None)
        if not context.items or best is None or best < settings.rag_min_evidence_score:
            status, abstention = RunStatus.ABSTAINED, AbstentionReason.INSUFFICIENT_EVIDENCE
        else:
            validated = _generate(state, context, timings, usages)
            if validated.abstained:
                status, abstention = RunStatus.ABSTAINED, AbstentionReason.MODEL_ABSTAINED
            else:
                checked = time.perf_counter()
                citations = _resolve_citations(state, context, validated)
                timings["validation_ms"] += (time.perf_counter() - checked) * 1000
                status = RunStatus.ANSWERED
    except _Fallback as exc:
        status, fallback, error_code = RunStatus.FALLBACK, exc.reason, exc.error_code
        validated, citations = None, {}

    usage = _sum_usage(usages)
    results: list[ResultItem] | None = None
    if status is not RunStatus.ANSWERED or req.include_search_results:
        results = _search_results(state, req, mode, request_id)
    total_ms = (time.perf_counter() - started) * 1000

    answered = status is RunStatus.ANSWERED and validated is not None
    claims = validated.claims if answered else []
    cited_labels = sorted({label for c in claims for label in c.labels}, key=_label_number)
    provider = getattr(state.llm, "provider", None)
    model = getattr(state.llm, "model", None)
    artifact_version = store.index_version if store else None
    with state.session_factory() as session:
        session.add(
            GenerationRun(
                id=run_id,
                question_sha256=qhash,
                question_text=req.question if settings.rag_store_question_text else None,
                index_version_id=version_id,
                passage_artifact_version=artifact_version,
                retrieval_mode=mode,
                languages=sorted(req.languages),
                provider=provider,
                model=model,
                prompt_version=PROMPT_VERSION,
                context_builder_version=CONTEXT_BUILDER_VERSION,
                status=status.value,
                abstained=status is RunStatus.ABSTAINED,
                fallback_used=status is RunStatus.FALLBACK,
                abstention_reason=abstention.value if abstention else None,
                fallback_reason=fallback.value if fallback else None,
                candidate_count=evidence.candidate_count if evidence else 0,
                passage_count=len(context.items) if context else 0,
                claim_count=len(claims),
                citation_count=len(cited_labels),
                retrieval_ms=round(timings["retrieval_ms"], 3),
                provider_ms=round(timings["provider_ms"], 3),
                validation_ms=round(timings["validation_ms"], 3),
                total_ms=round(total_ms, 3),
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                estimated_cost=_estimated_cost(settings, usage),
                cache_status=CacheStatus.BYPASS.value,
                error_code=error_code,
                completed_at=datetime.now(UTC),
            )
        )
        session.flush()  # the run row must exist before its citations reference it
        for claim in claims:
            for label in claim.labels:
                item = context.item(label)
                session.add(
                    GenerationCitation(
                        generation_run_id=run_id,
                        passage_id=uuid.UUID(item.passage.passage_id),
                        label=label,
                        claim_ordinal=claim.ordinal,
                        retrieval_rank=item.passage.rank,
                        retrieval_score=item.passage.score,
                    )
                )
        session.commit()

    log_event(
        "answer_completed",
        request_id=request_id, run_id=str(run_id), route="/api/v1/answers",
        question_sha256=qhash[:16], rag_status=status.value,
        abstention_reason=abstention.value if abstention else None,
        fallback_reason=fallback.value if fallback else None,
        error_code=error_code, mode=mode,
        index_version=bundle.version if bundle else None,
        passage_artifact_version=artifact_version,
        prompt_version=PROMPT_VERSION, context_builder_version=CONTEXT_BUILDER_VERSION,
        provider=provider, model=model,
        candidate_count=evidence.candidate_count if evidence else 0,
        passage_count=len(context.items) if context else 0,
        claim_count=len(claims), citation_count=len(cited_labels),
        retrieval_ms=round(timings["retrieval_ms"], 3),
        provider_ms=round(timings["provider_ms"], 3),
        validation_ms=round(timings["validation_ms"], 3), total_ms=round(total_ms, 3),
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        cache_status=CacheStatus.BYPASS.value,
    )  # fmt: skip
    return AnswerResponse(
        run_id=str(run_id),
        answer=validated.answer if answered else None,
        claims=[AnswerClaim(text=c.text, citation_labels=c.labels) for c in claims],
        citations=[citations[label] for label in cited_labels],
        abstained=status is RunStatus.ABSTAINED,
        abstention_reason=abstention,
        fallback_used=status is RunStatus.FALLBACK,
        fallback_reason=fallback,
        search_results=results,
        versions=AnswerVersions(
            index_version=bundle.version if bundle else None,
            passage_artifact_version=artifact_version,
            prompt_version=PROMPT_VERSION,
            context_builder_version=CONTEXT_BUILDER_VERSION,
            provider=provider,
            model=model,
        ),
        latency=AnswerLatency(
            retrieval_ms=round(timings["retrieval_ms"], 3),
            provider_ms=round(timings["provider_ms"], 3) if usages else None,
            validation_ms=round(timings["validation_ms"], 3),
            total_ms=round(total_ms, 3),
        ),
        token_usage=(
            TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None
        ),
        cache_status=CacheStatus.BYPASS,
    )
