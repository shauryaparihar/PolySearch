# ruff: noqa: E501
"""Run the frozen answer benchmark under four configurations and summarise it.

Configurations
    search_only   existing search, document-level evidence only, no generation
    no_retrieval  the model sees the bare question and *no* corpus text (comparison only)
    dense_rag     passage retrieval from dense candidates, then one bounded generation
    hybrid_rag    the same with hybrid candidates

The RAG path reuses the production building blocks (``retrieve_evidence``, ``build_context``,
the versioned prompt, ``parse_provider_output`` / ``validate_answer`` and the one repair
attempt). It does not go through HTTP, PostgreSQL, the token budget or the cache, so latency
here is local orchestration, not a service measurement. Every report says whether the
generation came from a deterministic mock or a live provider; mock numbers say nothing about
how a real model behaves.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.evaluation import rag_metrics as m
from app.evaluation.rag_benchmark import Benchmark, Case
from app.rag.context import REPAIR_SUFFIX, SYSTEM_PROMPT, build_context
from app.rag.llm import LLMClient, LLMError, LLMRequest, LLMUsage
from app.rag.retrieval import EvidencePassage, RetrievalConfig, retrieve_evidence
from app.rag.types import CONTEXT_BUILDER_VERSION, PROMPT_VERSION, FallbackReason
from app.rag.validation import AnswerRejected, parse_provider_output, validate_answer
from app.search.bundle import SearchBundle

CONFIGS = ("search_only", "no_retrieval", "dense_rag", "hybrid_rag")
GENERATION_CONFIGS = ("no_retrieval", "dense_rag", "hybrid_rag")
RETRIEVAL_MODE = {"dense_rag": "dense", "hybrid_rag": "hybrid"}
LANGUAGES = ["en", "es", "fr"]
NO_RETRIEVAL_SYSTEM = (
    "Answer the question briefly and factually from your own knowledge. "
    "If you do not know, say so. Plain text only."
)
SYNTHETIC = "synthetic:injected"


@dataclass
class CaseResult:
    case_id: str
    category: str
    config: str
    status: str  # answered | abstained | fallback | failed | retrieved (search_only)
    abstention_reason: str | None = None
    fallback_reason: str | None = None
    error: str | None = None  # an exception class name only
    answer_text: str | None = None
    claims: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # doc_key, rank, score (no text)
    cited: list[dict] = field(default_factory=list)  # doc_key
    doc_recall_at_5: float | None = None
    doc_recall_at_10: float | None = None
    passage_recall_at_3: float | None = None
    passage_recall_at_k: float | None = None
    passage_mrr: float | None = None
    key_facts_found: int | None = None
    key_facts_total: int | None = None
    correct: bool | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    supported_claim_rate: float | None = None
    forbidden_hit: bool | None = None
    retrieval_ms: float | None = None
    provider_ms: float | None = None
    total_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_calls: int = 0
    # filled in by people / an optional judge, never by this runner
    human_correct: str | None = None
    human_grounded: str | None = None
    human_notes: str | None = None
    judge_model: str | None = None
    judge_score: str | None = None


def _gold(case: Case):
    return case.gold_evidence


def _to_retrieved(passages) -> list[m.RetrievedPassage]:
    return [m.RetrievedPassage(p.doc_key, p.text) for p in passages]


def _inject(case: Case, evidence: list[EvidencePassage]) -> list[EvidencePassage]:
    """Hostile synthetic passages go first, as the worst case for the prompt. They carry score 0
    so they cannot change the evidence-strength gate, and they are never gold evidence."""
    fake = [
        EvidencePassage(
            rank=0, passage_id=f"synthetic-{i}", doc_key=SYNTHETIC, language=case.question_language,
            title=inj.title, ordinal=0, text=inj.text, char_start=0, char_end=len(inj.text),
            estimated_tokens=max(1, len(inj.text) // 3), score=0.0, document_rank=0,
        )
        for i, inj in enumerate(case.injected_passages)
    ]  # fmt: skip
    merged = fake + list(evidence)
    return [
        EvidencePassage(**{**asdict(p), "rank": rank}) for rank, p in enumerate(merged, start=1)
    ]


def _retrieval_metrics(result: CaseResult, case: Case, ranked: list[m.RetrievedPassage], k: int):
    result.passage_recall_at_3 = m.passage_recall_at_k(_gold(case), ranked, 3)
    result.passage_recall_at_k = m.passage_recall_at_k(_gold(case), ranked, k)
    result.passage_mrr = m.passage_reciprocal_rank(_gold(case), ranked)


def _score_answer(result: CaseResult, case: Case, ranked_cited: list[m.RetrievedPassage]) -> None:
    text = result.answer_text or ""
    if case.answerable and result.status == "answered":
        found, total = m.key_fact_hits(text, case.key_facts)
        result.key_facts_found, result.key_facts_total = found, total
        result.correct = found == total
        result.citation_precision = m.citation_precision(ranked_cited, _gold(case))
        result.citation_recall = m.citation_recall(ranked_cited, _gold(case))
    if case.forbidden_strings:
        result.forbidden_hit = m.contains_forbidden(text, case.forbidden_strings)


def _sum_usage(usages: list[LLMUsage | None]) -> tuple[int | None, int | None]:
    reported = [u for u in usages if u is not None]
    if not reported:
        return None, None
    return sum(u.input_tokens for u in reported), sum(u.output_tokens for u in reported)


def run_search_only(bundle: SearchBundle, case: Case, mode: str) -> CaseResult:
    result = CaseResult(case.id, case.category, "search_only", "retrieved")
    started = time.perf_counter()
    hits = bundle.search(case.question, mode, LANGUAGES, k=10, pool=10).hits
    result.retrieval_ms = (time.perf_counter() - started) * 1000
    keys = [h.key for h in hits]
    result.retrieved = [{"doc_key": k, "rank": i} for i, k in enumerate(keys, start=1)]
    result.doc_recall_at_5 = m.document_recall_at_k(_gold(case), keys, 5)
    result.doc_recall_at_10 = m.document_recall_at_k(_gold(case), keys, 10)
    result.total_ms = result.retrieval_ms
    return result


def run_no_retrieval(llm: LLMClient, settings: Settings, case: Case) -> CaseResult:
    """The model gets only the question. A test proves no corpus text is in the request."""
    result = CaseResult(case.id, case.category, "no_retrieval", "answered")
    started = time.perf_counter()
    try:
        response = llm.generate(
            LLMRequest(
                system=NO_RETRIEVAL_SYSTEM, user=case.question,
                max_output_tokens=settings.llm_max_output_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        )  # fmt: skip
    except LLMError as exc:
        result.status, result.fallback_reason = "fallback", exc.reason.value
        result.provider_calls = 1
    else:
        result.provider_calls = 1
        result.answer_text = response.text.strip()[:1500]
        result.input_tokens, result.output_tokens = _sum_usage([response.usage])
        _score_answer(result, case, [])
        result.citation_precision = result.citation_recall = None  # nothing to cite
    result.total_ms = result.provider_ms = (time.perf_counter() - started) * 1000
    return result


def run_rag(
    bundle: SearchBundle, store, llm: LLMClient, settings: Settings, case: Case, config: str
) -> CaseResult:
    result = CaseResult(case.id, case.category, config, "fallback")
    started = time.perf_counter()
    mode = RETRIEVAL_MODE[config]
    retrieval_config = RetrievalConfig.from_settings(settings)
    evidence = retrieve_evidence(bundle, store, case.question, mode, LANGUAGES, retrieval_config)
    result.retrieval_ms = (time.perf_counter() - started) * 1000
    real = list(evidence.passages)
    _retrieval_metrics(result, case, _to_retrieved(real), retrieval_config.final_passages)
    best = max((p.score for p in real), default=None)
    shown = _inject(case, real) if case.injected_passages else real
    result.retrieved = [
        {"doc_key": p.doc_key, "rank": p.rank, "score": round(p.score, 4)} for p in shown
    ]
    context = build_context(
        case.question, shown,
        max_passage_tokens=settings.rag_max_passage_tokens,
        max_context_tokens=settings.rag_max_context_tokens,
    )  # fmt: skip
    if not context.items or best is None or best < settings.rag_min_evidence_score:
        result.status, result.abstention_reason = "abstained", "INSUFFICIENT_EVIDENCE"
        result.total_ms = (time.perf_counter() - started) * 1000
        return result

    usages: list[LLMUsage | None] = []
    user, repairs, validated, failure = (
        context.user_message,
        settings.rag_max_repair_attempts,
        None,
        None,
    )
    provider_ms = 0.0
    while True:
        call = time.perf_counter()
        try:
            response = llm.generate(
                LLMRequest(
                    system=SYSTEM_PROMPT, user=user,
                    max_output_tokens=settings.llm_max_output_tokens,
                    timeout_seconds=settings.llm_timeout_seconds,
                )
            )  # fmt: skip
        except LLMError as exc:
            provider_ms += (time.perf_counter() - call) * 1000
            result.provider_calls += 1
            failure = exc.reason
            break
        provider_ms += (time.perf_counter() - call) * 1000
        result.provider_calls += 1
        usages.append(response.usage)
        try:
            validated = validate_answer(parse_provider_output(response.text), context)
            break
        except AnswerRejected as exc:
            failure = exc.reason
            if exc.reason is FallbackReason.INVALID_OUTPUT and repairs > 0:
                repairs -= 1
                user = context.user_message + REPAIR_SUFFIX
                continue
            break
    result.provider_ms = provider_ms
    result.input_tokens, result.output_tokens = _sum_usage(usages)
    if validated is None:
        result.fallback_reason = failure.value if failure else None
    elif validated.abstained:
        result.status, result.abstention_reason = "abstained", "MODEL_ABSTAINED"
    else:
        result.status = "answered"
        result.answer_text = validated.answer
        by_label = {i.label: i for i in context.items}
        claims_view, cited_all = [], {}
        for claim in validated.claims:
            texts = [by_label[label].passage.text for label in claim.labels]
            claims_view.append((claim.text, texts))
            for label in claim.labels:
                cited_all[label] = by_label[label].passage
        result.claims = [
            {"text": t, "labels": c.labels}
            for (t, _), c in zip(claims_view, validated.claims, strict=True)
        ]
        cited_passages = [m.RetrievedPassage(p.doc_key, p.text) for p in cited_all.values()]
        result.cited = [{"doc_key": p.doc_key} for p in cited_all.values()]
        result.supported_claim_rate = m.supported_claim_rate(claims_view)
        _score_answer(result, case, cited_passages)
        if case.forbidden_strings:
            result.forbidden_hit = m.contains_forbidden(
                result.answer_text + " " + " ".join(c["text"] for c in result.claims),
                case.forbidden_strings,
            )
    result.total_ms = (time.perf_counter() - started) * 1000
    return result


# =========================================================================== orchestration
def run_evaluation(
    bundle: SearchBundle,
    store,
    benchmark: Benchmark,
    llm: LLMClient,
    settings: Settings,
    configs: tuple[str, ...] = CONFIGS,
    default_mode: str = "dense",
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for config in configs:
        for case in benchmark.cases:
            try:
                if config == "search_only":
                    results.append(run_search_only(bundle, case, default_mode))
                elif config == "no_retrieval":
                    results.append(run_no_retrieval(llm, settings, case))
                else:
                    results.append(run_rag(bundle, store, llm, settings, case, config))
            except Exception as exc:  # noqa: BLE001  one broken case must not hide the others
                results.append(
                    CaseResult(case.id, case.category, config, "failed", error=type(exc).__name__)
                )
    return results


def summarise(
    results: list[CaseResult], benchmark: Benchmark, configs: tuple[str, ...], settings: Settings
) -> dict:
    """Per-configuration aggregates. Failed and missing cases are counted, never dropped."""
    cases = {c.id: c for c in benchmark.cases}
    out: dict[str, dict] = {}
    for config in configs:
        mine = [r for r in results if r.config == config]
        seen = {r.case_id for r in mine}
        completed = [r for r in mine if r.status != "failed"]
        summary: dict = {
            "cases_expected": len(cases),
            "cases_completed": len(completed),
            "cases_failed": sum(r.status == "failed" for r in mine),
            "cases_missing": len(set(cases) - seen),
        }
        with_gold = [r for r in completed if cases[r.case_id].gold_evidence]
        if config == "search_only":
            summary["document_recall_at_5"] = m.mean(r.doc_recall_at_5 for r in with_gold)
            summary["document_recall_at_10"] = m.mean(r.doc_recall_at_10 for r in with_gold)
        else:
            summary["passage_recall_at_3"] = m.mean(r.passage_recall_at_3 for r in with_gold)
            summary[f"passage_recall_at_{settings.rag_final_passages}"] = m.mean(
                r.passage_recall_at_k for r in with_gold
            )
            summary["passage_mrr"] = m.mean(r.passage_mrr for r in with_gold)
        summary["recall_by_category"] = {}
        key = "doc_recall_at_10" if config == "search_only" else "passage_recall_at_k"
        for category in sorted({cases[r.case_id].category for r in with_gold}):
            summary["recall_by_category"][category] = m.mean(
                getattr(r, key) for r in with_gold if cases[r.case_id].category == category
            )
        if config in GENERATION_CONFIGS:
            summary |= _generation_summary(completed, cases, config, settings)
        times = [r.total_ms for r in completed if r.total_ms is not None]
        summary["latency_ms"] = {"median": m.percentile(times, 50), "p95": m.percentile(times, 95)}
        out[config] = summary
    return out


def _generation_summary(
    completed: list[CaseResult], cases: dict[str, Case], config: str, settings
) -> dict:
    answerable = [r for r in completed if cases[r.case_id].answerable]
    answered = [r for r in answerable if r.status == "answered"]
    status_counts: dict[str, int] = {}
    for r in completed:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    matrix = m.abstention_matrix((cases[r.case_id].answerable, r.status) for r in completed)
    invalid = sum((r.fallback_reason or "").startswith("INVALID") for r in completed)
    tokens_in = sum(r.input_tokens or 0 for r in completed)
    tokens_out = sum(r.output_tokens or 0 for r in completed)
    successes = [r for r in completed if r.status == "answered"]
    cost = None
    if (
        settings.llm_input_price_per_mtok is not None
        and settings.llm_output_price_per_mtok is not None
        and successes
    ):
        total = (
            tokens_in * settings.llm_input_price_per_mtok
            + tokens_out * settings.llm_output_price_per_mtok
        ) / 1e6
        cost = total / len(successes)
    adversarial = [r for r in completed if cases[r.case_id].forbidden_strings]
    return {
        "status_counts": status_counts,
        "abstention": {
            "tp": matrix.tp,
            "fp": matrix.fp,
            "fn": matrix.fn,
            "tn": matrix.tn,
            "fallbacks_excluded": matrix.fallbacks,
            "precision": matrix.precision,
            "recall": matrix.recall,
        },  # fmt: skip
        "fallback_rate": m.rate(sum(r.status == "fallback" for r in completed), len(completed)),
        "invalid_output_rate": m.rate(invalid, len(completed)),
        "correct_when_answered": m.rate(sum(bool(r.correct) for r in answered), len(answered)),
        "correct_end_to_end": m.rate(sum(bool(r.correct) for r in answerable), len(answerable)),
        "key_fact_recall": m.mean(
            (r.key_facts_found / r.key_facts_total) if r.key_facts_total else None for r in answered
        ),
        "citation_precision": m.mean(r.citation_precision for r in answered),
        "citation_recall": m.mean(r.citation_recall for r in answered),
        "supported_claim_rate": m.mean(r.supported_claim_rate for r in answered),
        "supported_claim_note": "lexical overlap proxy, not an entailment check",
        "adversarial_cases": len(adversarial),
        "adversarial_forbidden_text_leaked": sum(bool(r.forbidden_hit) for r in adversarial),
        "tokens": {"input": tokens_in, "output": tokens_out, "total": tokens_in + tokens_out},
        "estimated_cost_per_successful_answer": cost,
        "provider_calls": sum(r.provider_calls for r in completed),
    }


# ============================================================================== metadata
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def collect_metadata(
    bundle: SearchBundle, store, benchmark: Benchmark, llm: LLMClient, settings: Settings,
    mode: str, default_mode: str, now: datetime | None = None,
) -> dict:  # fmt: skip
    return {
        "mode": mode,  # "mock" or "live"
        "generation_note": (
            "MOCK: answers come from a deterministic extractive stand-in, not a language model"
            if mode == "mock"
            else "LIVE: answers come from the configured provider"
        ),
        "timestamp_utc": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "benchmark": {
            "version": benchmark.version,
            "sha256": benchmark.sha256,
            "cases": len(benchmark.cases),
            "by_category": benchmark.by_category(),
        },  # fmt: skip
        "index_version": bundle.version,
        "corpus_sha256": bundle.manifest["corpus"]["sha256"],
        "passage_artifact": store.version if store is not None else None,
        "encoder": bundle.manifest["encoder"]["name"],
        "provider": getattr(llm, "provider", None),
        "model": getattr(llm, "model", None),
        "prompt_version": PROMPT_VERSION,
        "context_builder_version": CONTEXT_BUILDER_VERSION,
        "retrieval": {
            "search_only_mode": default_mode,
            "candidate_documents": settings.rag_candidate_documents,
            "final_passages": settings.rag_final_passages,
            "max_passages_per_document": settings.rag_max_passages_per_document,
            "min_evidence_score": settings.rag_min_evidence_score,
            "lexical_rescue": RetrievalConfig().lexical_rescue,
        },
        "generation": {
            "max_output_tokens": settings.llm_max_output_tokens,
            "temperature": 0,
            "json_mode": settings.llm_json_mode,
            "max_repair_attempts": settings.rag_max_repair_attempts,
            "timeout_seconds": settings.llm_timeout_seconds,
        },
        "environment": f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}",
        "latency_note": "local orchestration only: no HTTP, database, budget or cache involved",
    }


# ================================================================================ writers
def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(metadata: dict, summary: dict) -> str:
    lines = [
        f"# PolySearch answer-mode evaluation ({metadata['benchmark']['version']})",
        "",
        f"> **{metadata['generation_note']}.**",
        "> Retrieval numbers are deterministic. Generation numbers from a mock describe the "
        "pipeline, not any model.",
        "",
        "## Run",
        "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Git commit: `{metadata['git_commit']}`{' (uncommitted changes)' if metadata['git_dirty'] else ''}",
        f"- Benchmark: {metadata['benchmark']['version']}, {metadata['benchmark']['cases']} cases "
        f"{metadata['benchmark']['by_category']}, sha256 `{metadata['benchmark']['sha256'][:12]}`",
        f"- Index `{metadata['index_version']}`, passages `{metadata['passage_artifact']}`, "
        f"corpus sha256 `{metadata['corpus_sha256'][:12]}`",
        f"- Provider / model: {metadata['provider']} / {metadata['model']}",
        f"- Prompt `{metadata['prompt_version']}`, context builder `{metadata['context_builder_version']}`",
        f"- Retrieval: {metadata['retrieval']}",
        f"- Generation: {metadata['generation']}",
        f"- Environment: {metadata['environment']}",
        f"- Latency: {metadata['latency_note']}",
        "",
        "## Retrieval",
        "",
        "| Configuration | Cases ok / failed / missing | Doc R@5 | Doc R@10 | Passage R@3 | Passage R@K | Passage MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    k = metadata["retrieval"]["final_passages"]
    for config, s in summary.items():
        lines.append(
            f"| {config} | {s['cases_completed']} / {s['cases_failed']} / {s['cases_missing']} "
            f"| {_pct(s.get('document_recall_at_5'))} | {_pct(s.get('document_recall_at_10'))} "
            f"| {_pct(s.get('passage_recall_at_3'))} | {_pct(s.get(f'passage_recall_at_{k}'))} "
            f"| {_num(s.get('passage_mrr'), 3)} |"
        )
    lines += ["", "Recall by category (answerable cases with gold evidence):", ""]
    for config, s in summary.items():
        if s["recall_by_category"]:
            cells = ", ".join(f"{c} {_pct(v)}" for c, v in s["recall_by_category"].items())
            lines.append(f"- {config}: {cells}")
    gen = {c: s for c, s in summary.items() if "status_counts" in s}
    if gen:
        lines += ["", "## Generation", "", f"({metadata['generation_note']})", ""]
        lines += [
            "| Metric | " + " | ".join(gen) + " |",
            "|---|" + "---|" * len(gen),
        ]
        rows = [
            ("Answered / abstained / fallback", lambda s: " / ".join(str(s["status_counts"].get(x, 0)) for x in ("answered", "abstained", "fallback"))),
            ("Abstention precision", lambda s: _pct(s["abstention"]["precision"])),
            ("Abstention recall", lambda s: _pct(s["abstention"]["recall"])),
            ("Abstention TP / FP / FN / TN", lambda s: " / ".join(str(s["abstention"][x]) for x in ("tp", "fp", "fn", "tn"))),
            ("Correct when answered (key facts)", lambda s: _pct(s["correct_when_answered"])),
            ("Correct end to end (answerable)", lambda s: _pct(s["correct_end_to_end"])),
            ("Citation precision", lambda s: _pct(s["citation_precision"])),
            ("Citation recall (completeness)", lambda s: _pct(s["citation_recall"])),
            ("Supported claims (lexical proxy)", lambda s: _pct(s["supported_claim_rate"])),
            ("Fallback rate", lambda s: _pct(s["fallback_rate"])),
            ("Invalid-output rate", lambda s: _pct(s["invalid_output_rate"])),
            ("Adversarial leaks", lambda s: f"{s['adversarial_forbidden_text_leaked']} of {s['adversarial_cases']}"),
            ("Tokens in / out", lambda s: f"{s['tokens']['input']} / {s['tokens']['output']}"),
            ("Est. cost per answer", lambda s: _num(s["estimated_cost_per_successful_answer"], 6)),
            ("Latency median / p95 (ms)", lambda s: f"{_num(s['latency_ms']['median'])} / {_num(s['latency_ms']['p95'])}"),
        ]  # fmt: skip
        for label, fn in rows:
            lines.append(f"| {label} | " + " | ".join(fn(s) for s in gen.values()) + " |")
    lines += [
        "",
        "## Not measured here",
        "",
        "- Human review columns (`human_correct`, `human_grounded`, `human_notes`) in the CSV are "
        "blank and must be filled by a person on a small sample.",
        "- `judge_*` columns are empty: no LLM judge was run.",
        "- Key-fact correctness is a substring rule; supported-claim rate is lexical overlap. "
        "Neither proves entailment.",
        "- `no_retrieval` has no abstention measure: it receives only the question.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    out_dir: Path, metadata: dict, summary: dict, results: list[CaseResult]
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "cases.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_markdown(metadata, summary), encoding="utf-8")
    columns = list(asdict(CaseResult("x", "x", "x", "x")))
    flat = [c for c in columns if c not in ("claims", "retrieved", "cited")]
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    return [out_dir / n for n in ("summary.json", "cases.json", "summary.md", "cases.csv")]
