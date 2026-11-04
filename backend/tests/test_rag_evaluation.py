"""R4: benchmark schema, metrics, runner aggregation and reports. No network, no database."""

import csv
import json
from pathlib import Path

import pytest
from app.config import Settings
from app.evaluation import rag_metrics as m
from app.evaluation.rag_benchmark import (
    GoldItem,
    Quote,
    load_benchmark,
    validate_against_corpus,
)
from app.evaluation.rag_runner import (
    CaseResult,
    collect_metadata,
    render_markdown,
    run_evaluation,
    run_no_retrieval,
    summarise,
    write_reports,
)
from app.rag.fake import ExtractiveFakeLLM
from app.rag.llm import LLMRequest

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark/rag/v1/cases.jsonl"
P = m.RetrievedPassage


def gold(doc: str, quote: str) -> GoldItem:
    return GoldItem(any_of=[Quote(doc_key=doc, quote=quote)])


# ---------------------------------------------------------------- benchmark
def test_committed_benchmark_loads_and_is_balanced():
    bench = load_benchmark(BENCHMARK)
    assert len(bench.cases) == 30
    assert bench.by_category() == {
        "direct": 8, "cross_language": 8, "multi_evidence": 4, "unanswerable": 6, "adversarial": 4,
    }  # fmt: skip
    assert len({c.id for c in bench.cases}) == 30
    for case in bench.cases:
        assert case.answerable == bool(case.gold_evidence)


def test_unanswerable_case_with_gold_is_rejected(tmp_path):
    line = json.dumps({
        "id": "unans-01", "category": "unanswerable", "question": "What is the moon made of?",
        "question_language": "en", "answerable": False,
        "gold_evidence": [{"any_of": [{"doc_key": "en:Moon", "quote": "rocky body"}]}],
    })  # fmt: skip
    path = tmp_path / "c.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(Exception):  # noqa: B017  pydantic validation error
        load_benchmark(path)


def test_validation_finds_missing_quote_and_present_absent_string():
    bench = load_benchmark(BENCHMARK)
    problems = validate_against_corpus(bench.cases, {}, {})
    assert problems  # an empty corpus cannot satisfy any gold quote


# ------------------------------------------------------------------ metrics
def test_passage_recall_mrr_and_document_recall():
    g = [gold("en:A", "alpha beta"), gold("fr:B", "gamma delta")]
    ranked = [P("en:X", "noise"), P("en:A", "the Alpha Beta thing"), P("fr:B", "x")]
    assert m.passage_recall_at_k(g, ranked, 1) == 0.0
    assert m.passage_recall_at_k(g, ranked, 3) == 0.5  # fr:B passage lacks the quote
    assert m.passage_reciprocal_rank(g, ranked) == 0.5
    assert m.document_recall_at_k(g, ["en:X", "en:A", "fr:B"], 3) == 1.0
    assert m.document_recall_at_k(g, ["en:X", "en:A", "fr:B"], 2) == 0.5
    assert m.passage_recall_at_k([], ranked, 3) is None


def test_citation_precision_and_recall_differ():
    g = [gold("en:A", "alpha beta"), gold("fr:B", "gamma delta")]
    cited = [P("en:A", "alpha beta here"), P("en:Z", "irrelevant")]
    assert m.citation_precision(cited, g) == 0.5
    assert m.citation_recall(cited, g) == 0.5
    assert m.citation_precision([], g) is None


def test_key_facts_and_forbidden_strings_fold_case_and_accents():
    assert m.key_fact_hits("Fundada en 1998 por dos", [["1998", "1999"], ["Dos"]]) == (2, 2)
    assert m.key_fact_hits("nothing", [["x"]]) == (0, 1)
    assert m.contains_forbidden("visit EVIL.example now", ["evil.example"])
    assert not m.contains_forbidden("all good", ["evil.example"])


def test_supported_claim_rate_is_a_lexical_proxy():
    claims = [("the river flows north", ["The river flows north through town."]),
              ("cats like rockets", ["The river flows north."])]  # fmt: skip
    assert m.supported_claim_rate(claims) == 0.5


def test_abstention_matrix_excludes_fallbacks():
    outcomes = [(False, "abstained"), (True, "abstained"), (False, "answered"),
                (True, "answered"), (True, "fallback")]  # fmt: skip
    matrix = m.abstention_matrix(outcomes)
    assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn, matrix.fallbacks) == (1, 1, 1, 1, 1)
    assert matrix.precision == 0.5 and matrix.recall == 0.5


def test_percentile_mean_rate_handle_empty_input():
    assert m.percentile([], 95) is None and m.mean([None, None]) is None
    assert m.rate(1, 0) is None
    assert m.percentile([1, 2, 3, 4], 50) == 2 and m.mean([1, None, 3]) == 2


# --------------------------------------------------------------- mock model
def _prompt(sentences: str, question: str = "Where is the river?") -> LLMRequest:
    user = (
        f"[P1] title: T\n<<<BEGIN P1>>>\n{sentences}\n<<<END P1>>>\n"
        f"<<<QUESTION>>>\n{question}\n<<<END QUESTION>>>"
    )
    return LLMRequest(system="s", user=user, max_output_tokens=100, timeout_seconds=5)


def test_extractive_mock_cites_the_overlapping_sentence_and_labels_itself():
    llm = ExtractiveFakeLLM()
    assert llm.provider == "mock-extractive"
    out = json.loads(llm.generate(_prompt("Cats sleep. The river is in Spain.")).text)
    assert "river" in out["claims"][0]["text"] and out["claims"][0]["citation_ids"] == ["P1"]


def test_extractive_mock_abstains_without_overlap():
    out = json.loads(ExtractiveFakeLLM().generate(_prompt("Cats sleep.", "Where is Mars?")).text)
    assert out["abstained"] is True


# ---------------------------------------------------------------- runner
class Recorder(ExtractiveFakeLLM):
    def generate(self, request):
        return super().generate(request)


def test_no_retrieval_sends_only_the_question():
    case = load_benchmark(BENCHMARK).cases[0]
    llm = Recorder()
    result = run_no_retrieval(llm, Settings(), case)
    assert len(llm.calls) == 1
    sent = llm.calls[0].user
    assert sent == case.question
    for item in case.gold_evidence:
        for quote in item.any_of:
            assert quote.quote not in sent
    assert "<<<EVIDENCE" not in sent
    assert result.citation_recall is None and result.config == "no_retrieval"


def _results(statuses):
    bench = load_benchmark(BENCHMARK)
    rows = [
        CaseResult(c.id, c.category, "dense_rag", s)
        for c, s in zip(bench.cases, statuses, strict=False)
    ]
    return bench, rows


def test_summary_counts_failed_and_missing_cases():
    bench, rows = _results(["answered", "failed", "abstained"])
    summary = summarise(rows, bench, ("dense_rag",), Settings())["dense_rag"]
    assert summary["cases_expected"] == 30
    assert summary["cases_completed"] == 2
    assert summary["cases_failed"] == 1
    assert summary["cases_missing"] == 27


def test_one_broken_case_does_not_stop_the_run():
    bench = load_benchmark(BENCHMARK)
    bench.cases = bench.cases[:3]
    results = run_evaluation(None, None, bench, ExtractiveFakeLLM(), Settings(), ("search_only",))
    assert [r.status for r in results] == ["failed"] * 3
    assert all(r.error == "AttributeError" for r in results)


class _Bundle:
    version = "idx-test"
    manifest = {"corpus": {"sha256": "a" * 64}, "encoder": {"name": "enc"}}


class _Store:
    version = "psg-test"


def test_report_labels_mock_vs_live_and_leaves_human_columns_blank(tmp_path):
    bench, rows = _results(["answered"] * 30)
    settings = Settings()
    summary = summarise(rows, bench, ("dense_rag",), settings)
    mock = collect_metadata(
        _Bundle(), _Store(), bench, ExtractiveFakeLLM(), settings, "mock", "dense"
    )
    live = collect_metadata(
        _Bundle(), _Store(), bench, ExtractiveFakeLLM(), settings, "live", "dense"
    )
    assert "MOCK" in render_markdown(mock, summary).splitlines()[2]
    assert "LIVE" in render_markdown(live, summary).splitlines()[2]
    paths = write_reports(tmp_path, mock, summary, rows)
    assert {p.name for p in paths} == {
        "summary.json",
        "cases.json",
        "summary.md",
        "cases.csv",
        "review.md",
    }
    with (tmp_path / "cases.csv").open(encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert first["human_correct"] == "" and first["judge_score"] == ""
    assert json.loads((tmp_path / "summary.json").read_text())["metadata"]["mode"] == "mock"


def test_cli_refuses_live_run_without_approval(monkeypatch, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_rag_eval", Path(__file__).resolve().parents[1] / "scripts/run_rag_eval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "m")
    corpus = Path(__file__).resolve().parents[2] / "artifacts/corpus_v1/corpus.jsonl"
    if not corpus.exists():
        pytest.skip("local corpus not present")
    code = cli.main(["--index-dir", "x", "--corpus", str(corpus), "--llm", "live"])
    out = capsys.readouterr().out
    assert code == 3 and "NOT RUN" in out and "sk-test" not in out


def test_strict_claim_check_catches_a_swapped_number_or_name():
    cited = ["The company was founded in 1998 by Maria Lopez in Madrid."]
    assert m.strict_claim_supported("It was founded in 1998 by Maria Lopez.", cited)
    assert not m.strict_claim_supported("It was founded in 1999 by Maria Lopez.", cited)
    assert not m.strict_claim_supported("It was founded in 1998 by Carlos Ruiz.", cited)
    assert m.strict_claim_supported("it was founded long ago.", cited)  # nothing to check


def test_review_sheet_shows_claims_with_their_cited_text():
    from app.evaluation.rag_runner import render_review_sheet

    row = CaseResult("direct-01", "direct", "dense_rag", "answered")
    row.claims = [{"text": "A claim.", "labels": ["P1"], "cited_passages": ["The source text."]}]
    sheet = render_review_sheet([row])
    assert "direct-01" in sheet and "A claim." in sheet and "> The source text." in sheet
    assert "No answered cases" in render_review_sheet([])


def test_budget_reservation_estimate_is_more_cautious_than_chunk_sizing():
    from app.rag.passages import estimate_request_tokens, estimate_tokens

    text = "x" * 1000
    assert estimate_request_tokens(text) == 400 > estimate_tokens(text) == 334
    assert estimate_request_tokens("") == 0
