import pytest
from app.evaluation.benchmark import load_benchmark
from app.evaluation.corpus import Corpus, CorpusDoc
from app.evaluation.runner import evaluate, render_report, write_result
from bench_helpers import qrel, query, write_benchmark


class FakeRetriever:
    name = "fake"
    config = {"kind": "scripted"}

    def __init__(self, script, fail_on=None):
        self.script, self.fail_on, self.calls = script, fail_on, []

    def search(self, query_text, target_languages, k):
        self.calls.append(list(target_languages))
        if query_text == self.fail_on:
            raise RuntimeError("boom")
        return [(key, 1.0) for key in self.script[query_text]]


@pytest.fixture
def bench(tmp_path):
    write_benchmark(
        tmp_path,
        [query("q1", "c1", "en", "alpha"), query("q2", "c2", "es", "beta")],
        [
            qrel("q1", "es:A", "es"),
            qrel("q1", "fr:A", "fr"),
            qrel("q2", "en:B", "en"),
            qrel("q2", "fr:B", "fr"),
        ],
        ["c1", "c2"],
    )
    corpus = Corpus(
        [CorpusDoc(k, k[:2], k, "s") for k in ("es:A", "fr:A", "en:B", "fr:B")], "h" * 64
    )
    return load_benchmark(tmp_path), corpus


def test_metrics_are_averaged_per_query_and_same_language_is_excluded(bench):
    benchmark, corpus = bench
    retriever = FakeRetriever({"alpha": ["es:A", "x", "fr:A"], "beta": ["z", "en:B"]})
    result = evaluate(retriever, benchmark, corpus)
    # q1: RR 1, recall 1 (2/2). q2: RR 1/2, recall 1/2.
    assert result["mrr"] == pytest.approx(0.75)
    assert result["recall@10"] == pytest.approx(0.75)
    assert retriever.calls == [["es", "fr"], ["en", "fr"]]  # never the query's own language
    assert result["per_query_language"]["es"]["queries"] == 1
    assert result["per_target_language"]["fr"]["queries"] == 2


def test_failed_queries_are_counted_and_scored_as_misses(bench):
    benchmark, corpus = bench
    retriever = FakeRetriever({"alpha": ["es:A", "fr:A"], "beta": []}, fail_on="beta")
    result = evaluate(retriever, benchmark, corpus)
    assert result["failure_count"] == 1
    assert result["mrr"] == pytest.approx(0.5)
    assert result["per_query"][1]["error"] == "RuntimeError"


def test_result_files_are_byte_identical_across_runs(bench, tmp_path):
    benchmark, corpus = bench
    script = {"alpha": ["es:A", "fr:A"], "beta": ["en:B"]}
    first = write_result(evaluate(FakeRetriever(script), benchmark, corpus), tmp_path / "a.json")
    second = write_result(evaluate(FakeRetriever(script), benchmark, corpus), tmp_path / "b.json")
    assert first == second


def _result(mrr, queries=100):
    return {
        "mrr": mrr,
        "recall@10": mrr,
        "ndcg@10": mrr,
        "total_queries": queries,
        "failure_count": 0,
        "benchmark": {
            "version": "v1",
            "queries": queries,
            "qrels": 2 * queries,
            "queries_sha256": "a" * 64,
            "qrels_sha256": "b" * 64,
        },
        "corpus": {"sha256": "c" * 64, "documents": 10, "per_language": {"en": 10}},
        "limitations": ["constructed"],
    }


def test_no_best_method_is_named_with_a_single_method():
    report = render_report({"lexical": _result(0.7)})
    assert "No comparison can be drawn" in report
    assert "highest MRR" not in report


def test_no_best_method_is_named_when_the_other_method_is_degenerate():
    report = render_report({"lexical": _result(0.7), "dense": _result(0.0)})
    assert "No comparison can be drawn" in report


def test_comparison_appears_only_with_two_valid_methods():
    report = render_report({"lexical": _result(0.4), "dense": _result(0.6)})
    assert "dense has the highest MRR@10" in report
