import pytest
from app.evidence import (
    BenchmarkValidationError,
    result_problems,
    superiority_conclusion,
    validate_qrels_in_corpus,
)


def _result(mrr=0.5, recall=0.6, ndcg=0.55, queries=100):
    return {"mrr": mrr, "recall@10": recall, "ndcg@10": ndcg, "total_queries": queries}


def test_no_conclusion_with_a_single_valid_method():
    statement, notes = superiority_conclusion({"dense": _result()})
    assert statement is None
    assert any("at least 2" in n for n in notes)


def test_no_conclusion_when_the_other_method_is_all_zero():
    # Mirrors the legacy artifact: one all-zero result must not count as a comparison.
    results = {"dense": _result(), "legacy": _result(0.0, 0.0, 0.0, 500)}
    statement, notes = superiority_conclusion(results)
    assert statement is None
    assert any("all metrics are zero" in n for n in notes)


def test_conclusion_is_limited_and_names_the_leader_with_two_valid_methods():
    results = {"lexical": _result(0.3, 0.4, 0.35), "dense": _result(0.6, 0.7, 0.65)}
    statement, _ = superiority_conclusion(results)
    assert statement is not None
    assert "dense" in statement and "not a claim about real-user relevance" in statement


@pytest.mark.parametrize("bad", [{"total_queries": 0}, _result(mrr=1.5), _result(mrr=float("nan"))])
def test_invalid_results_are_flagged(bad):
    merged = {**_result(), **bad}
    assert result_problems(merged)


def test_qrels_pointing_outside_the_corpus_are_rejected():
    qrels = [
        {"query_id": "q1", "relevant_doc_id": "en_1"},
        {"query_id": "q2", "relevant_doc_id": "es_99"},
    ]
    with pytest.raises(BenchmarkValidationError, match="1 qrels"):
        validate_qrels_in_corpus(qrels, ["en_1", "fr_2"])


def test_qrels_inside_the_corpus_pass():
    validate_qrels_in_corpus([{"query_id": "q1", "relevant_doc_id": "en_1"}], ["en_1"])


def test_result_without_optional_ndcg_is_still_valid():
    assert not result_problems({"mrr": 0.4, "recall@10": 0.5, "total_queries": 10})
