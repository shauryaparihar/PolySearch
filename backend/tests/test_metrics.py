import math

import pytest
from app.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank_is_one_over_the_first_relevant_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "x"], {"a", "b"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_results_beyond_k_do_not_count():
    ranked = [f"d{i}" for i in range(10)] + ["a"]  # relevant document sits at rank 11
    assert reciprocal_rank(ranked, {"a"}, k=10) == 0.0
    assert recall_at_k(ranked, {"a"}, k=10) == 0.0
    assert ndcg_at_k(ranked, {"a"}, k=10) == 0.0


def test_recall_is_the_fraction_of_relevant_documents_found():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_ndcg_matches_a_hand_calculation():
    # relevant a (rank 1) and b (rank 3): DCG = 1/log2(2) + 1/log2(4) = 1.5
    # ideal puts both first:               IDCG = 1/log2(2) + 1/log2(3)
    expected = 1.5 / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_query_without_relevant_documents_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set())
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set())
