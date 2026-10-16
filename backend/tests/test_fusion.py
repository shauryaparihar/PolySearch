import pytest
from app.search.fusion import RRF_K, rrf_fuse


def test_rrf_matches_a_hand_calculation():
    # dense:   a=1 b=2 c=3      lexical: c=1 b=2 d=3         k = 60
    # a = 1/61                           = 0.016393
    # b = 1/62 + 1/62                    = 0.032258
    # c = 1/63 + 1/61                    = 0.032266
    # d = 1/63                           = 0.015873
    fused = rrf_fuse({"dense": ["a", "b", "c"], "lexical": ["c", "b", "d"]}, k=60)
    assert [key for key, _, _ in fused] == ["c", "b", "a", "d"]
    scores = {key: score for key, score, _ in fused}
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["a"] == pytest.approx(1 / 61)
    ranks = {key: r for key, _, r in fused}
    assert ranks["a"] == {"dense": 1} and ranks["c"] == {"dense": 3, "lexical": 1}


def test_ties_are_broken_by_key_and_default_k_is_sixty():
    fused = rrf_fuse({"dense": ["b", "a"], "lexical": ["a", "b"]})
    assert [key for key, _, _ in fused] == ["a", "b"]  # equal scores -> alphabetical
    assert RRF_K == 60


def test_a_document_missing_from_one_method_still_counts_for_the_other():
    fused = rrf_fuse({"dense": ["x"], "lexical": []})
    assert fused == [("x", pytest.approx(1 / 61), {"dense": 1})]
