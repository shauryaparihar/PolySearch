import pytest
from app.search.default_mode import HYBRID_P95_BUDGET_MS, select_default_mode


def _res(mrr, ndcg):
    return {"mrr": mrr, "recall@10": mrr, "ndcg@10": ndcg, "total_queries": 100}


def _lat(p95):
    return {"hybrid": {"total": {"p95_ms": p95}}, "dense": {"total": {"p95_ms": 5.0}}}


def test_hybrid_becomes_default_only_when_it_wins_and_is_fast_enough():
    out = select_default_mode({"dense": _res(0.8, 0.8), "hybrid": _res(0.9, 0.85)}, _lat(50))
    assert out["default_mode"] == "hybrid"


def test_dense_stays_default_when_hybrid_does_not_win():
    out = select_default_mode({"dense": _res(0.94, 0.94), "hybrid": _res(0.91, 0.90)}, _lat(50))
    assert out["default_mode"] == "dense" and "does not beat dense" in out["reason"]
    tie = select_default_mode({"dense": _res(0.9, 0.9), "hybrid": _res(0.9, 0.9)}, _lat(50))
    assert tie["default_mode"] == "dense"  # no strict improvement -> keep the simpler mode


def test_dense_stays_default_when_hybrid_is_too_slow():
    out = select_default_mode(
        {"dense": _res(0.8, 0.8), "hybrid": _res(0.95, 0.95)}, _lat(HYBRID_P95_BUDGET_MS + 1)
    )
    assert out["default_mode"] == "dense" and "latency budget" in out["reason"]


def test_the_choice_follows_the_metrics_not_a_constant():
    better = select_default_mode({"dense": _res(0.5, 0.5), "hybrid": _res(0.6, 0.6)}, _lat(10))
    worse = select_default_mode({"dense": _res(0.6, 0.6), "hybrid": _res(0.5, 0.5)}, _lat(10))
    assert better["default_mode"] != worse["default_mode"]


def test_missing_or_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="missing evaluation result"):
        select_default_mode({"hybrid": _res(0.9, 0.9)}, _lat(10))
    with pytest.raises(ValueError, match="not valid"):
        select_default_mode({"dense": _res(0.0, 0.0), "hybrid": _res(0.9, 0.9)}, _lat(10))
    with pytest.raises(ValueError, match="latency"):
        select_default_mode({"dense": _res(0.8, 0.8), "hybrid": _res(0.9, 0.9)}, {})
