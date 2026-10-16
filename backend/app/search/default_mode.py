"""Choose the default search mode from measured evidence, never from a hard-coded belief."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.evidence import result_problems

# Hybrid may become the default only if it is also fast enough: its p95 total latency must
# stay under this budget (milliseconds, measured by scripts/benchmark_latency.py).
HYBRID_P95_BUDGET_MS = 500.0


def select_default_mode(
    results: Mapping[str, Mapping[str, Any]], latency: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """Return ``{"default_mode": ..., "reason": ...}``.

    Hybrid is chosen only if it beats dense on the frozen benchmark (higher MRR@10 and no lower
    nDCG@10) and meets the latency budget. Otherwise dense stays the default. Raises if the
    results needed for a fair comparison are missing or invalid.
    """
    for name in ("dense", "hybrid"):
        if name not in results:
            raise ValueError(f"missing evaluation result for {name!r}")
        problems = result_problems(results[name])
        if problems:
            raise ValueError(f"{name} result is not valid: {'; '.join(problems)}")
    dense, hybrid = results["dense"], results["hybrid"]
    if "hybrid" not in latency or "p95_ms" not in latency["hybrid"].get("total", {}):
        raise ValueError("missing hybrid latency measurement")
    hybrid_p95 = float(latency["hybrid"]["total"]["p95_ms"])

    beats = hybrid["mrr"] > dense["mrr"] and hybrid["ndcg@10"] >= dense["ndcg@10"]
    fast_enough = hybrid_p95 <= HYBRID_P95_BUDGET_MS
    summary = (
        f"hybrid MRR@10 {hybrid['mrr']:.4f} vs dense {dense['mrr']:.4f}; "
        f"nDCG@10 {hybrid['ndcg@10']:.4f} vs {dense['ndcg@10']:.4f}; "
        f"hybrid p95 {hybrid_p95:.1f} ms (budget {HYBRID_P95_BUDGET_MS:.0f} ms)"
    )
    if beats and fast_enough:
        return {
            "default_mode": "hybrid",
            "reason": f"hybrid beats dense and is fast enough: {summary}",
        }
    why = "hybrid does not beat dense" if not beats else "hybrid exceeds the latency budget"
    return {"default_mode": "dense", "reason": f"{why}: {summary}"}
