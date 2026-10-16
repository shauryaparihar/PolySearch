"""Ranking metrics with multiple relevant documents per query (binary relevance).

All functions take ``ranked`` (document keys, best first) and ``relevant`` (a set of keys).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for rank, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Binary-gain nDCG: DCG = sum 1/log2(rank+1) over relevant hits; IDCG places all
    (up to k) relevant documents first."""
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, key in enumerate(ranked[:k], start=1) if key in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
