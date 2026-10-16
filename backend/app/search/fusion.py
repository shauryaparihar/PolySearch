"""Reciprocal Rank Fusion (RRF).

Dense cosine similarities and TF-IDF scores live on different scales, so adding or averaging
raw scores would be arbitrary. RRF uses only ranks:

    rrf_score(d) = sum over methods of 1 / (k + rank_in_method(d))      (ranks start at 1)

A document missing from a method's candidate list simply contributes nothing for that method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

RRF_K = 60  # fixed, documented constant (the value used in the original RRF paper)


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked lists of document keys.

    Returns ``(key, score, {method: rank})`` sorted by score descending, ties broken by key.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ranked in rankings.items():
        for position, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            ranks.setdefault(key, {})[method] = position
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [(key, scores[key], ranks[key]) for key in ordered]
