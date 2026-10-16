"""Deterministic top-k selection shared by every retriever."""

from __future__ import annotations

import numpy as np


def top_k(
    scores: np.ndarray, keys: np.ndarray, allowed: np.ndarray, k: int
) -> list[tuple[str, float]]:
    """Highest score first; ties broken by ascending document key, so output is deterministic."""
    idx = np.flatnonzero(allowed)
    order = np.lexsort((keys[idx], -scores[idx]))
    return [(str(keys[idx[i]]), float(scores[idx[i]])) for i in order[:k]]
