"""Deterministic stand-in for the transformer: hashed character-trigram bag, L2-normalised.

Texts that share letter sequences (cognates such as "volcano"/"volcán") get similar vectors,
which is enough to exercise dense retrieval without downloading a model.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence

import numpy as np
from app.search.encoder import normalise


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


class FakeEncoder:
    name = "fake/trigram-encoder"
    revision = "rev-1"

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            folded = f"  {_fold(text)}  "
            for i in range(len(folded) - 2):
                digest = hashlib.md5(folded[i : i + 3].encode()).digest()
                out[row, int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
            if not out[row].any():
                out[row, 0] = 1.0
        return normalise(out)
