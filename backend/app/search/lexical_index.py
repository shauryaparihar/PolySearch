"""Character n-gram TF-IDF index with a pickle-free on-disk format.

Maths (identical to scikit-learn's ``TfidfVectorizer(sublinear_tf=True)`` defaults):
    tf'      = 1 + ln(count)
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1
    vector   = L2-normalise(tf' * idf)
Documents and queries are scored by cosine similarity (a dot product of unit vectors).

Saved as ``vocab`` (JSON), ``idf`` (.npy) and ``matrix`` (scipy .npz): all loaded with
``allow_pickle=False``, so a tampered file cannot execute code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from app.search.ranking import top_k

CONFIG = {
    "analyzer": "char_wb",
    "ngram_range": [3, 5],
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
}


def _vectorizer(vocabulary: dict[str, int] | None = None) -> CountVectorizer:
    return CountVectorizer(
        analyzer=CONFIG["analyzer"],
        ngram_range=tuple(CONFIG["ngram_range"]),
        lowercase=False,
        vocabulary=vocabulary,
        dtype=np.float64,
    )


def _weight(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    counts = counts.tocsr(copy=True)
    counts.data = 1.0 + np.log(counts.data)
    weighted = counts.multiply(idf).tocsr()
    return normalize(weighted, norm="l2", copy=False)


@dataclass
class LexicalIndex:
    vocab: dict[str, int]
    idf: np.ndarray
    matrix: sp.csr_matrix
    keys: np.ndarray
    languages: np.ndarray

    @classmethod
    def build(cls, texts: Sequence[str], keys: Sequence[str], languages: Sequence[str]):
        vec = _vectorizer()
        counts = vec.fit_transform(texts).tocsr()
        n_docs = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return cls(
            vocab={term: int(i) for term, i in vec.vocabulary_.items()},
            idf=idf,
            matrix=_weight(counts, idf),
            keys=np.array(keys),
            languages=np.array(languages),
        )

    def score(self, query_text: str) -> np.ndarray:
        counts = _vectorizer(self.vocab).transform([query_text])
        if counts.nnz == 0:
            return np.zeros(self.matrix.shape[0])
        q = _weight(counts, self.idf)
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def search(
        self, query_text: str, target_languages: Sequence[str], k: int
    ) -> list[tuple[str, float]]:
        allowed = np.isin(self.languages, list(target_languages))
        return top_k(self.score(query_text), self.keys, allowed, k)

    # ------------------------------------------------------------------ storage
    FILES = ("lexical_vocab.json", "lexical_idf.npy", "lexical_matrix.npz", "lexical_docs.json")

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        (directory / "lexical_vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        np.save(directory / "lexical_idf.npy", self.idf, allow_pickle=False)
        sp.save_npz(directory / "lexical_matrix.npz", self.matrix)
        (directory / "lexical_docs.json").write_text(
            json.dumps({"keys": self.keys.tolist(), "languages": self.languages.tolist()}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LexicalIndex:
        directory = Path(directory)
        docs = json.loads((directory / "lexical_docs.json").read_text(encoding="utf-8"))
        return cls(
            vocab=json.loads((directory / "lexical_vocab.json").read_text(encoding="utf-8")),
            idf=np.load(directory / "lexical_idf.npy", allow_pickle=False),
            matrix=sp.load_npz(directory / "lexical_matrix.npz").tocsr(),
            keys=np.array(docs["keys"]),
            languages=np.array(docs["languages"]),
        )
