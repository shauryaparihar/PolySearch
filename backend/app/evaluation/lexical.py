"""Lexical retrieval over the corpus (the deterministic baseline).

Character n-gram TF-IDF was kept over word-level BM25 on measured results (see
``docs/lexical_baseline_choice.md``). It ranks only documents in the requested target
languages and breaks score ties by document key, so a given corpus and query always produce
the same ranking.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.evaluation.corpus import Corpus
from app.ingestion.text import lexical_text


def _top_k(
    scores: np.ndarray, keys: np.ndarray, allowed: np.ndarray, k: int
) -> list[tuple[str, float]]:
    idx = np.flatnonzero(allowed)
    # Highest score first; ties broken by ascending document key (deterministic).
    order = np.lexsort((keys[idx], -scores[idx]))
    return [(str(keys[idx[i]]), float(scores[idx[i]])) for i in order[:k]]


class _Base:
    name = "lexical"

    def __init__(self, corpus: Corpus):
        self._keys = np.array([d.key for d in corpus.docs])
        self._langs = np.array([d.language for d in corpus.docs])

    def _allowed(self, target_languages: Collection[str]) -> np.ndarray:
        return np.isin(self._langs, list(target_languages))

    def search(
        self, query: str, target_languages: Sequence[str], k: int
    ) -> list[tuple[str, float]]:
        raise NotImplementedError


class CharNgramTfidf(_Base):
    name = "lexical-tfidf-char"
    config = {"analyzer": "char_wb", "ngram_range": [3, 5], "sublinear_tf": True}

    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self._vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            lowercase=False,
            dtype=np.float64,
        )
        self._matrix = self._vec.fit_transform(
            [lexical_text(d.title, d.summary) for d in corpus.docs]
        )

    def search(self, query, target_languages, k):
        q = self._vec.transform([lexical_text(query, "")])
        scores = (self._matrix @ q.T).toarray().ravel()
        return _top_k(scores, self._keys, self._allowed(target_languages), k)


RETRIEVERS = {CharNgramTfidf.name: CharNgramTfidf}
