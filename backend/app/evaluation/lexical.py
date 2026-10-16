"""Standalone lexical retriever over a corpus (the deterministic baseline).

Character n-gram TF-IDF was kept over word-level BM25 on measured results (see
``docs/lexical_baseline_choice.md``). It shares its implementation with the deployed lexical
artifact (``app.search.lexical_index``), so evaluation and serving use the same maths.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.evaluation.corpus import Corpus
from app.ingestion.text import lexical_text
from app.search import lexical_index


class CharNgramTfidf:
    name = "lexical-tfidf-char"
    config = dict(lexical_index.CONFIG)

    def __init__(self, corpus: Corpus):
        self._index = lexical_index.LexicalIndex.build(
            [lexical_text(d.title, d.summary) for d in corpus.docs],
            [d.key for d in corpus.docs],
            [d.language for d in corpus.docs],
        )

    def search(
        self, query: str, target_languages: Sequence[str], k: int
    ) -> list[tuple[str, float]]:
        return self._index.search(lexical_text(query, ""), target_languages, k)


RETRIEVERS = {CharNgramTfidf.name: CharNgramTfidf}
