"""Two-stage evidence retrieval.

1. The existing search engine picks candidate *documents* (dense or hybrid, as configured).
2. Only the passages of those documents are scored against the question with exact cosine
   similarity (one matrix product; the vectors are unit length), then filtered for near-duplicates,
   capped per document and cut to the final evidence size.

Nothing here calls a language model. Passage ids and offsets come from the artifact, so every
returned passage can be traced back to its document and the exact stored text.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.rag.passage_store import PassageArtifactError, PassageStore
from app.search.bundle import SearchBundle

DUPLICATE_SIMILARITY = 0.97  # two passages this alike are treated as the same evidence


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_documents: int = 10
    final_passages: int = 6
    max_per_document: int = 2
    duplicate_similarity: float = DUPLICATE_SIMILARITY

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalConfig:
        return cls(
            candidate_documents=settings.rag_candidate_documents,
            final_passages=settings.rag_final_passages,
            max_per_document=settings.rag_max_passages_per_document,
        )


@dataclass(frozen=True)
class EvidencePassage:
    rank: int  # position among the selected evidence, 1-based
    passage_id: str
    doc_key: str
    language: str
    title: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    estimated_tokens: int
    score: float  # cosine similarity between the question and the passage
    document_rank: int  # rank of the passage's document among the candidates, 1-based


@dataclass(frozen=True)
class Evidence:
    passages: list[EvidencePassage]
    candidate_count: int
    timings_ms: dict[str, float]


def select_evidence(
    store: PassageStore,
    query_vector: np.ndarray,
    candidate_keys: Sequence[str],
    config: RetrievalConfig,
) -> list[EvidencePassage]:
    """Rank the candidates' passages and apply the duplicate, per-document and size limits."""
    document_rank = {key: rank for rank, key in enumerate(candidate_keys, start=1)}
    rows = store.rows_for(candidate_keys)
    if not rows:
        return []
    scores = store.vectors[rows] @ query_vector.astype(np.float32)
    # best score first; ties broken by document rank, position and id so results are deterministic
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            -round(float(scores[i]), 6),
            document_rank[store.passages[rows[i]].doc_key],
            store.passages[rows[i]].ordinal,
            store.passages[rows[i]].passage_id,
        ),
    )
    chosen: list[int] = []
    per_document: dict[str, int] = {}
    for i in order:
        passage = store.passages[rows[i]]
        if per_document.get(passage.doc_key, 0) >= config.max_per_document:
            continue
        if (
            chosen
            and (store.vectors[[rows[c] for c in chosen]] @ store.vectors[rows[i]]).max()
            >= config.duplicate_similarity
        ):
            continue  # near-identical to evidence we already have
        chosen.append(i)
        per_document[passage.doc_key] = per_document.get(passage.doc_key, 0) + 1
        if len(chosen) == config.final_passages:
            break
    return [
        EvidencePassage(
            rank=rank,
            passage_id=(p := store.passages[rows[i]]).passage_id,
            doc_key=p.doc_key,
            language=p.language,
            title=p.title,
            ordinal=p.ordinal,
            text=p.text,
            char_start=p.char_start,
            char_end=p.char_end,
            estimated_tokens=p.estimated_tokens,
            score=float(scores[i]),
            document_rank=document_rank[p.doc_key],
        )
        for rank, i in enumerate(chosen, start=1)
    ]


def retrieve_evidence(
    bundle: SearchBundle,
    store: PassageStore,
    question: str,
    mode: str,
    languages: Sequence[str],
    config: RetrievalConfig,
) -> Evidence:
    """Candidate documents from the search engine, then the best passages inside them."""
    if store.index_version != bundle.version:
        raise PassageArtifactError("passages do not belong to the served index version")
    started = time.perf_counter()
    result = bundle.search(
        question, mode, languages, k=config.candidate_documents, pool=config.candidate_documents
    )
    candidate_keys = [h.key for h in result.hits]
    stage = time.perf_counter()
    query_vector = bundle.encoder.encode([question])[0]  # the passage vectors are dense
    encode_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    passages = select_evidence(store, query_vector, candidate_keys, config)
    passage_ms = (time.perf_counter() - stage) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return Evidence(
        passages=passages,
        candidate_count=len(candidate_keys),
        timings_ms={
            "document_search_ms": round(result.timings_ms.get("total_ms", 0.0), 3),
            "question_encode_ms": round(encode_ms, 3),
            "passage_rank_ms": round(passage_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
