"""Benchmark files (queries, qrels, concepts) and their validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from app.config import SUPPORTED_LANGUAGES
from app.evidence import BenchmarkValidationError, validate_qrels_in_corpus

QUERY_FIELDS = [
    "query_id",
    "concept_id",
    "query_language",
    "query_type",
    "query_text",
    "text_equals_target_title",
    "text_contains_target_title",
]
QRELS_FIELDS = ["query_id", "doc_key", "doc_language", "relevance"]


@dataclass(frozen=True)
class Query:
    query_id: str
    concept_id: str
    language: str
    query_type: str
    text: str
    contains_target_title: bool = False


@dataclass(frozen=True)
class Benchmark:
    version: str
    queries: list[Query]
    qrels: dict[str, dict[str, str]]  # query_id -> {doc_key: doc_language}
    concept_ids: list[str]
    queries_sha256: str
    qrels_sha256: str

    @property
    def qrel_count(self) -> int:
        return sum(len(v) for v in self.qrels.values())


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = Path(path).read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    return rows, hashlib.sha256(raw).hexdigest()


def load_benchmark(directory: Path, version: str = "v1") -> Benchmark:
    directory = Path(directory)
    query_rows, q_hash = _read_csv(directory / "queries.csv")
    qrel_rows, r_hash = _read_csv(directory / "qrels.csv")
    concepts = json.loads((directory / "concepts.json").read_text(encoding="utf-8"))
    queries = [
        Query(
            r["query_id"],
            r["concept_id"],
            r["query_language"],
            r["query_type"],
            r["query_text"],
            r.get("text_contains_target_title") == "true",
        )
        for r in query_rows
    ]
    qrels: dict[str, dict[str, str]] = {}
    for r in qrel_rows:
        if int(r["relevance"]) > 0:
            qrels.setdefault(r["query_id"], {})[r["doc_key"]] = r["doc_language"]
    return Benchmark(version, queries, qrels, [c["concept_id"] for c in concepts], q_hash, r_hash)


def validate_benchmark(benchmark: Benchmark, corpus_keys: Collection[str]) -> None:
    """Raise ``BenchmarkValidationError`` listing every problem that makes the benchmark unfit."""
    problems: list[str] = []

    ids = [q.query_id for q in benchmark.queries]
    if len(ids) != len(set(ids)):
        problems.append("duplicate query ids")
    if len(benchmark.concept_ids) != len(set(benchmark.concept_ids)):
        problems.append("duplicate concept ids")
    known_concepts = set(benchmark.concept_ids)
    query_lang = {q.query_id: q.language for q in benchmark.queries}

    for q in benchmark.queries:
        if q.language not in SUPPORTED_LANGUAGES:
            problems.append(f"{q.query_id}: unsupported query language {q.language!r}")
        if q.concept_id not in known_concepts:
            problems.append(f"{q.query_id}: unknown concept {q.concept_id}")
        if not q.text.strip():
            problems.append(f"{q.query_id}: empty query text")
        if not benchmark.qrels.get(q.query_id):
            problems.append(f"{q.query_id}: no relevant documents")

    for query_id, docs in benchmark.qrels.items():
        if query_id not in query_lang:
            problems.append(f"qrels reference unknown query {query_id}")
            continue
        for key, language in docs.items():
            if language == query_lang[query_id]:
                problems.append(f"{query_id}: relevant doc {key} is in the query's own language")
            if key.split(":", 1)[0] != language:
                problems.append(
                    f"{query_id}: doc {key} does not match declared language {language}"
                )

    if problems:
        shown = "; ".join(problems[:8])
        raise BenchmarkValidationError(f"benchmark invalid ({len(problems)} problem(s)): {shown}")
    # Coverage: every relevant document must exist in the searchable corpus.
    validate_qrels_in_corpus(
        [
            {"query_id": qid, "relevant_doc_id": key}
            for qid, docs in benchmark.qrels.items()
            for key in docs
        ],
        corpus_keys,
    )
