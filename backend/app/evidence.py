"""Evidence checks: refuse to draw conclusions that the data cannot support.

Two guards live here:

* ``validate_qrels_in_corpus`` stops an evaluation whose relevant documents
  are not in the searchable corpus (the defect in the legacy benchmark).
* ``superiority_conclusion`` only names a "best" method when at least two
  methods have valid, comparable results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REQUIRED_METRICS = ("mrr", "recall@10")
OPTIONAL_METRICS = ("ndcg@10",)  # legacy baseline results do not report it


class BenchmarkValidationError(ValueError):
    """Raised when queries/qrels cannot be evaluated against the corpus."""


def validate_qrels_in_corpus(qrels: Iterable[Mapping[str, Any]], corpus_ids: Iterable[str]) -> None:
    """Raise if any relevant document id is absent from the corpus.

    Each qrel is a mapping with ``query_id`` and ``relevant_doc_id``.
    """
    corpus = set(corpus_ids)
    missing = [
        (q.get("query_id"), q.get("relevant_doc_id"))
        for q in qrels
        if q.get("relevant_doc_id") not in corpus
    ]
    if missing:
        sample = ", ".join(f"{qid}->{doc}" for qid, doc in missing[:5])
        raise BenchmarkValidationError(
            f"{len(missing)} qrels reference documents missing from the corpus "
            f"(first: {sample}). Evaluation would be meaningless."
        )


def result_problems(result: Mapping[str, Any]) -> list[str]:
    """Return reasons a method result cannot be trusted; empty means valid."""
    problems: list[str] = []
    if not result.get("total_queries"):
        problems.append("no queries evaluated")
    values = []
    for key in REQUIRED_METRICS + OPTIONAL_METRICS:
        value = result.get(key)
        if value is None and key in OPTIONAL_METRICS:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"missing metric {key}")
        elif not math.isfinite(value) or not 0.0 <= value <= 1.0:
            problems.append(f"metric {key} out of range")
        else:
            values.append(value)
    if values and not problems and all(v == 0.0 for v in values):
        problems.append("all metrics are zero (likely an invalid benchmark)")
    return problems


def valid_methods(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {name: r for name, r in results.items() if not result_problems(r)}


def superiority_conclusion(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, Sequence[str]]:
    """Return ``(statement, notes)``; ``statement`` is None if not supportable."""
    notes = []
    for name, r in results.items():
        problems = result_problems(r)
        if problems:
            notes.append(f"{name}: {', '.join(problems)}")
    valid = valid_methods(results)
    if len(valid) < 2:
        notes.append(
            f"{len(valid)} valid method(s); at least 2 comparable methods are needed "
            "to compare retrieval quality."
        )
        return None, notes
    ranked = sorted(valid.items(), key=lambda item: item[1]["mrr"], reverse=True)
    (best, best_r), (second, second_r) = ranked[0], ranked[1]
    statement = (
        f"On this benchmark {best} has the highest MRR@10 ({best_r['mrr']:.4f}) "
        f"versus {second} ({second_r['mrr']:.4f}). This is not a claim about "
        "real-user relevance."
    )
    return statement, notes
