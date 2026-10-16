"""Helpers that write a tiny benchmark directory for tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.evaluation.benchmark import QRELS_FIELDS, QUERY_FIELDS


def write_benchmark(directory: Path, queries: list[dict], qrels: list[dict], concepts: list[str]):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "queries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_FIELDS, lineterminator="\n", restval="false")
        w.writeheader()
        w.writerows(queries)
    with (directory / "qrels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QRELS_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(qrels)
    (directory / "concepts.json").write_text(
        json.dumps([{"concept_id": c} for c in concepts]), encoding="utf-8"
    )


def query(qid="c1-en-title", concept="c1", lang="en", text="Cancer", qtype="title"):
    return {
        "query_id": qid,
        "concept_id": concept,
        "query_language": lang,
        "query_type": qtype,
        "query_text": text,
    }


def qrel(qid, key, language, relevance=1):
    return {"query_id": qid, "doc_key": key, "doc_language": language, "relevance": relevance}
