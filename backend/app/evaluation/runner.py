"""Run a retriever on the benchmark and produce a deterministic result document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from app.config import SUPPORTED_LANGUAGES
from app.evaluation.benchmark import Benchmark
from app.evaluation.corpus import Corpus
from app.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.evidence import superiority_conclusion

K = 10
LIMITATIONS = [
    "Constructed benchmark: concepts were chosen by hand and aligned through Wikipedia "
    "interlanguage links; it is not a set of real user relevance judgements.",
    "Queries come from the same article text the corpus is built from (title or lead sentence), "
    "so named entities and cognates can inflate lexical overlap.",
    "Only linked pages count as relevant; other genuinely relevant documents are scored as misses.",
    "Small query set: differences between methods of a few points are not statistically "
    "meaningful.",
    "Primary run excludes same-language documents, so it measures cross-language retrieval only.",
]


class Retriever(Protocol):
    name: str
    config: dict

    def search(
        self, query: str, target_languages: list[str], k: int
    ) -> list[tuple[str, float]]: ...


def _round(x: float) -> float:
    return round(float(x), 6)


def _mean(values: list[float]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def evaluate(retriever: Retriever, benchmark: Benchmark, corpus: Corpus, k: int = K) -> dict:
    """Evaluate ``retriever``; same-language documents are excluded from every query."""
    per_query, failures = [], 0
    by_qlang: dict[str, dict[str, list[float]]] = {}
    by_target: dict[str, dict[str, list[float]]] = {}
    by_overlap: dict[str, dict[str, list[float]]] = {}

    for q in benchmark.queries:
        relevant = set(benchmark.qrels[q.query_id])
        targets = [lang for lang in SUPPORTED_LANGUAGES if lang != q.language]
        error = None
        try:
            ranked = [key for key, _ in retriever.search(q.text, targets, k)]
        except Exception as exc:  # a failed query counts as a miss and is reported
            ranked, error = [], type(exc).__name__
            failures += 1
        rr = reciprocal_rank(ranked, relevant, k)
        rec = recall_at_k(ranked, relevant, k)
        ndcg = ndcg_at_k(ranked, relevant, k)
        bucket = by_qlang.setdefault(q.language, {"mrr": [], "recall": [], "ndcg": []})
        bucket["mrr"].append(rr)
        bucket["recall"].append(rec)
        bucket["ndcg"].append(ndcg)
        ob = by_overlap.setdefault(
            "query_contains_target_title" if q.contains_target_title else "no_title_overlap",
            {"mrr": [], "recall": [], "ndcg": []},
        )
        ob["mrr"].append(rr)
        ob["recall"].append(rec)
        ob["ndcg"].append(ndcg)
        for target in targets:
            rel_t = {key for key, lang in benchmark.qrels[q.query_id].items() if lang == target}
            if not rel_t:
                continue
            tb = by_target.setdefault(target, {"mrr": [], "recall": []})
            tb["mrr"].append(reciprocal_rank(ranked, rel_t, k))
            tb["recall"].append(recall_at_k(ranked, rel_t, k))
        per_query.append(
            {
                "query_id": q.query_id,
                "reciprocal_rank": _round(rr),
                "recall@10": _round(rec),
                "ndcg@10": _round(ndcg),
                "top_10": ranked,
                **({"error": error} if error else {}),
            }
        )

    n = len(per_query)
    return {
        "method": retriever.name,
        "retriever_config": retriever.config,
        "k": k,
        "benchmark": {
            "version": benchmark.version,
            "queries_sha256": benchmark.queries_sha256,
            "qrels_sha256": benchmark.qrels_sha256,
            "queries": n,
            "qrels": benchmark.qrel_count,
        },
        "corpus": {
            "sha256": corpus.sha256,
            "documents": len(corpus.docs),
            "per_language": corpus.language_counts(),
        },
        "same_language_results_excluded": True,
        "total_queries": n,
        "mrr": _mean([r["reciprocal_rank"] for r in per_query]),
        "recall@10": _mean([r["recall@10"] for r in per_query]),
        "ndcg@10": _mean([r["ndcg@10"] for r in per_query]),
        "per_query_language": {
            lang: {
                "queries": len(v["mrr"]),
                "mrr": _mean(v["mrr"]),
                "recall@10": _mean(v["recall"]),
                "ndcg@10": _mean(v["ndcg"]),
            }
            for lang, v in sorted(by_qlang.items())
        },
        "per_target_language": {
            lang: {
                "queries": len(v["mrr"]),
                "mrr": _mean(v["mrr"]),
                "recall@10": _mean(v["recall"]),
            }
            for lang, v in sorted(by_target.items())
        },
        "by_title_overlap": {
            name: {
                "queries": len(v["mrr"]),
                "mrr": _mean(v["mrr"]),
                "recall@10": _mean(v["recall"]),
                "ndcg@10": _mean(v["ndcg"]),
            }
            for name, v in sorted(by_overlap.items())
        },
        "failure_count": failures,
        "limitations": LIMITATIONS,
        "per_query": per_query,
    }


def write_result(result: dict, path: Path) -> str:
    """Write deterministic JSON (sorted keys, no timestamps); return the file text."""
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_report(results: dict[str, dict]) -> str:
    lines = ["# Cross-language retrieval results", ""]
    any_result = next(iter(results.values()), None)
    if any_result:
        b, c = any_result["benchmark"], any_result["corpus"]
        lines += [
            f"Benchmark {b['version']}: {b['queries']} queries, {b['qrels']} relevant documents "
            f"(queries sha256 `{b['queries_sha256'][:12]}`, qrels `{b['qrels_sha256'][:12]}`).",
            f"Corpus: {c['documents']} documents {c['per_language']} "
            f"(sha256 `{c['sha256'][:12]}`).",
            "",
        ]
    lines += ["| Method | MRR@10 | Recall@10 | nDCG@10 | Failures |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['mrr']:.4f} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} "
            f"| {r['failure_count']} |"
        )
    for name, r in results.items():
        overlap = r.get("by_title_overlap", {})
        if overlap:
            parts = ", ".join(
                f"{key}: n={v['queries']} MRR@10={v['mrr']:.4f}" for key, v in overlap.items()
            )
            lines.append(f"\n{name} by title overlap: {parts}")
    statement, notes = superiority_conclusion(results)
    lines += [
        "",
        "## Conclusion",
        "",
        statement or "No comparison can be drawn from these results.",
    ]
    lines += [f"- {note}" for note in notes]
    lines += ["", "## Limitations", ""] + [
        f"- {x}" for x in (any_result or {}).get("limitations", [])
    ]
    return "\n".join(lines) + "\n"
