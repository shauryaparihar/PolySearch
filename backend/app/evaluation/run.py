"""Evaluate retrieval methods on the benchmark.

    python -m app.evaluation.run --benchmark ../benchmark/v1 \
        --corpus ../artifacts/corpus_v1/corpus.jsonl --out ../benchmark/v1/results \
        --method lexical-tfidf-char
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evaluation.benchmark import load_benchmark, validate_benchmark
from app.evaluation.corpus import load_corpus
from app.evaluation.lexical import RETRIEVERS
from app.evaluation.runner import evaluate, render_report, write_result
from app.evidence import BenchmarkValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", action="append", choices=sorted(RETRIEVERS), required=True)
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    benchmark = load_benchmark(args.benchmark)
    try:
        validate_benchmark(benchmark, corpus.keys)
    except BenchmarkValidationError as exc:
        print(f"BENCHMARK INVALID - evaluation stopped: {exc}", file=sys.stderr)
        return 2

    results = {}
    for method in args.method:
        result = evaluate(RETRIEVERS[method](corpus), benchmark, corpus)
        write_result(result, args.out / f"{method}.json")
        results[method] = result
        print(
            f"{method}: MRR@10={result['mrr']:.4f} Recall@10={result['recall@10']:.4f} "
            f"nDCG@10={result['ndcg@10']:.4f} failures={result['failure_count']}"
        )
    report = render_report(results)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
