"""Evaluate retrieval methods on the benchmark.

    python -m app.evaluation.run --benchmark ../benchmark/v1 \
        --corpus ../artifacts/corpus_v1/corpus.jsonl --out ../benchmark/v1/results \
        --method lexical-tfidf-char [--bundle ../artifacts/index/<version>] \
        --method dense-minilm --method hybrid-rrf

Dense and hybrid methods need ``--bundle``. The bundle must have been built from exactly
the corpus given in ``--corpus``.
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
from app.search.artifacts import ArtifactError
from app.search.bundle import BundleRetriever, SearchBundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, help="artifact directory for dense/hybrid methods")
    by_name = {name: mode for mode, name in BundleRetriever.NAMES.items()}
    parser.add_argument(
        "--method", action="append", choices=sorted(set(RETRIEVERS) | set(by_name)), required=True
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    benchmark = load_benchmark(args.benchmark)
    try:
        validate_benchmark(benchmark, corpus.keys)
    except BenchmarkValidationError as exc:
        print(f"BENCHMARK INVALID - evaluation stopped: {exc}", file=sys.stderr)
        return 2

    bundle = None
    if args.bundle or any(m in by_name and m not in RETRIEVERS for m in args.method):
        if not args.bundle:
            print("dense/hybrid methods need --bundle", file=sys.stderr)
            return 2
        try:
            bundle = SearchBundle.load(args.bundle)
        except ArtifactError as exc:
            print(f"ARTIFACTS INVALID - evaluation stopped: {exc}", file=sys.stderr)
            return 2
        if bundle.manifest["corpus"]["sha256"] != corpus.sha256:
            print("bundle was built from a different corpus than --corpus", file=sys.stderr)
            return 2

    results = {}
    for method in args.method:
        if bundle is not None:
            retriever = BundleRetriever(bundle, by_name[method])
        else:
            retriever = RETRIEVERS[method](corpus)
        result = evaluate(retriever, benchmark, corpus)
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
