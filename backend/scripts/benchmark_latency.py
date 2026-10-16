"""Measure per-stage search latency after warm-up and write benchmark/v1/results/latency.json.

    python backend/scripts/benchmark_latency.py --bundle artifacts/index/<version>

Queries are the benchmark queries, so no user data is involved. Results depend on this
machine; the file records the conditions next to every number. API/database/cache latency
is measured in a later phase.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
import app.search  # noqa: E402,F401  (sets the macOS OpenMP guard before torch/faiss load)
import faiss  # noqa: E402,I001
import numpy as np  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402  (must be imported before faiss; see app/search/__init__.py)
from app.config import SUPPORTED_LANGUAGES  # noqa: E402
from app.evaluation.benchmark import load_benchmark  # noqa: E402
from app.search.bundle import MODES, SearchBundle  # noqa: E402

WARMUP = 20


def summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"median_ms": round(statistics.median(ordered), 3), "p95_ms": round(p95, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark/v1")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/v1/results/latency.json")
    args = parser.parse_args()

    bundle = SearchBundle.load(args.bundle)
    queries = load_benchmark(args.benchmark).queries
    plan = [(q.text, [lang for lang in SUPPORTED_LANGUAGES if lang != q.language]) for q in queries]
    result: dict = {}
    for mode in MODES:
        for text, targets in plan[:WARMUP]:  # warm-up, not measured
            bundle.search(text, mode, targets, k=10)
        stages: dict[str, list[float]] = {}
        for text, targets in plan:
            for stage, ms in bundle.search(text, mode, targets, k=10).timings_ms.items():
                stages.setdefault(stage.removesuffix("_ms"), []).append(ms)
        result[mode] = {stage: summarise(v) for stage, v in sorted(stages.items())}

    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    report = {
        "what": "per-stage search latency, single process, sequential queries, k=10, pool=50",
        "queries_measured_per_mode": len(plan),
        "warmup_queries_per_mode": WARMUP,
        "index_version": bundle.version,
        "corpus_documents": bundle.manifest["corpus"]["documents"],
        "environment": {
            "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "cpu_count": os.cpu_count(),
            "ram_gb": round(mem_bytes / 1024**3, 1),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "faiss": faiss.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "latency": result,
        "not_measured": ["HTTP/API overhead", "database", "cache hits", "concurrent load"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mode, stages in result.items():
        total = stages["total"]
        print(f"{mode:8s} total median={total['median_ms']}ms p95={total['p95_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
