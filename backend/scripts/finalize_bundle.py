"""Record evaluation metrics and the evidence-based default mode in a bundle's manifest.

python backend/scripts/finalize_bundle.py --bundle artifacts/index/<version>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.search.bundle import SearchBundle  # noqa: E402
from app.search.default_mode import select_default_mode  # noqa: E402

NAMES = {"dense": "dense-minilm", "lexical": "lexical-tfidf-char", "hybrid": "hybrid-rrf"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmark/v1/results")
    args = parser.parse_args()

    bundle = SearchBundle.load(args.bundle)  # refuse to finalise a corrupt bundle
    results = {m: json.loads((args.results / f"{n}.json").read_text()) for m, n in NAMES.items()}
    latency = json.loads((args.results / "latency.json").read_text())["latency"]
    for mode, result in results.items():
        if result["corpus"]["sha256"] != bundle.manifest["corpus"]["sha256"]:
            raise SystemExit(f"{mode} result was not produced from this bundle's corpus")
    choice = select_default_mode(results, latency)

    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"] = {
        "benchmark_version": results["dense"]["benchmark"]["version"],
        "queries_sha256": results["dense"]["benchmark"]["queries_sha256"],
        "qrels_sha256": results["dense"]["benchmark"]["qrels_sha256"],
        "metrics": {
            m: {k: r[k] for k in ("mrr", "recall@10", "ndcg@10", "total_queries")}
            for m, r in results.items()
        },
    }
    manifest["default_mode"] = choice
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    SearchBundle.load(args.bundle)  # still valid after the manifest update
    print(json.dumps(choice, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
