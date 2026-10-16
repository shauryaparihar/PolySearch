"""Build benchmark/v1 (queries, qrels, concepts, manifest) and export the canonical corpus.

Run from the repository root with the dev database migrated and the legacy corpus imported:

    python backend/scripts/build_benchmark.py [--offline]

Raw API responses are cached in benchmark/v1/cache so the build can be repeated offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.base import make_engine, make_session_factory  # noqa: E402
from app.evaluation.build import build_benchmark, corpus_manifest  # noqa: E402
from app.evaluation.corpus import export_corpus  # noqa: E402
from app.ingestion.replay import RecordReplayTransport  # noqa: E402
from app.ingestion.wikipedia import WikipediaSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark" / "v1")
    parser.add_argument(
        "--corpus-out", type=Path, default=ROOT / "artifacts" / "corpus_v1" / "corpus.jsonl"
    )
    parser.add_argument(
        "--offline", action="store_true", help="replay only; never touch the network"
    )
    args = parser.parse_args()

    transport = RecordReplayTransport(args.out / "cache", offline=args.offline)
    source = WikipediaSource(transport=transport)
    factory = make_session_factory(make_engine())
    try:
        manifest = build_benchmark(source, factory, args.out)
    finally:
        source.close()
    print(
        f"benchmark: {manifest['concepts_kept']} concepts, {manifest['queries']} queries, "
        f"{manifest['qrels']} qrels; dropped {len(manifest['dropped'])}"
    )
    print(f"api cache: {transport.hits} replayed, {transport.misses} fetched")
    with factory() as session:
        corpus_manifest_data = export_corpus(session, args.corpus_out)
    full = corpus_manifest(corpus_manifest_data, ROOT)
    (args.out / "corpus_manifest.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(full["legacy_reconciliation"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
