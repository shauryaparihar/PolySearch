"""Offline artifact builder: encode the corpus and write a versioned, checksummed index.

    python backend/scripts/build_artifacts.py \
        --corpus artifacts/corpus_v1/corpus.jsonl --out artifacts/index

Downloads the pinned encoder on first use (about 470 MB). Never runs inside the API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation.corpus import load_corpus  # noqa: E402
from app.search.artifacts import build_index  # noqa: E402
from app.search.bundle import SearchBundle  # noqa: E402
from app.search.encoder import DEFAULT_ENCODER, SentenceTransformerEncoder  # noqa: E402

# Pinned MiniLM revision (resolved from the Hugging Face hub on 2026-10-16).
PINNED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts/corpus_v1/corpus.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/index")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"corpus: {len(corpus.docs)} docs {corpus.language_counts()} sha256={corpus.sha256[:12]}")
    encoder = SentenceTransformerEncoder(args.encoder, args.revision)
    print(f"encoder: {encoder.name}@{encoder.revision[:12]} dim={encoder.dimension}")
    start = time.perf_counter()
    path = build_index(corpus, encoder, args.out)
    print(f"built {path.name} in {time.perf_counter() - start:.1f}s")

    bundle = SearchBundle.load(path, encoder=encoder)  # reload from disk = full validation
    print(f"reloaded and validated: {len(bundle.document_keys())} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
