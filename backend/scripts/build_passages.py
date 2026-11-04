"""Build the optional passage artifact for an existing search index version.

    # corpus exported from the same database the index was built from
    python backend/scripts/build_passages.py --index-dir artifacts/index/idx-<version> \
        --from-database --corpus artifacts/passages_corpus.jsonl

    # also write the passages to PostgreSQL (the index version must already be registered:
    # start the API once, or run a re-index)
    ... --sync-db

The corpus must be the one the index was built from (checked by SHA-256). Downloads nothing new:
the encoder is the one pinned in the index manifest. No language model is called.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation.corpus import load_corpus  # noqa: E402
from app.rag.passage_store import (  # noqa: E402
    PassageArtifactError,
    PassageStore,
    build_passage_artifact,
    passage_dir,
)
from app.rag.passages import ChunkConfig  # noqa: E402
from app.search.encoder import SentenceTransformerEncoder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts/corpus_v1/corpus.jsonl")
    parser.add_argument("--from-database", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200, help="estimated tokens per passage")
    parser.add_argument("--sync-db", action="store_true")
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="an earlier passage artifact directory whose vectors may be reused for unchanged "
        "text (default: the existing artifact of this index version, if any)",
    )
    args = parser.parse_args(argv)

    manifest = json.loads((args.index_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.from_database:
        from app.config import get_settings
        from app.db.base import make_engine, make_session_factory
        from app.evaluation.corpus import export_corpus

        with make_session_factory(make_engine(get_settings().database_url))() as session:
            exported = export_corpus(session, args.corpus)
        print(f"exported {exported['documents']} active documents to {args.corpus}")
    corpus = load_corpus(args.corpus)
    if corpus.sha256 != manifest["corpus"]["sha256"]:
        print(
            "the corpus differs from the one this index was built from; refusing", file=sys.stderr
        )
        return 2

    enc = manifest["encoder"]
    encoder = SentenceTransformerEncoder(enc["name"], enc["revision"])
    config = ChunkConfig.from_max_tokens(args.max_tokens)
    started = time.perf_counter()
    earlier = args.reuse_from or passage_dir(args.index_dir.parent, manifest["version"])
    try:
        previous = PassageStore.load(earlier) if (earlier / "manifest.json").is_file() else None
    except PassageArtifactError:
        previous = None  # unreadable: embed everything instead
    stats: dict = {}
    path = build_passage_artifact(
        corpus, encoder, manifest["version"], corpus.sha256, args.index_dir.parent, config,
        reuse=previous, stats=stats,
    )  # fmt: skip
    store = PassageStore.load(
        path,
        expect_index_version=manifest["version"],
        expect_encoder=(enc["name"], enc["revision"]),
    )  # reload from disk = full validation
    chars = [len(p.text) for p in store.passages]
    print(
        f"built {store.version} for {store.index_version} in {time.perf_counter() - started:.1f}s"
    )
    print(f"documents {store.manifest['documents']}, passages {store.count}")
    print(f"embedded {stats['encoded']}, reused {stats['reused']} vectors")
    print(f"per language {store.manifest['per_language']}")
    print(
        f"passage characters: median {statistics.median(chars):.0f}, max {max(chars)} "
        f"(limit {config.max_chars}); estimated tokens max "
        f"{max(p.estimated_tokens for p in store.passages)}"
    )
    print(f"artifact: {path}")

    if args.sync_db:
        from app.config import get_settings
        from app.db.base import make_engine, make_session_factory
        from app.db.models import IndexVersion
        from app.rag.passage_store import sync_passages
        from sqlalchemy import select

        with make_session_factory(make_engine(get_settings().database_url))() as session:
            version = session.scalar(
                select(IndexVersion).where(IndexVersion.version == store.index_version)
            )
            if version is None:
                print("index version is not registered in the database yet", file=sys.stderr)
                return 3
            result = sync_passages(session, store, version)
            session.commit()
        print(f"database: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
