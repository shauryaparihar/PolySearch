"""Operator CLI for ingestion. Run from the backend/ directory:

    python -m app.ingestion.cli import-legacy --raw-dir ../data/raw
    python -m app.ingestion.cli sync --language en --titles "Mathematics" "Physics"
    python -m app.ingestion.cli sync --language es --enumerate --max-pages 100 [--resume]
    python -m app.ingestion.cli refresh --language fr --max-pages 100

Prints job counters only; never prints database URLs or credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from app.config import SUPPORTED_LANGUAGES
from app.db.base import make_engine, make_session_factory
from app.db.models import IngestionJob
from app.ingestion import service
from app.ingestion.sources import SourceError
from app.ingestion.wikipedia import WikipediaSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    legacy = sub.add_parser("import-legacy", help="import the original raw JSON corpus")
    legacy.add_argument("--raw-dir", type=Path, default=Path("../data/raw"))
    legacy.add_argument("--limit", type=int, help="max articles per language (testing)")

    sync = sub.add_parser("sync", help="seed from Wikipedia titles or by enumeration")
    sync.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES)
    group = sync.add_mutually_exclusive_group(required=True)
    group.add_argument("--titles", nargs="+")
    group.add_argument("--enumerate", action="store_true", dest="enumerate_pages")
    sync.add_argument("--max-pages", type=int)
    sync.add_argument("--resume", action="store_true", help="continue from the last stored token")

    refresh = sub.add_parser("refresh", help="re-check stored documents against live revisions")
    refresh.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES)
    refresh.add_argument("--max-pages", type=int)
    return parser


def _report(factory, status: str) -> None:
    with factory() as s:
        job = s.scalar(select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(1))
        print(
            f"job {job.id} status={status} fetched={job.fetched_count} created={job.created_count} "
            f"updated={job.updated_count} unchanged={job.unchanged_count} failed={job.failed_count}"
        )
        for item in (job.failed_items or [])[:10]:
            print(f"  failed: {item['item']} ({item['code']})")
        if job.error_code:
            print(f"  error: {job.error_code}: {job.error_message}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factory = make_session_factory(make_engine())
    source = WikipediaSource()
    try:
        if args.command == "import-legacy":
            status = service.import_legacy(factory, source, args.raw_dir, args.limit)
        elif args.command == "sync" and args.titles:
            status = service.sync_titles(factory, source, args.language, args.titles)
        elif args.command == "sync":
            token = service.last_continuation(factory, args.language) if args.resume else None
            status = service.sync_enumerate(factory, source, args.language, args.max_pages, token)
        else:
            status = service.sync_incremental(factory, source, args.language, args.max_pages)
    except SourceError as exc:
        print(f"error: {exc.code}: {exc.public_message}", file=sys.stderr)
        return 2
    finally:
        source.close()
    _report(factory, status)
    return 0 if status in ("completed", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
