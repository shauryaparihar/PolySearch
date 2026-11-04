"""List (default) or delete passage artifacts whose search index no longer exists.

    python backend/scripts/prune_passages.py --artifact-root artifacts/index --keep idx-<served>
    python backend/scripts/prune_passages.py --artifact-root artifacts/index --keep idx-X --delete

A passage directory is kept when its index directory is still present or it is named by --keep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.lifecycle import prune_passage_artifacts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--keep", action="append", default=[])
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)
    names = prune_passage_artifacts(args.artifact_root, set(args.keep), delete=args.delete)
    verb = "deleted" if args.delete else "would delete"
    for name in names:
        print(f"{verb}: passages/{name}")
    print(f"{len(names)} passage artifact(s) {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
