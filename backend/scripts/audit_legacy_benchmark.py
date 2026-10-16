"""Reproduce the legacy corpus/query mismatch and the stored zero metrics.

Read-only. Run from the repository root:

    python backend/scripts/audit_legacy_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.evidence import BenchmarkValidationError, validate_qrels_in_corpus  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
LEGACY_RESULT = ROOT / "docs" / "legacy" / "sentence-transformers_LaBSE_results.json"


def main() -> int:
    train = pd.read_csv(PROCESSED / "all_train.csv")
    test = pd.read_csv(PROCESSED / "all_test.csv")
    queries = pd.read_csv(PROCESSED / "all_queries.csv")

    relevant = queries["relevant_doc_id"]
    print(f"indexed (train) documents : {len(train):,}")
    print(f"held-out (test) documents : {len(test):,}")
    print(f"queries                   : {len(queries):,}")
    print(f"distinct relevant docs    : {relevant.nunique():,}")
    print(f"relevant docs in train    : {relevant.isin(train['doc_id']).mean():.1%}")
    print(f"relevant docs in test     : {relevant.isin(test['doc_id']).mean():.1%}")

    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    print(
        "stored legacy metrics     : "
        f"MRR={legacy['mrr']} Recall@10={legacy['recall@10']} nDCG@10={legacy['ndcg@10']} "
        f"(queries={legacy['total_queries']})"
    )

    try:
        validate_qrels_in_corpus(queries.to_dict("records"), train["doc_id"])
    except BenchmarkValidationError as exc:
        print(f"validator verdict         : REJECTED - {exc}")
        return 1
    print("validator verdict         : ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
