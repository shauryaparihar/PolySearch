"""Measure how well the best-passage cosine score separates answerable from unanswerable
questions, and suggest a value for RAG_MIN_EVIDENCE_SCORE.

    python backend/scripts/calibrate_evidence_threshold.py \
        --index-dir artifacts/index/idx-<version> --questions <questions.jsonl>

Questions file, one JSON object per line: {"question": "...", "answerable": true|false,
"languages": ["en","es","fr"]}  ("languages" optional). Retrieval only: no language model is
called and nothing is sent anywhere.

What the numbers are, and are not: they describe the question set you give it. A small,
hand-written set shows whether a threshold is *plausible*; it does not validate it for real
traffic. Report the set size and how it was written next to any number you quote.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.passage_store import PassageStore, passage_dir  # noqa: E402
from app.rag.retrieval import RetrievalConfig, retrieve_evidence  # noqa: E402
from app.search.bundle import SearchBundle  # noqa: E402


def auc(positives: list[float], negatives: list[float]) -> float:
    """P(score of a random answerable question > score of a random unanswerable one)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def sweep(answerable: list[float], unanswerable: list[float], lo=0.2, hi=0.95, step=0.01):
    rows = []
    for i in range(round((hi - lo) / step) + 1):
        t = round(lo + i * step, 4)  # integer steps: repeated float addition drifts past ties
        false_abstain = sum(s < t for s in answerable) / len(answerable)
        false_answer = sum(s >= t for s in unanswerable) / len(unanswerable)
        rows.append(
            {
                "threshold": round(t, 2),
                "false_abstain_rate": round(false_abstain, 4),
                "false_answer_rate": round(false_answer, 4),
                "balanced_accuracy": round(1 - (false_abstain + false_answer) / 2, 4),
            }
        )
    return rows


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    return {
        "n": len(ordered), "min": round(ordered[0], 4), "p25": round(q[0], 4),
        "median": round(q[1], 4), "p75": round(q[2], 4), "max": round(ordered[-1], 4),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write the full per-question report here")
    parser.add_argument("--mode", choices=["dense", "hybrid", "lexical"], default=None)
    args = parser.parse_args(argv)

    bundle = SearchBundle.load(args.index_dir)
    store = PassageStore.load(
        passage_dir(args.index_dir.parent, bundle.version), expect_index_version=bundle.version
    )
    mode = args.mode or (bundle.manifest.get("default_mode") or {}).get("default_mode") or "dense"
    config = RetrievalConfig()
    items = [json.loads(line) for line in args.questions.read_text("utf-8").splitlines() if line]

    report = []
    for item in items:
        evidence = retrieve_evidence(
            bundle, store, item["question"], mode, item.get("languages", ["en", "es", "fr"]), config
        )
        top = evidence.passages[0] if evidence.passages else None
        report.append(
            {
                "question": item["question"],
                "answerable": bool(item["answerable"]),
                "top_score": round(top.score, 4) if top else None,
                "top_title": top.title if top else None,
                "top_language": top.language if top else None,
            }
        )
    yes = [r["top_score"] for r in report if r["answerable"] and r["top_score"] is not None]
    no = [r["top_score"] for r in report if not r["answerable"] and r["top_score"] is not None]
    if not yes or not no:
        print("need at least one answerable and one unanswerable question", file=sys.stderr)
        return 2

    table = sweep(yes, no)
    best = max(table, key=lambda r: (r["balanced_accuracy"], -abs(r["threshold"] - 0.6)))
    safe = [r for r in table if r["false_answer_rate"] == 0.0]
    summary = {
        "index_version": bundle.version,
        "mode": mode,
        "encoder": bundle.manifest["encoder"]["name"],
        "questions": len(report),
        "answerable": describe(yes),
        "unanswerable": describe(no),
        "auc": round(auc(yes, no), 4),
        "best_balanced_accuracy": best,
        "lowest_threshold_with_no_false_answers": safe[0] if safe else None,
        "note": "set size and authorship limit what this shows; see the script docstring",
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "sweep": table, "questions": report}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
