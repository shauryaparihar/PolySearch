"""Run the answer-mode benchmark and write JSON, CSV and Markdown reports.

    python backend/scripts/run_rag_eval.py --index-dir artifacts/index/idx-<version> \
        --corpus artifacts/corpus_v1/corpus.jsonl

Default is the deterministic mock (no network, free). ``--llm live`` needs a configured
provider (LLM_* settings) AND ``--approved-paid-run``; without the flag it only prints the
worst-case request count and token budget, then exits with code 3.
Exit codes: 0 ok, 2 bad input, 3 live run not approved.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.evaluation.rag_benchmark import load_benchmark, validate_against_corpus  # noqa: E402
from app.evaluation.rag_runner import (  # noqa: E402
    CONFIGS,
    GENERATION_CONFIGS,
    collect_metadata,
    run_evaluation,
    summarise,
    write_reports,
)
from app.rag.fake import ExtractiveFakeLLM  # noqa: E402
from app.rag.openai_client import build_llm_client  # noqa: E402
from app.rag.passage_store import PassageStore, passage_dir  # noqa: E402
from app.search.bundle import SearchBundle  # noqa: E402

DEFAULT_BENCHMARK = ROOT / "benchmark/rag/v1/cases.jsonl"


def estimate(settings: Settings, cases: int, configs: tuple[str, ...]) -> dict:
    calls = cases * sum(c in GENERATION_CONFIGS for c in configs)
    worst_requests = calls * (1 + settings.rag_max_repair_attempts)
    per_request = settings.rag_max_context_tokens + settings.llm_max_output_tokens
    return {"requests_expected": calls, "requests_worst_case": worst_requests,
            "max_tokens_worst_case": worst_requests * per_request}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--llm", choices=["mock", "live"], default="mock")
    parser.add_argument("--approved-paid-run", action="store_true")
    parser.add_argument("--mode", default="dense", help="search mode for the search_only config")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    configs = tuple(c for c in args.configs.split(",") if c)
    if not configs or any(c not in CONFIGS for c in configs):
        print(f"--configs must be a subset of {CONFIGS}", file=sys.stderr)
        return 2
    try:
        benchmark = load_benchmark(args.benchmark)
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark invalid: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summaries: dict[str, str] = {}
    languages: dict[str, str] = {}
    for line in args.corpus.read_text("utf-8").splitlines():
        if line:
            doc = json.loads(line)
            summaries[doc["key"]] = doc["summary"]
            languages[doc["key"]] = doc["language"]
    problems = validate_against_corpus(benchmark.cases, summaries, languages)
    if problems:
        print("benchmark does not match the corpus:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if args.validate_only:
        print(f"ok: {len(benchmark.cases)} cases, sha256 {benchmark.sha256[:12]}")
        return 0

    settings = Settings()
    if args.llm == "live":
        plan = estimate(settings, len(benchmark.cases), configs)
        client = build_llm_client(settings)
        if client is None:
            print("live run needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL", file=sys.stderr)
            return 2
        if not args.approved_paid_run:
            print(f"NOT RUN. A live run would make up to {plan['requests_worst_case']} paid "
                  f"requests (expected {plan['requests_expected']}), at most "
                  f"{plan['max_tokens_worst_case']} tokens. Re-run with --approved-paid-run "
                  "after the owner approves.")  # fmt: skip
            return 3
        llm = client
    else:
        llm = ExtractiveFakeLLM()

    bundle = SearchBundle.load(args.index_dir)
    store = PassageStore.load(
        passage_dir(args.index_dir.parent, bundle.version), expect_index_version=bundle.version
    )
    results = run_evaluation(bundle, store, benchmark, llm, settings, configs, args.mode)
    metadata = collect_metadata(bundle, store, benchmark, llm, settings, args.llm, args.mode)
    summary = summarise(results, benchmark, configs, settings)
    out = args.out_dir or ROOT / "benchmark/rag/private/runs" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    for path in write_reports(out, metadata, summary, results):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
