# Answer benchmark v1 (`rag-v1`)

30 hand-written questions over the frozen Wikipedia corpus (`artifacts/corpus_v1`):
8 direct, 8 cross-language, 4 multi-evidence, 6 unanswerable, 4 adversarial.

- `cases.jsonl` is frozen. Change it only by creating `v2`. Gold evidence is a document key plus an
  exact quote, not a passage id, so it survives re-chunking.
- Cross-language cases: the supporting fact exists only in an article of another language than the
  question (checked by the validator).
- Unanswerable cases: `absent_strings` must not occur anywhere in the corpus (checked).
- Adversarial cases add a synthetic hostile passage at rank 1; the answer must not contain the
  `forbidden_strings` canaries. These passages are not in the corpus.

Check the file against a corpus and run (free, no network, deterministic mock model):

    python backend/scripts/run_rag_eval.py --index-dir artifacts/index/idx-<v> \
        --corpus artifacts/corpus_v1/corpus.jsonl --validate-only
    python backend/scripts/run_rag_eval.py --index-dir artifacts/index/idx-<v> \
        --corpus artifacts/corpus_v1/corpus.jsonl

Reports go to `benchmark/rag/private/runs/<time>/` (git-ignored): `summary.json`, `summary.md`,
`cases.json`, `cases.csv`. A `--llm live` run needs a configured provider and the explicit flag
`--approved-paid-run`; without it the script prints the worst-case request and token count and exits.

What this does not show: mock-model generation numbers describe the pipeline, not any model.
Key-fact correctness is a substring rule; supported-claim rate is lexical overlap. With 30 cases,
one case moves a rate by about 3 points. Human-review and judge columns are blank until filled.
