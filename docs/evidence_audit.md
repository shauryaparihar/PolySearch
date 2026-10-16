# Evidence audit of the original prototype

Date: 2026-10-16. Scope: what the original academic prototype (LaBSE + FAISS +
Streamlit) can and cannot support. Everything below was observed in this repository;
nothing is estimated.

## Summary

The prototype's evaluation could not have produced a meaningful score, and the
quality and latency figures quoted in the original paper and README are not backed by
any result in this repository.

## 1. Corpus/query mismatch (reproducible)

`src/preprocessing.py` splits each language 90/10 into train and test, then builds
evaluation queries from the **test** articles. `src/dense_retrieval.py` and
`src/baseline.py` index only `all_train.csv`. The relevant document for every query
is therefore never in the index.

Reproduce (read-only): `python backend/scripts/audit_legacy_benchmark.py`

Observed on 2026-10-16:

| Measure | Value |
|---|---|
| Indexed (train) documents | 5,036 |
| Held-out (test) documents | 561 |
| Queries | 1,108 |
| Distinct relevant documents | 561 |
| Relevant documents found in the indexed set | 0.0% |
| Relevant documents found in the held-out set | 100.0% |

A frozen pretrained encoder is not trained on these documents, so a "train/test"
split has no purpose here; the split only removed the answers from the index.

## 2. The stored result is all zeros

`docs/legacy/sentence-transformers_LaBSE_results.json` (moved from
`results/dense/`): MRR 0.0, Recall@10 0.0, nDCG@10 0.0 over 500 queries. This is the
expected outcome of defect 1, not a measurement of model quality.

## 3. Expected versus observed

| Claim | Where | Evidence in repo |
|---|---|---|
| MRR 0.68, Recall@10 0.82 for LaBSE | Paper abstract, README, PROJECT_SUMMARY | None. Stored result is 0.0 |
| BM25 beaten by 51% (MRR), translate-then-retrieve by 31% | Paper abstract | None. No baseline result files exist |
| 20 ms end-to-end query latency on CPU | Paper abstract | None. No latency measurement exists |

The README and PROJECT_SUMMARY tables labelled "expected results" were targets, not measurements. They have been removed from active documentation.

## 4. Other defects verified by reading the code

- Queries are same-language queries built from each document's own title or first
  sentence. They do not test cross-language retrieval.
- `results/baseline/` was empty (an accidental brace-named directory existed instead).
- The translate-test baseline sorts BM25 scores from three separate indexes together.
  BM25 scores are not comparable across indexes.
- `src/evaluation.py` printed a conclusion that dense retrieval "significantly
  outperforms" baselines regardless of the data. It now uses
  `backend/app/evidence.py`, which refuses to name a better method unless at least two
  methods have valid, non-degenerate results.
- `src/dense_retrieval.py` loads an unversioned pickle of embeddings without a checksum
  or a check that rows match documents. It is not used by any new code.

## 5. What was preserved

- `docs/legacy/Cross_lingual_project_paper.pdf`: the original paper, unchanged (kept on the
  author's disk and not published in the repository; its hash is recorded below).
- `docs/legacy/sentence-transformers_LaBSE_results.json`: the original stored result,
  unchanged (also present in the first Git commit).
- `docs/legacy/CHECKSUMS.sha256`: SHA-256 of both files (`shasum -a 256 -c CHECKSUMS.sha256`).

The paper itself was not edited. It still contains the unsupported figures above; that
is a known issue to resolve by revising the paper once corrected results exist.

## 6. Policy going forward

Quality and latency claims are generated only from the corrected benchmark and the
final code, with the conditions recorded next to every number.
