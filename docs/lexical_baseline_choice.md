# Choice of the lexical baseline

Two deterministic lexical retrievers were compared once, on benchmark v1 (100 queries,
200 relevant documents) over the 5,735-document corpus (sha256 `6eddc47f…`), same-language
documents excluded, k = 10. Measured 2026-10-16 during Phase 2.

| Candidate | MRR@10 | Recall@10 | nDCG@10 |
|---|---|---|---|
| Character n-gram TF-IDF (`char_wb`, n = 3–5, sublinear tf) | 0.7484 | 0.8100 | 0.7238 |
| Word-level BM25 (k1 = 1.5, b = 0.75) | 0.4370 | 0.4550 | 0.4021 |

**Kept: character n-gram TF-IDF.** The gap is large and holds for every query language, so
the choice does not depend on a close call. The plan requires keeping only the clearer result,
so the BM25 implementation was removed; these BM25 numbers are recorded here but cannot be
regenerated from this repository.

Why the difference is plausible: across languages, whole words rarely match, but cognates and
named entities share many character sequences ("volcán"/"volcan", "Tsunami", "Cáncer"/"Cancer").
Character n-grams reward that partial overlap; word-level BM25 only matches identical tokens.

## What this number does and does not mean

- It is a baseline for comparison with dense and hybrid retrieval later; it is not a quality claim.
- 29 of the 100 queries contain a target-language title verbatim (after case/accent folding),
  which helps any lexical method. Split by that flag, the kept baseline scores MRR@10 0.8153
  (29 queries with overlap) and 0.7210 (71 queries without). Both are reported in
  `benchmark/v1/results/`.
- The benchmark is constructed from Wikipedia interlanguage links, not real user judgements,
  and has only 100 queries, so small differences are not meaningful.
