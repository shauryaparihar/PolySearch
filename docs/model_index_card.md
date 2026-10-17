# Model and index card

What is searched, with what, how it was measured, and what is not claimed. Numbers are copied
from `benchmark/v1/results/` and `docs/load_check_results.json`.

## Corpus
- **Source:** Wikipedia (en, es, fr) via the MediaWiki API. Licence read from the API on
  2026-10-16: Creative Commons Attribution-Share Alike 4.0
  ([`source_licence_verification.md`](source_licence_verification.md)).
- **Benchmark corpus (v1):** 5,735 articles (en 1,901, es 1,953, fr 1,881), sha256 `6eddc47fdd97…`.
  Imported from the original dataset (5,585 articles; 10 rejected for an empty summary) plus the
  benchmark's concept articles fetched through the API.
- **Document text for indexing:** title, then the lead summary (bounded length).
- **Provenance:** canonical URL, revision id (empty for imported documents until a sync fills it),
  content hash and licence pointer per document.

## Encoder
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, pinned to revision
  `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; 384 dimensions, L2-normalised, CPU.
- Chosen because it is small, multilingual and fast on CPU. It was not compared with larger
  encoders; that is a possible improvement, not a finding.

## Index
- One **exact** FAISS `IndexFlatIP` per language (inner product on normalised vectors = cosine),
  so there is no approximate-search error at this size.
- A **lexical** index: character n-grams (3-5, word-boundary aware), sublinear TF-IDF, stored
  without pickle. Chosen over BM25 after BM25 scored MRR@10 0.4370 on this cross-language
  benchmark ([`lexical_baseline_choice.md`](lexical_baseline_choice.md)).
- **Hybrid** = reciprocal rank fusion of the dense and lexical rankings (k = 60).
- Versions are immutable, named `idx-<hash of file hashes>`, with a manifest listing checksums,
  counts, encoder revision and (after evaluation) metrics. Per-language partitions let a query
  search only the requested target languages.

## Benchmark (v1)
- 50 hand-chosen concepts, each linked across en/es/fr by Wikipedia interlanguage links.
- 100 queries (a title query and a lead-sentence query per concept; query language rotates);
  200 relevant documents (the linked articles in the *other* two languages).
- It is **constructed**, not real user relevance. 29 of 100 queries contain a target-language
  title; metrics are reported for both groups.

| Method | MRR@10 | Recall@10 | nDCG@10 |
|---|---|---|---|
| Lexical | 0.7484 | 0.8100 | 0.7238 |
| Dense | 0.9400 | 0.9500 | 0.9374 |
| Hybrid | 0.9100 | 0.9650 | 0.8981 |

**Selected default: dense.** Rule: hybrid would be chosen only if it beat dense on MRR@10 with no
lower nDCG@10 *and* keep its p95 latency under 500 ms. It met the latency budget (p95 29.5 ms) but did not beat dense on MRR@10 (0.9100 vs 0.9400), so dense stays the default. A new index is only
activated if its benchmark MRR@10 drops by no more than 0.05 against the active one.

## Latency (method and conditions matter)
- **Search core** (encode + vector search + fusion; one process, sequential, 100 queries after
  20 warm-up, Apple M4, CPU, torch 1 thread): dense median 9.1 ms / p95 11.3 ms; hybrid median
  25.6 / 29.5 ms; lexical 16.8 / 19.1 ms. Excludes HTTP, database and cache.
- **End to end** (Docker Compose, one uvicorn worker, client on the same machine):
  [`load_check.md`](load_check.md): median 43 ms uncached and 10 ms cache hit with one request in
  flight; throughput about 30 requests/s uncached; latency rises with concurrency.

## Privacy decisions
Query text is not stored by default (SHA-256 only); logs use an allow-list of fields; analytics
are admin-only; popular-query suggestions need an explicit flag, a minimum count and a
sensitive-text filter; click and feedback rows reference documents, not people (no accounts; client addresses are held in memory only for rate limiting, never stored).

## Limitations and intended use
Intended for demonstrating and studying cross-language retrieval on a small encyclopedic corpus.
Not validated on real user queries or other domains; the benchmark is small and constructed; a
cross-language top result is not guaranteed relevant (see [`demo.md`](demo.md) for visible
misses); the corpus is the original dataset plus API fetches, a fixed snapshot rather than live Wikipedia; no hosted
deployment has been measured.
