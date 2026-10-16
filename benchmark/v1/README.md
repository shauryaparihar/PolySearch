# Benchmark v1 (constructed cross-language retrieval benchmark)

Built by `backend/scripts/build_benchmark.py` from Wikipedia interlanguage links. It is a
**constructed** benchmark, not real user relevance judgements.

| File | Content |
|---|---|
| `concepts.json` | 50 concept groups (en/es/fr article, URL and revision for each) |
| `queries.csv` | 100 queries: a title query and a lead-sentence query per concept; the query language rotates en → es → fr |
| `qrels.csv` | 200 relevance judgements: the linked articles in the *other* two languages |
| `manifest.json` | counts, dropped concepts, file hashes, overlap statistics |
| `corpus_manifest.json` | the searchable corpus (hash, counts) and reconciliation with the legacy dataset |
| `cache/` | raw MediaWiki API responses, so the build can be repeated offline (`--offline`) |
| `results/` | deterministic evaluation output and report |

Lead-sentence queries have parenthetical asides removed because those hold foreign-language
names and pronunciations that would leak the answer.

## Source and attribution

Texts and metadata come from Wikipedia (en/es/fr) through the official MediaWiki API. Wikipedia
text is available under Creative Commons Attribution-Share Alike 4.0 (verified live; see
`docs/source_licence_verification.md`). Each concept lists the canonical article URL and revision.
The cached API responses are verbatim API output.
