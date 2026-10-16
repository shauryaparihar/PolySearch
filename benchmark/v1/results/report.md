# Cross-language retrieval results

Benchmark v1: 100 queries, 200 relevant documents (queries sha256 `7d2f2476145b`, qrels `9d76d4ca9c54`).
Corpus: 5735 documents {'en': 1901, 'es': 1953, 'fr': 1881} (sha256 `6eddc47fdd97`).

| Method | MRR@10 | Recall@10 | nDCG@10 | Failures |
|---|---|---|---|---|
| lexical-tfidf-char | 0.7484 | 0.8100 | 0.7238 | 0 |

lexical-tfidf-char by title overlap: no_title_overlap: n=71 MRR@10=0.7210, query_contains_target_title: n=29 MRR@10=0.8153

## Conclusion

No comparison can be drawn from these results.
- 1 valid method(s); at least 2 comparable methods are needed to compare retrieval quality.

## Limitations

- Constructed benchmark: concepts were chosen by hand and aligned through Wikipedia interlanguage links; it is not a set of real user relevance judgements.
- Queries come from the same article text the corpus is built from (title or lead sentence), so named entities and cognates can inflate lexical overlap.
- Only linked pages count as relevant; other genuinely relevant documents are scored as misses.
- Small query set: differences between methods of a few points are not statistically meaningful.
- Primary run excludes same-language documents, so it measures cross-language retrieval only.
