# Cross-language retrieval results

Benchmark v1: 100 queries, 200 relevant documents (queries sha256 `7d2f2476145b`, qrels `9d76d4ca9c54`).
Corpus: 5735 documents {'en': 1901, 'es': 1953, 'fr': 1881} (sha256 `6eddc47fdd97`).

| Method | MRR@10 | Recall@10 | nDCG@10 | Failures |
|---|---|---|---|---|
| lexical-tfidf-char | 0.7484 | 0.8100 | 0.7238 | 0 |
| dense-minilm | 0.9400 | 0.9500 | 0.9374 | 0 |
| hybrid-rrf | 0.9100 | 0.9650 | 0.8981 | 0 |

lexical-tfidf-char by title overlap: no_title_overlap: n=71 MRR@10=0.7210, query_contains_target_title: n=29 MRR@10=0.8153

dense-minilm by title overlap: no_title_overlap: n=71 MRR@10=0.9225, query_contains_target_title: n=29 MRR@10=0.9828

hybrid-rrf by title overlap: no_title_overlap: n=71 MRR@10=0.9049, query_contains_target_title: n=29 MRR@10=0.9224

## Conclusion

On this benchmark dense-minilm has the highest MRR@10 (0.9400) versus hybrid-rrf (0.9100). This is not a claim about real-user relevance.

## Limitations

- Constructed benchmark: concepts were chosen by hand and aligned through Wikipedia interlanguage links; it is not a set of real user relevance judgements.
- Queries come from the same article text the corpus is built from (title or lead sentence), so named entities and cognates can inflate lexical overlap.
- Only linked pages count as relevant; other genuinely relevant documents are scored as misses.
- Small query set: differences between methods of a few points are not statistically meaningful.
- Primary run excludes same-language documents, so it measures cross-language retrieval only.
