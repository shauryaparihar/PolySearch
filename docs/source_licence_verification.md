# Source and licence verification (Wikipedia / MediaWiki)

Verified live on 2026-10-16 with `action=query&meta=siteinfo&siprop=rightsinfo`
against each language host. Nothing here was copied from memory; the connector
(`WikipediaSource.fetch_rights_info`) repeats this query and stores the result in the
`content_sources` table the first time a source row is created.

| Host | `rightsinfo.text` | `rightsinfo.url` |
|---|---|---|
| en.wikipedia.org | Creative Commons Attribution-Share Alike 4.0 | https://creativecommons.org/licenses/by-sa/4.0/deed.en |
| es.wikipedia.org | Creative Commons Attribution-Share Alike 4.0 | https://creativecommons.org/licenses/by-sa/4.0/deed.es |
| fr.wikipedia.org | Creative Commons Attribution-Share Alike 4.0 | https://creativecommons.org/licenses/by-sa/4.0/deed.fr |

The three hosts agree on the licence name; the deed links differ only by language. If a
future run finds different names, source creation fails (`rights_mismatch`) instead of
guessing. The stored `licence_url` is the English deed link exactly as reported.

What is stored per document: canonical article URL, revision id (`lastrevid`), source
update time, and a pointer to the source row holding licence name, licence URL and the
attribution sentence. Documents imported from the original dataset have no revision id
because that dataset never recorded one; it is left empty until a sync fills it in.

This file records observed metadata only. It makes no legal determination about reuse.
Phase 9 documents how attribution is displayed to users.
