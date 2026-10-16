from app.evaluation.corpus import Corpus, CorpusDoc
from app.evaluation.lexical import CharNgramTfidf

DOCS = [
    CorpusDoc("en:Volcano", "en", "Volcano", "A volcano erupts lava and ash."),
    CorpusDoc("es:Volcán", "es", "Volcán", "Un volcán expulsa lava y ceniza."),
    CorpusDoc("fr:Volcan", "fr", "Volcan", "Un volcan projette de la lave et des cendres."),
    CorpusDoc("es:Glaciar", "es", "Glaciar", "Un glaciar es una masa de hielo."),
    CorpusDoc("fr:Glacier", "fr", "Glacier", "Un glacier est une masse de glace."),
    CorpusDoc("en:Glacier", "en", "Glacier", "A glacier is a mass of ice."),
]


def _retriever():
    return CharNgramTfidf(Corpus(DOCS, "x"))


def test_ranking_is_deterministic():
    r1, r2 = _retriever(), _retriever()
    for query in ("volcan lava", "glacier ice"):
        assert r1.search(query, ["es", "fr"], 5) == r2.search(query, ["es", "fr"], 5)


def test_cognate_query_retrieves_other_language_documents_and_respects_filters():
    hits = _retriever().search("Volcano", ["es", "fr"], 3)
    assert [k for k, _ in hits][:2] in (["es:Volcán", "fr:Volcan"], ["fr:Volcan", "es:Volcán"])
    assert all(not key.startswith("en:") for key, _ in hits)  # same language excluded


def test_ties_break_by_document_key():
    docs = [CorpusDoc(f"es:D{i}", "es", "mismo", "mismo texto") for i in (3, 1, 2)]
    hits = CharNgramTfidf(Corpus(docs, "x")).search("mismo texto", ["es"], 3)
    assert [k for k, _ in hits] == ["es:D1", "es:D2", "es:D3"]


def test_accents_and_case_are_folded():
    hits = _retriever().search("VOLCÁN", ["es"], 1)
    assert hits[0][0] == "es:Volcán"
