import pytest
from app.config import Settings
from app.db.models import Document
from app.evaluation.benchmark import load_benchmark, validate_benchmark
from app.evaluation.build import build_benchmark, lead_sentence
from app.evaluation.corpus import export_corpus, load_corpus
from app.evidence import BenchmarkValidationError
from app.ingestion.replay import RecordReplayTransport
from app.ingestion.sources import SourceError
from app.ingestion.wikipedia import WikipediaSource
from fake_mediawiki import FakeMediaWiki
from sqlalchemy import select

LEAD = {
    "en": "Volcanoes (from Latin) are openings in a planet's crust that let lava out. More text.",
    "es": "Un volcán (en latín volcanus) es una abertura en la corteza que deja salir lava. Más.",
    "fr": "Un volcan (du latin volcanus) est une ouverture de la croûte par où sort la lave. Plus.",
}


def _pages(with_fr_link_for_glacier=False):
    def doc(lang, title, extract=None, **extra):
        return {"revid": 7, "extract": extract or LEAD[lang], **extra}

    return {
        "en": {
            "Volcano": doc("en", "Volcano", langlinks={"es": "Volcanes", "fr": "Volcan"}),
            "Glacier": doc(
                "en",
                "Glacier",
                extract=LEAD["en"].replace("Volcanoes", "Glaciers"),
                langlinks={
                    "es": "Glaciar",
                    **({"fr": "Glacier"} if with_fr_link_for_glacier else {}),
                },
            ),
        },
        "es": {
            "Volcanes": {"revid": 8, "redirect_to": "Volcán"},  # link points at a redirect
            "Volcán": doc("es", "Volcán"),
            "Glaciar": doc("es", "Glaciar", extract=LEAD["es"].replace("volcán", "glaciar")),
        },
        "fr": {
            "Volcan": doc("fr", "Volcan"),
            "Glacier": doc("fr", "Glacier", extract=LEAD["fr"].replace("volcan", "glacier")),
        },
    }


def _source(pages):
    fake = FakeMediaWiki(pages, page_size=50)
    return fake, WikipediaSource(Settings(), transport=fake.transport(), sleep=lambda _s: None)


def test_resolve_titles_normalises_and_follows_redirects():
    _, source = _source(_pages())
    resolved = source.resolve_titles("es", ["Volcanes", "Volcán", "Glaciar", "No existe"])
    assert resolved == {
        "Volcanes": "Volcán",
        "Volcán": "Volcán",
        "Glaciar": "Glaciar",
        "No existe": None,
    }


def test_langlinks_map_each_source_page_to_its_linked_title():
    _, source = _source(_pages())
    links = source.fetch_langlinks("en", ["Volcano", "Glacier"], "es")
    assert links == {"Volcano": "Volcanes", "Glacier": "Glaciar"}


def test_lead_sentence_removes_parenthetical_asides_and_rejects_short_ones():
    text = lead_sentence(LEAD["es"])
    assert "latín" not in text and "volcanus" not in text
    assert text == "Un volcán es una abertura en la corteza que deja salir lava."
    assert lead_sentence("Corto. Otro.") is None


def test_benchmark_is_built_validated_and_rotates_query_languages(session_factory, tmp_path):
    _, source = _source(_pages(with_fr_link_for_glacier=True))
    out = tmp_path / "b"
    manifest = build_benchmark(
        source, session_factory, out, seeds=["Volcano", "Glacier"], min_concepts=2
    )
    assert manifest["concepts_kept"] == 2 and manifest["dropped"] == []
    with session_factory() as s:
        corpus_path = tmp_path / "corpus.jsonl"
        export_corpus(s, corpus_path)
    corpus = load_corpus(corpus_path)
    bench = load_benchmark(out)
    validate_benchmark(bench, corpus.keys)  # qrels reference real foreign-language documents
    assert "es:Volcán" in bench.qrels["c000-en-title"]  # redirect resolved to the article
    assert {q.language for q in bench.queries} == {"en", "es"}  # rotation: concept 0 -> en, 1 -> es
    assert all(len(v) == 2 for v in bench.qrels.values())
    with session_factory() as s:
        assert (
            s.scalar(select(Document).where(Document.external_id == "Volcán")).source_revision
            == "7"
        )


def test_concepts_without_a_complete_set_of_links_are_dropped_not_padded(session_factory, tmp_path):
    _, source = _source(_pages(with_fr_link_for_glacier=False))
    with pytest.raises(BenchmarkValidationError, match="refusing to pad"):
        build_benchmark(
            source, session_factory, tmp_path / "b", seeds=["Volcano", "Glacier"], min_concepts=2
        )
    manifest = build_benchmark(
        source, session_factory, tmp_path / "c", seeds=["Volcano", "Glacier"], min_concepts=1
    )
    assert manifest["dropped"] == [{"seed": "Glacier", "reason": "no linked page in fr"}]


def test_record_replay_transport_replays_offline_and_refuses_cache_misses(tmp_path):
    fake = FakeMediaWiki(_pages(), page_size=50)
    cache = tmp_path / "cache"
    online = RecordReplayTransport(cache, inner=fake.transport())
    source = WikipediaSource(Settings(), transport=online, sleep=lambda _s: None)
    first = source.resolve_titles("es", ["Volcanes"])
    assert online.misses == 1

    offline = RecordReplayTransport(cache, offline=True)
    replay = WikipediaSource(Settings(http_max_retries=0), transport=offline)
    assert replay.resolve_titles("es", ["Volcanes"]) == first
    assert offline.hits == 1 and offline.misses == 0
    with pytest.raises(SourceError) as exc:  # a request that was never recorded cannot be served
        replay.resolve_titles("es", ["Glaciar"])
    assert exc.value.code == "network_error"
