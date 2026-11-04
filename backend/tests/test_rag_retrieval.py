"""Two-stage retrieval: candidates from search, then the best passages inside them."""

from __future__ import annotations

import numpy as np
import pytest
from app.evaluation.corpus import Corpus, CorpusDoc
from app.rag.passage_store import PassageArtifactError, PassageStore, build_passage_artifact
from app.rag.passages import ChunkConfig
from app.rag.retrieval import RetrievalConfig, retrieve_evidence, select_evidence
from app.search.artifacts import build_index
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder

CONFIG = ChunkConfig(max_chars=130, overlap_max_chars=60)

VOLCANO_ES = (
    "Un volcán es una estructura geológica por donde sale el magma. "
    "Las erupciones del volcán pueden ser muy violentas. "
    "Los glaciares no tienen relación con el volcán. Hay miles de volcanes."
)
DOCS = [
    CorpusDoc(
        "en:Volcano",
        "en",
        "Volcano",
        "A volcano is a rupture in the crust where lava and ash escape. "
        "Volcano eruptions can be violent.",
    ),  # fmt: skip
    CorpusDoc("es:Volcán", "es", "Volcán", VOLCANO_ES),
    CorpusDoc(
        "fr:Volcan",
        "fr",
        "Volcan",
        "Un volcan est une montagne qui crache de la lave. L'éruption du volcan est rare.",
    ),
    CorpusDoc("en:Glacier", "en", "Glacier", "A glacier is a slow river of ice. Glaciers melt."),
    CorpusDoc("es:Jazz", "es", "Jazz", "El jazz es un género musical nacido en Nueva Orleans."),
    CorpusDoc("fr:Opéra", "fr", "Opéra", "L'opéra est un spectacle chanté avec orchestre."),
]


@pytest.fixture
def env(tmp_path):
    corpus = Corpus(sorted(DOCS, key=lambda d: d.key), "e" * 64)
    encoder = FakeEncoder()
    index_path = build_index(corpus, encoder, tmp_path / "index")
    bundle = SearchBundle.load(index_path, encoder=encoder)
    path = build_passage_artifact(
        corpus, encoder, bundle.version, corpus.sha256, tmp_path / "index", CONFIG
    )
    store = PassageStore.load(path, expect_index_version=bundle.version)
    return bundle, store


def retrieve(env, question, languages=("en", "es", "fr"), mode="dense", **config):
    bundle, store = env
    return retrieve_evidence(bundle, store, question, mode, languages, RetrievalConfig(**config))


def test_a_spanish_question_finds_english_and_french_evidence(env):
    evidence = retrieve(env, "erupción volcán magma", languages=("en", "fr"))
    assert evidence.passages
    assert {p.language for p in evidence.passages} <= {"en", "fr"}  # the language filter holds
    assert evidence.passages[0].title in ("Volcano", "Volcan")


def test_evidence_is_ranked_by_score_with_sequential_ranks(env):
    evidence = retrieve(env, "volcán erupción")
    scores = [p.score for p in evidence.passages]
    assert scores == sorted(scores, reverse=True)
    assert [p.rank for p in evidence.passages] == list(range(1, len(scores) + 1))


def test_every_passage_carries_full_provenance(env):
    bundle, store = env
    evidence = retrieve(env, "volcano eruption")
    by_id = {p.passage_id: p for p in store.passages}
    for item in evidence.passages:
        stored = by_id[item.passage_id]
        assert (item.doc_key, item.language, item.title, item.ordinal) == (
            stored.doc_key, stored.language, stored.title, stored.ordinal
        )  # fmt: skip
        assert item.text == stored.text and (item.char_start, item.char_end) == (
            stored.char_start, stored.char_end
        )  # fmt: skip
        assert item.estimated_tokens == stored.estimated_tokens and item.document_rank >= 1
    assert evidence.candidate_count > 0
    assert set(evidence.timings_ms) == {
        "document_search_ms", "question_encode_ms", "passage_rank_ms", "total_ms"
    }  # fmt: skip


def test_no_document_may_fill_the_whole_context(env):
    evidence = retrieve(env, "volcán", max_per_document=1, final_passages=6)
    counts = {}
    for p in evidence.passages:
        counts[p.doc_key] = counts.get(p.doc_key, 0) + 1
    assert counts and max(counts.values()) == 1
    capped_two = retrieve(env, "volcán", max_per_document=2, final_passages=6)
    assert (
        max(
            sum(1 for p in capped_two.passages if p.doc_key == k)
            for k in {p.doc_key for p in capped_two.passages}
        )
        <= 2
    )


def test_the_final_size_limit_is_respected(env):
    assert len(retrieve(env, "volcán", final_passages=2).passages) == 2
    assert len(retrieve(env, "volcán", final_passages=1).passages) == 1


def test_only_candidate_documents_contribute_passages(env):
    wide = retrieve(env, "volcán", candidate_documents=6)
    narrow = retrieve(env, "volcán", candidate_documents=1)
    assert narrow.candidate_count == 1 and {p.document_rank for p in narrow.passages} == {1}
    assert len({p.doc_key for p in narrow.passages}) == 1
    assert len({p.doc_key for p in wide.passages}) > 1


def test_hybrid_and_lexical_candidates_also_work(env):
    for mode in ("hybrid", "lexical"):
        evidence = retrieve(env, "volcano lava", mode=mode)
        assert evidence.passages and evidence.passages[0].title in ("Volcano", "Volcan", "Volcán")


def test_retrieval_is_deterministic(env):
    first = retrieve(env, "erupción del volcán")
    second = retrieve(env, "erupción del volcán")
    assert [p.passage_id for p in first.passages] == [p.passage_id for p in second.passages]
    assert [p.score for p in first.passages] == [p.score for p in second.passages]


def test_near_identical_passages_are_kept_only_once(tmp_path):
    twin = "El jazz es un género musical nacido en Nueva Orleans."
    corpus = Corpus(
        sorted([CorpusDoc("es:Jazz", "es", "Jazz", twin),
                CorpusDoc("es:Jazz_2", "es", "Jazz", twin),
                CorpusDoc("en:Glacier", "en", "Glacier", "A glacier is ice.")],
               key=lambda d: d.key),
        "f" * 64,
    )  # fmt: skip
    encoder = FakeEncoder()
    bundle = SearchBundle.load(build_index(corpus, encoder, tmp_path / "i"), encoder=encoder)
    path = build_passage_artifact(corpus, encoder, bundle.version, corpus.sha256, tmp_path / "i")
    store = PassageStore.load(path, expect_index_version=bundle.version)
    evidence = retrieve_evidence(
        bundle, store, "jazz musical", "dense", ["es", "en"], RetrievalConfig(final_passages=6)
    )
    texts = [p.text for p in evidence.passages]
    assert texts.count(twin) == 1  # the duplicate was dropped, not shown twice
    control = retrieve_evidence(  # with de-duplication off both copies appear: the test is real
        bundle, store, "jazz musical", "dense", ["es", "en"],
        RetrievalConfig(final_passages=6, duplicate_similarity=2.0),
    )  # fmt: skip
    assert [p.text for p in control.passages].count(twin) == 2


def test_a_question_with_no_candidate_passages_returns_nothing(env):
    bundle, store = env
    assert select_evidence(store, np.ones(64, dtype=np.float32), [], RetrievalConfig()) == []
    assert (
        select_evidence(store, np.ones(64, dtype=np.float32), ["xx:none"], RetrievalConfig()) == []
    )


def test_passages_of_another_index_version_are_refused(env, tmp_path):
    bundle, store = env
    other = Corpus([CorpusDoc("en:Glacier", "en", "Glacier", "A glacier is ice.")], "g" * 64)
    encoder = FakeEncoder()
    other_bundle = SearchBundle.load(build_index(other, encoder, tmp_path / "o"), encoder=encoder)
    assert other_bundle.version != bundle.version
    with pytest.raises(PassageArtifactError):
        retrieve_evidence(other_bundle, store, "glacier", "dense", ["en"], RetrievalConfig())


def test_config_is_built_from_settings():
    from app.config import Settings

    config = RetrievalConfig.from_settings(
        Settings(
            _env_file=None,
            rag_candidate_documents=7,
            rag_final_passages=3,
            rag_max_passages_per_document=1,
        )  # fmt: skip
    )
    assert (config.candidate_documents, config.final_passages, config.max_per_document) == (7, 3, 1)
