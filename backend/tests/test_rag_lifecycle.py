"""Passages follow the served index; failures only switch answer mode off, never search."""

from __future__ import annotations

import pytest
from api_support import SEARCH, docs
from app.api.state import AppState
from app.db.models import DocumentPassage
from app.evaluation.corpus import Corpus
from app.rag import lifecycle
from app.rag.passage_store import PassageStore, build_passage_artifact, passage_dir
from app.rag.passages import ChunkConfig
from fake_encoder import FakeEncoder
from job_support import KEY, add_document, get_job, start_index_job
from sqlalchemy import func, select

PROVIDER = dict(
    rag_enabled=True, llm_base_url="https://provider.invalid/v1", llm_api_key="k", llm_model="m"
)


SMALL_CHUNKS = ChunkConfig(max_chars=120, overlap_max_chars=50)


def build_for(bundle, rag_chunk=SMALL_CHUNKS):
    corpus_sha = bundle.manifest["corpus"]["sha256"]
    return build_passage_artifact(
        Corpus(docs(), "d" * 64), FakeEncoder(), bundle.version, corpus_sha,
        bundle.path.parent, rag_chunk,
    )  # fmt: skip


def _state(client) -> AppState:
    return client.app.state.polysearch


def _row_count(factory) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(DocumentPassage))


def _search_ok(client) -> None:
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200
    assert client.get("/readyz").status_code == 200  # search readiness is never affected


# ------------------------------------------------------------------------ at startup
def test_passages_are_loaded_at_startup_when_answers_are_enabled(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client(**PROVIDER)
    state = _state(client)
    assert state.passages is not None and state.passage_error is None
    assert state.passages.index_version == api_bundle.version
    _search_ok(client)


def test_nothing_is_loaded_when_answers_are_disabled(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client()  # rag_enabled defaults to false
    assert _state(client).passages is None
    assert _state(client).passage_error == lifecycle.RAG_DISABLED
    _search_ok(client)
    body = client.get("/readyz/rag")
    assert body.status_code == 503 and body.json()["checks"]["enabled"] is False


def test_missing_passages_switch_answer_mode_off_only(make_client):
    client = make_client(**PROVIDER)  # enabled, but no artifact was built
    assert _state(client).passage_error == lifecycle.PASSAGES_MISSING
    _search_ok(client)
    rag = client.get("/readyz/rag")
    assert rag.status_code == 503 and rag.json()["reason"] == lifecycle.PASSAGES_MISSING


@pytest.mark.parametrize("damage", ["vectors.npy", "passages.jsonl", "manifest.json"])
def test_a_corrupt_artifact_never_makes_search_unhealthy(make_client, api_bundle, damage):
    path = build_for(api_bundle)
    data = bytearray((path / damage).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (path / damage).write_bytes(bytes(data))
    client = make_client(**PROVIDER)
    assert _state(client).passages is None
    assert _state(client).passage_error == lifecycle.PASSAGES_INVALID
    _search_ok(client)
    assert client.get("/readyz/rag").json()["reason"] == lifecycle.PASSAGES_INVALID


def test_an_artifact_built_for_another_index_is_not_used(make_client, api_bundle, tmp_path):
    path = build_for(api_bundle)
    other = passage_dir(api_bundle.path.parent, "idx-someotherversion")
    path.rename(other)  # same files, wrong place and wrong parent version
    client = make_client(**PROVIDER)
    assert _state(client).passages is None
    _search_ok(client)


# ------------------------------------------------------------------- readiness detail
def test_rag_readiness_is_separate_and_checks_the_database_rows(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client(**PROVIDER)
    first = client.get("/readyz/rag")
    assert first.status_code == 503
    assert first.json()["checks"] == {
        "enabled": True, "provider_configured": True, "passages_loaded": True,
        "passages_match_index": True, "passages_in_database": False,
    }  # fmt: skip
    state = _state(client)
    assert lifecycle.sync_to_database(state.session_factory, state.passages) is True
    ready = client.get("/readyz/rag")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert _row_count(state.session_factory) == state.passages.count


def test_without_a_provider_answer_mode_is_not_ready_but_says_why(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client(rag_enabled=True)  # no URL, key or model
    body = client.get("/readyz/rag").json()
    assert body["ready"] is False and body["checks"]["provider_configured"] is False
    assert body["checks"]["passages_loaded"] is True
    _search_ok(client)


def test_the_rag_probe_exposes_no_paths_or_secrets(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client(**PROVIDER)
    text = client.get("/readyz/rag").text
    assert "provider.invalid" not in text and str(api_bundle.path) not in text


# --------------------------------------------------------------------- re-index flow
@pytest.fixture
def admin(make_client):
    def factory(**settings):
        return make_client(admin_api_key=KEY, **settings)

    return factory


def test_a_reindex_builds_syncs_and_activates_passages_with_the_new_index(
    admin, session_factory, api_bundle
):
    client = admin(**PROVIDER)
    old_version = api_bundle.version
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed" and job["step"] == "activated"
    state = _state(client)
    new_version = job["index_version"]
    assert new_version != old_version
    assert state.passages is not None and state.passages.index_version == new_version
    assert state.passage_error is None
    assert (passage_dir(api_bundle.path.parent, new_version) / "manifest.json").is_file()
    assert _row_count(session_factory) == state.passages.count  # synced for the new version
    assert client.get("/readyz/rag").status_code == 200
    assert any(p.doc_key == "es:Zebra_Rayada" for p in state.passages.passages)
    _search_ok(client)


def test_a_reindex_with_answers_disabled_builds_no_passages(admin, session_factory, api_bundle):
    client = admin()
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed"
    assert not (api_bundle.path.parent / "passages").exists()
    assert _row_count(session_factory) == 0


def test_a_failed_passage_build_does_not_block_the_new_search_index(
    admin, session_factory, api_bundle, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(lifecycle, "build_passage_artifact", boom)
    client = admin(**PROVIDER)
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed" and job["step"] == "activated"  # search went live
    state = _state(client)
    assert state.passages is None and state.passage_error == lifecycle.PASSAGES_BUILD_FAILED
    assert client.get("/api/v1/stats").json()["index_version"] == job["index_version"]
    _search_ok(client)
    assert client.get("/readyz/rag").json()["reason"] == lifecycle.PASSAGES_BUILD_FAILED


def test_a_failed_database_sync_disables_answers_but_not_search(
    admin, session_factory, api_bundle, monkeypatch
):
    monkeypatch.setattr(lifecycle, "sync_to_database", lambda *a, **k: False)
    client = admin(**PROVIDER)
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed"
    assert _state(client).passage_error == lifecycle.PASSAGES_SYNC_FAILED
    _search_ok(client)


def test_activating_an_index_drops_passages_that_belong_to_the_old_one(make_client, api_bundle):
    build_for(api_bundle)
    client = make_client(**PROVIDER)
    state = _state(client)
    assert isinstance(state.passages, PassageStore)
    bundle, version_id = state.snapshot()
    state.activate(bundle, version_id, state.suggester)  # an activation without passages
    assert state.passages is None and state.passage_error == lifecycle.PASSAGES_PENDING
    state.activate(bundle, version_id, state.suggester, passages=None, passage_error="X")
    assert state.passage_error == "X"


def test_an_unchanged_corpus_reindex_keeps_the_existing_passages(admin, api_bundle):
    build_for(api_bundle)
    client = admin(**PROVIDER)
    before = _state(client).passages
    job = get_job(client, "index", start_index_job(client))
    assert job["step"] == "no_changes"
    assert _state(client).passages is before


def test_a_reindex_only_embeds_passages_that_are_new_or_changed(
    admin, session_factory, api_bundle, log_records
):
    from app.rag.passages import DEFAULT_CONFIG

    build_passage_artifact(
        Corpus(docs(), "d" * 64), FakeEncoder(), api_bundle.version,
        api_bundle.manifest["corpus"]["sha256"], api_bundle.path.parent, DEFAULT_CONFIG,
    )  # fmt: skip
    client = admin(**PROVIDER)
    before = _state(client).passages.count
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed"
    built = next(r for r in log_records if r.getMessage() == "passages_built")
    assert built.fields["passages_reused"] == before  # every old passage kept its vector
    assert 0 < built.fields["passages_encoded"] <= 3  # only the new document was embedded
    assert _state(client).passages.count == before + built.fields["passages_encoded"]
