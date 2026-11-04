"""Passage artifacts: build, validate, tamper detection, atomic publishing, PostgreSQL sync."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from api_support import docs, seed_documents
from app.db.models import Document, DocumentPassage, IndexVersion
from app.evaluation.corpus import Corpus, CorpusDoc
from app.rag import passage_store
from app.rag.passage_store import (
    PassageArtifactError,
    PassageStore,
    build_passage_artifact,
    passage_dir,
    sync_passages,
)
from app.rag.passages import ChunkConfig
from fake_encoder import FakeEncoder
from sqlalchemy import func, select

CONFIG = ChunkConfig(max_chars=110, overlap_max_chars=60)
VERSION = "idx-test0000001"


def corpus() -> Corpus:
    long_text = (
        "Un volcán es una estructura geológica. Por ella emerge magma desde el interior. "
        "Las erupciones pueden ser violentas. Existen miles en el mundo."
    )
    return Corpus(
        [
            CorpusDoc(
                "en:Volcano", "en", "Volcano", "A volcano is a rupture in a crust. Magma erupts."
            ),
            CorpusDoc("es:Volcán", "es", "Volcán", long_text),
            CorpusDoc(
                "fr:Volcan", "fr", "Volcan", "Un volcan est une montagne. Il entre en éruption."
            ),
        ],
        "c" * 64,
    )


def build(tmp_path, encoder=None, version=VERSION, c=None):
    return build_passage_artifact(
        c or corpus(), encoder or FakeEncoder(), version, "c" * 64, tmp_path, CONFIG
    )


def load(path, **kwargs):
    return PassageStore.load(path, expect_index_version=VERSION, **kwargs)


# ----------------------------------------------------------------------- build / load
def test_build_then_load_round_trips_and_describes_itself(tmp_path):
    path = build(tmp_path)
    assert path == passage_dir(tmp_path, VERSION)
    store = load(path, expect_encoder=("fake/trigram-encoder", "rev-1"))
    manifest = store.manifest
    assert store.version.startswith("psg-") and store.index_version == VERSION
    assert manifest["schema_version"] == 1 and manifest["chunker"]["max_chars"] == 110
    assert manifest["parent"] == {"index_version": VERSION, "corpus_sha256": "c" * 64}
    assert manifest["documents"] == 3 and manifest["passages"] == store.count > 3
    for language, count in manifest["per_language"].items():
        assert count == sum(1 for p in store.passages if p.language == language) > 0
    assert store.vectors.shape == (store.count, 64) and store.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(store.vectors, axis=1), 1.0, atol=1e-5)
    assert [p.doc_key for p in store.passages] == sorted(p.doc_key for p in store.passages)


def test_the_artifact_files_are_reproducible(tmp_path):
    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    assert load(a).manifest["files"] == load(b).manifest["files"]
    assert load(a).version == load(b).version


def test_passage_ids_are_stable_across_builds(tmp_path):
    first = [p.passage_id for p in load(build(tmp_path / "a")).passages]
    second = [p.passage_id for p in load(build(tmp_path / "b")).passages]
    assert first == second and len(set(first)) == len(first)


def test_rows_for_returns_only_the_requested_documents(tmp_path):
    store = load(build(tmp_path))
    rows = store.rows_for(["es:Volcán"])
    assert rows and {store.passages[r].doc_key for r in rows} == {"es:Volcán"}
    assert store.rows_for(["xx:Unknown"]) == []


def test_an_empty_corpus_is_refused(tmp_path):
    with pytest.raises(PassageArtifactError):
        build(tmp_path, c=Corpus([], "c" * 64))
    assert not (tmp_path / "passages").exists() or not list((tmp_path / "passages").glob("*"))


# ----------------------------------------------------------------- validation failures
def _rewrite_manifest(path, **changes):
    manifest = json.loads((path / "manifest.json").read_text())
    for key, value in changes.items():
        manifest[key] = value
    (path / "manifest.json").write_text(json.dumps(manifest))


def _refresh_hashes(path):
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"] = {n: passage_store._sha256(path / n) for n in manifest["files"]}
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_a_flipped_byte_in_either_file_is_detected(tmp_path):
    for name in ("vectors.npy", "passages.jsonl"):
        path = build(tmp_path / name.split(".")[0])
        data = bytearray((path / name).read_bytes())
        data[len(data) // 2] ^= 0xFF
        (path / name).write_bytes(bytes(data))
        with pytest.raises(PassageArtifactError, match="checksum"):
            load(path)


def test_missing_files_and_manifest_are_detected(tmp_path):
    path = build(tmp_path)
    (path / "vectors.npy").unlink()
    with pytest.raises(PassageArtifactError, match="missing file"):
        load(path)
    (path / "manifest.json").unlink()
    with pytest.raises(PassageArtifactError, match="manifest"):
        load(path)


def test_a_garbled_manifest_is_a_clean_error(tmp_path):
    path = build(tmp_path)
    (path / "manifest.json").write_text("{not json")
    with pytest.raises(PassageArtifactError):
        load(path)
    (path / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(PassageArtifactError):
        load(path)


def test_rows_and_vectors_must_agree_even_if_checksums_were_updated(tmp_path):
    path = build(tmp_path)
    vectors = np.load(path / "vectors.npy")
    np.save(path / "vectors.npy", vectors[:-1], allow_pickle=False)  # one vector short
    _refresh_hashes(path)
    with pytest.raises(PassageArtifactError, match="differ"):
        load(path)


def test_wrong_dimension_nan_and_unnormalised_vectors_are_rejected(tmp_path):
    for label, mutate in {
        "dim": lambda v: np.hstack([v, v[:, :1]]),
        "nan": lambda v: np.where(np.arange(v.size).reshape(v.shape) == 3, np.nan, v),
        "scale": lambda v: v * 2,
        "dtype": lambda v: v.astype(np.float64),
    }.items():
        path = build(tmp_path / label)
        np.save(path / "vectors.npy", mutate(np.load(path / "vectors.npy")), allow_pickle=False)
        _refresh_hashes(path)
        with pytest.raises(PassageArtifactError):
            load(path)


def test_a_passage_edited_after_the_fact_fails_its_hash(tmp_path):
    path = build(tmp_path)
    lines = (path / "passages.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["text"] = "tampered text"
    lines[0] = json.dumps(row)
    (path / "passages.jsonl").write_text("\n".join(lines) + "\n")
    _refresh_hashes(path)
    with pytest.raises(PassageArtifactError, match="hash or offsets"):
        load(path)


def test_duplicate_ids_and_wrong_language_counts_are_rejected(tmp_path):
    path = build(tmp_path / "dup")
    lines = (path / "passages.jsonl").read_text().splitlines()
    lines[1] = lines[0]
    (path / "passages.jsonl").write_text("\n".join(lines) + "\n")
    _refresh_hashes(path)
    with pytest.raises(PassageArtifactError):
        load(path)
    path = build(tmp_path / "counts")
    _rewrite_manifest(path, per_language={"en": 99, "es": 0, "fr": 0})
    with pytest.raises(PassageArtifactError, match="per-language"):
        load(path)


def test_the_artifact_must_belong_to_the_served_index_and_encoder(tmp_path):
    path = build(tmp_path)
    with pytest.raises(PassageArtifactError, match="different index"):
        PassageStore.load(path, expect_index_version="idx-other")
    with pytest.raises(PassageArtifactError, match="different encoder"):
        PassageStore.load(path, expect_encoder=("other/model", "rev-1"))
    with pytest.raises(PassageArtifactError, match="different encoder"):
        PassageStore.load(path, expect_encoder=("fake/trigram-encoder", "rev-2"))


def test_an_unknown_schema_version_is_refused(tmp_path):
    path = build(tmp_path)
    _rewrite_manifest(path, schema_version=99)
    with pytest.raises(PassageArtifactError, match="schema"):
        load(path)


# ------------------------------------------------------------------ atomic publishing
class ExplodingEncoder(FakeEncoder):
    def encode(self, texts):
        raise RuntimeError("encoder exploded")


def test_a_failed_build_leaves_no_partial_artifact_and_keeps_the_previous_one(tmp_path):
    with pytest.raises(RuntimeError):
        build(tmp_path, encoder=ExplodingEncoder())
    parent = tmp_path / "passages"
    assert not any(parent.iterdir())  # no staging directory left behind

    good = build(tmp_path)
    before = (good / "manifest.json").read_text()
    with pytest.raises(RuntimeError):
        build(tmp_path, encoder=ExplodingEncoder())
    assert (good / "manifest.json").read_text() == before  # the old artifact is untouched
    assert load(good).count > 0
    assert [p.name for p in parent.iterdir()] == [VERSION]


def test_rebuilding_the_same_version_replaces_it_atomically(tmp_path):
    first = build(tmp_path)
    old_version = load(first).version
    changed = Corpus(corpus().docs[:2], "c" * 64)
    second = build(tmp_path, c=changed)
    assert second == first and load(second).version != old_version
    assert [p.name for p in (tmp_path / "passages").iterdir()] == [VERSION]  # no .old leftovers


# --------------------------------------------------------------------- database sync
@pytest.fixture
def registered(session_factory, api_bundle, tmp_path):
    from app.api.service import register_index_version

    seed_documents(session_factory)
    path = build_passage_artifact(
        Corpus(docs(), "d" * 64), FakeEncoder(), api_bundle.version, "d" * 64, tmp_path, CONFIG
    )
    store = PassageStore.load(path, expect_index_version=api_bundle.version)
    with session_factory() as session:
        register_index_version(session, api_bundle)
        session.commit()
    return session_factory, store, api_bundle.version


def _stored(factory):
    with factory() as session:
        return list(session.scalars(select(DocumentPassage)))


def test_sync_writes_every_passage_with_the_artifacts_ids(registered):
    factory, store, version = registered
    with factory() as session:
        index = session.scalar(select(IndexVersion).where(IndexVersion.version == version))
        result = sync_passages(session, store, index)
        session.commit()
    assert result == {"inserted": store.count, "skipped_inactive_documents": 0}
    rows = _stored(factory)
    assert {str(r.id) for r in rows} == {p.passage_id for p in store.passages}
    by_id = {str(r.id): r for r in rows}
    for p in store.passages:
        row = by_id[p.passage_id]
        assert (row.text, row.ordinal, row.char_start, row.char_end) == (
            p.text, p.ordinal, p.char_start, p.char_end
        )  # fmt: skip
        assert row.content_sha256 == p.content_sha256 and row.language == p.language


def test_sync_is_idempotent(registered):
    factory, store, version = registered
    for _ in range(2):
        with factory() as session:
            index = session.scalar(select(IndexVersion).where(IndexVersion.version == version))
            sync_passages(session, store, index)
            session.commit()
    assert len(_stored(factory)) == store.count


def test_sync_skips_documents_that_are_no_longer_active(registered):
    factory, store, version = registered
    with factory() as session:
        doc = session.scalars(select(Document).where(Document.language == "en")).first()
        doc.is_active = False
        index = session.scalar(select(IndexVersion).where(IndexVersion.version == version))
        result = sync_passages(session, store, index)
        session.commit()
        total = session.scalar(select(func.count()).select_from(DocumentPassage))
    assert result["skipped_inactive_documents"] >= 1
    assert total == result["inserted"] == store.count - result["skipped_inactive_documents"]


def test_sync_refuses_passages_from_another_index_version(registered):
    factory, store, _ = registered
    with factory() as session:
        other = IndexVersion(
            version="idx-other", encoder_name="n", encoder_revision="r", corpus_sha256="a" * 64,
            manifest_sha256="b" * 64, file_hashes={}, document_count=1, per_language_counts={},
            vector_dimension=64, status="validated",
        )  # fmt: skip
        session.add(other)
        session.flush()
        with pytest.raises(PassageArtifactError):
            sync_passages(session, store, other)


def test_removing_the_artifact_directory_is_all_it_takes_to_discard_it(tmp_path):
    path = build(tmp_path)
    shutil.rmtree(path)
    assert not path.exists()
