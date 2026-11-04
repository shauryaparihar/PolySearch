"""``build_passages.py``: builds passages for an existing index, refuses a different corpus."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from api_support import docs, seed_documents
from app.api.service import register_index_version
from app.db.models import DocumentPassage
from app.rag.passage_store import PassageStore
from app.search.artifacts import build_index
from conftest import TEST_URL
from fake_encoder import FakeEncoder
from sqlalchemy import func, select

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_passages.py"


@pytest.fixture
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_passages_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SentenceTransformerEncoder", lambda *a, **k: FakeEncoder())
    return module


def _index_and_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    from app.evaluation.corpus import _line, load_corpus

    rows = [
        {"key": d.key, "language": d.language, "title": d.title, "summary": d.summary,
         "source_url": "https://x", "source_revision": "1", "content_sha256": "0" * 64}
        for d in docs()
    ]  # fmt: skip
    corpus_path.write_text("".join(_line(r) + "\n" for r in rows), encoding="utf-8")
    index = build_index(load_corpus(corpus_path), FakeEncoder(), tmp_path / "index")
    return index, corpus_path


def test_it_builds_validates_and_reports_without_printing_secrets(cli, tmp_path, capsys):
    index, corpus_path = _index_and_corpus(tmp_path)
    assert cli.main(["--index-dir", str(index), "--corpus", str(corpus_path)]) == 0
    out = capsys.readouterr().out
    assert "passages" in out and "per language" in out and "limit 600" in out
    store = PassageStore.load(tmp_path / "index" / "passages" / index.name)
    assert store.count >= len(docs()) and store.index_version == index.name
    assert TEST_URL not in out


def test_it_refuses_a_corpus_the_index_was_not_built_from(cli, tmp_path, capsys):
    index, corpus_path = _index_and_corpus(tmp_path)
    other = tmp_path / "other.jsonl"
    first = json.loads(corpus_path.read_text().splitlines()[0])
    first["summary"] = "A different summary entirely."
    other.write_text(json.dumps(first) + "\n", encoding="utf-8")
    assert cli.main(["--index-dir", str(index), "--corpus", str(other)]) == 2
    assert "differs" in capsys.readouterr().err
    assert not (tmp_path / "index" / "passages").exists()


def test_sync_db_requires_the_version_to_be_registered_then_writes_rows(
    cli, tmp_path, session_factory, monkeypatch, capsys
):
    from app.config import get_settings

    seed_documents(session_factory)
    index, corpus_path = _index_and_corpus(tmp_path)
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    get_settings.cache_clear()
    try:
        args = ["--index-dir", str(index), "--corpus", str(corpus_path), "--sync-db"]
        assert cli.main(args) == 3  # built, but the index version is not in the database yet
        assert "not registered" in capsys.readouterr().err
        from app.search.bundle import SearchBundle

        bundle = SearchBundle.load(index, encoder=FakeEncoder())
        with session_factory() as session:
            register_index_version(session, bundle)
            session.commit()
        assert cli.main(args) == 0
        with session_factory() as session:
            stored = session.scalar(select(func.count()).select_from(DocumentPassage))
        assert stored >= len(docs()) and "inserted" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_a_second_build_reuses_the_vectors_of_the_first(cli, tmp_path, capsys):
    index, corpus_path = _index_and_corpus(tmp_path)
    args = ["--index-dir", str(index), "--corpus", str(corpus_path)]
    assert cli.main(args) == 0
    assert "reused 0 vectors" in capsys.readouterr().out
    assert cli.main(args) == 0
    again = capsys.readouterr().out
    assert "embedded 0," in again and "reused 0 vectors" not in again
