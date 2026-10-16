"""`build_artifacts.py --from-database` creates the first index of a fresh environment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from api_support import docs, seed_documents
from app.config import get_settings
from app.search.bundle import SearchBundle
from conftest import TEST_URL
from fake_encoder import FakeEncoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_artifacts.py"


def test_the_first_index_is_built_from_the_database_it_will_serve(
    session_factory, tmp_path, monkeypatch, capsys
):
    seed_documents(session_factory)
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    get_settings.cache_clear()
    spec = importlib.util.spec_from_file_location("build_artifacts_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SentenceTransformerEncoder", lambda *a, **k: FakeEncoder())
    out, corpus = tmp_path / "index", tmp_path / "corpus.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["x", "--from-database", "--corpus", str(corpus), "--out", str(out)]
    )
    try:
        assert module.main() == 0
    finally:
        get_settings.cache_clear()
    text = capsys.readouterr().out
    assert f"exported {len(docs())} active documents" in text
    (version_dir,) = [p for p in out.iterdir() if p.name.startswith("idx-")]
    bundle = SearchBundle.load(version_dir, encoder=FakeEncoder())
    assert len(bundle.document_keys()) == len(docs())  # index == database, document for document
    assert TEST_URL not in text and "polysearch_dev" not in text  # no credentials printed
