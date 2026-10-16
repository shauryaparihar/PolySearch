"""A restart must serve the index the database says is active, not whatever the env var names."""

from __future__ import annotations

import pytest
from api_support import SEARCH
from api_support import test_settings as make_settings
from app.main import create_app
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder
from fastapi.testclient import TestClient
from job_support import KEY, add_document, get_job, start_index_job
from sqlalchemy import select


@pytest.fixture
def fake_encoder_on_load(monkeypatch):
    """``SearchBundle.load`` without an injected encoder would download the real model."""
    original = SearchBundle.load.__func__

    def load(cls, path, encoder=None):
        return original(cls, path, encoder=encoder or FakeEncoder())

    monkeypatch.setattr(SearchBundle, "load", classmethod(load))


def _restart(session_factory, configured_path, root):
    """A fresh process: no injected bundle, only the configured path/root and the database."""
    settings = make_settings(active_index_path=configured_path, artifact_root=root)
    client = TestClient(create_app(settings, session_factory=session_factory))
    client.__enter__()
    return client


def _active_version(factory) -> str:
    from app.db.models import IndexVersion

    with factory() as s:
        return s.scalar(select(IndexVersion.version).where(IndexVersion.status == "active"))


def test_restart_after_a_reindex_keeps_serving_the_activated_version(
    make_client, session_factory, api_bundle, fake_encoder_on_load
):
    admin = make_client(admin_api_key=KEY)
    add_document(session_factory, "Zebra Rayada", "es")
    new_version = get_job(admin, "index", start_index_job(admin))["index_version"]
    assert new_version != api_bundle.version
    admin.__exit__(None, None, None)

    # the environment still names the ORIGINAL index directory
    client = _restart(session_factory, api_bundle.path, api_bundle.path.parent)
    try:
        assert client.get("/api/v1/stats").json()["index_version"] == new_version
        assert _active_version(session_factory) == new_version  # not silently reverted
        assert client.get("/readyz").status_code == 200
        assert client.post("/api/v1/search", json=SEARCH).json()["index_version"] == new_version
    finally:
        client.__exit__(None, None, None)


def test_restart_falls_back_to_the_configured_path_when_the_active_files_are_gone(
    make_client, session_factory, api_bundle, fake_encoder_on_load
):
    import shutil

    admin = make_client(admin_api_key=KEY)
    add_document(session_factory, "Zebra Rayada", "es")
    new_version = get_job(admin, "index", start_index_job(admin))["index_version"]
    admin.__exit__(None, None, None)
    shutil.rmtree(api_bundle.path.parent / new_version)

    client = _restart(session_factory, api_bundle.path, api_bundle.path.parent)
    try:
        assert client.get("/api/v1/stats").json()["index_version"] == api_bundle.version
    finally:
        client.__exit__(None, None, None)


def test_restart_on_an_empty_database_uses_the_configured_path(
    session_factory, api_bundle, fake_encoder_on_load
):
    client = _restart(session_factory, api_bundle.path, api_bundle.path.parent)
    try:
        assert client.get("/api/v1/stats").json()["index_version"] == api_bundle.version
    finally:
        client.__exit__(None, None, None)
