"""A database outage must give clean 503s, honest readiness and no SQL or parameters in logs."""

from __future__ import annotations

import logging

import pytest
from api_support import SEARCH
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout

SECRET_SQL = "INSERT INTO search_runs (query_sha256) VALUES ('top-secret-parameter')"


def _boom(*_args, **_kwargs):
    raise OperationalError(SECRET_SQL, {"p": "top-secret-parameter"}, Exception("connection lost"))


@pytest.fixture
def down(client, monkeypatch):
    """The same running app, but every database session now fails."""
    monkeypatch.setattr(client.app.state.polysearch, "session_factory", _boom)
    return client


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/v1/search", SEARCH),
        ("get", "/api/v1/documents/00000000-0000-0000-0000-000000000000", None),
        ("get", "/api/v1/searches/00000000-0000-0000-0000-000000000000/results", None),
        ("post", "/api/v1/searches/00000000-0000-0000-0000-000000000000/clicks",
         {"document_id": "00000000-0000-0000-0000-000000000000"}),
    ],
)  # fmt: skip
def test_database_errors_become_503_without_leaking_anything(down, caplog, method, path, body):
    caplog.set_level(logging.DEBUG)
    response = getattr(down, method)(path, **({"json": body} if body else {}))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    error = response.json()["error"]
    assert error["code"] == "DATABASE_UNAVAILABLE" and error["request_id"]
    assert "top-secret-parameter" not in response.text and "INSERT" not in response.text
    assert "top-secret-parameter" not in caplog.text and "INSERT" not in caplog.text
    assert "unhandled_error" not in caplog.text  # it is a handled, expected failure mode


def test_pool_exhaustion_is_also_a_503(client, monkeypatch):
    def exhausted(*_a, **_k):
        raise PoolTimeout("QueuePool limit of size 5 overflow 10 reached")

    monkeypatch.setattr(client.app.state.polysearch, "session_factory", exhausted)
    response = client.post("/api/v1/search", json=SEARCH)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_liveness_stays_up_but_readiness_notices_immediately(client, monkeypatch):
    """Readiness results are cached; a lost database must still be reported at once."""
    state = client.app.state.polysearch
    state.settings.readiness_cache_seconds = 600
    assert client.get("/readyz").status_code == 200  # fills the cache
    monkeypatch.setattr(state, "session_factory", _boom)
    assert client.get("/healthz").status_code == 200  # the process itself is fine
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["database"] is False
    monkeypatch.undo()
    assert client.get("/readyz").status_code == 200  # recovers on its own


def test_recovery_needs_no_restart(client, monkeypatch):
    state = client.app.state.polysearch
    original = state.session_factory
    monkeypatch.setattr(state, "session_factory", _boom)
    assert client.post("/api/v1/search", json=SEARCH).status_code == 503
    monkeypatch.setattr(state, "session_factory", original)
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200
