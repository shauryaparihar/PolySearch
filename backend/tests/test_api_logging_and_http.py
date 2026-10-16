import json
import logging

import pytest
from api_support import SEARCH
from app.api.logging_config import LOGGER_NAME, JsonFormatter, log_event
from app.api.service import register_index_version
from app.config import Settings
from app.db.models import IndexVersion
from sqlalchemy.exc import IntegrityError


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[dict] = []
        self.setFormatter(JsonFormatter())

    def emit(self, record):
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def logs():
    handler = Capture()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


SECRET_QUERY = "zxq-confidential-volcano-phrase"


def test_logs_carry_request_ids_and_never_the_raw_query(client, logs):
    r = client.post("/api/v1/search", json={**SEARCH, "query": SECRET_QUERY})
    rid = r.headers["x-request-id"]
    raw = json.dumps(logs)
    assert SECRET_QUERY not in raw and "confidential" not in raw
    done = next(x for x in logs if x["event"] == "search_completed")
    assert done["request_id"] == rid and done["cache_hit"] is False
    assert done["result_count"] == 16 and done["mode"] == "dense"
    assert done["languages"] == ["es", "fr"] and done["index_version"].startswith("idx-")
    assert len(done["query_sha256"]) == 16 and "search_id" in done
    access = next(x for x in logs if x["event"] == "request")
    assert (
        access["route"] == "/api/v1/search"
        and access["status"] == 200
        and access["duration_ms"] >= 0
    )


def test_error_responses_and_logs_share_the_same_request_id(client, logs):
    r = client.post("/api/v1/search", json={**SEARCH, "query": "x" * 600})
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
    assert any(
        x["event"] == "request" and x["request_id"] == r.headers["x-request-id"] for x in logs
    )
    assert "xxxxxxxxxx" not in json.dumps(logs)


def test_unhandled_errors_return_a_generic_500_and_log_only_the_exception_class(
    client, logs, api_bundle
):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail: postgresql://user:password@host/db")

    api_bundle.search = boom
    safe = client.__class__(client.app, raise_server_exceptions=False)
    r = safe.post("/api/v1/search", json=SEARCH)
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "password" not in r.text and "postgresql" not in r.text
    assert "password" not in json.dumps(logs)
    assert any(x["event"] == "unhandled_error" and x["error_code"] == "RuntimeError" for x in logs)


def test_safe_client_request_ids_are_echoed_and_unsafe_ones_replaced(client):
    ok = client.get("/healthz", headers={"x-request-id": "client-req-12345"})
    assert ok.headers["x-request-id"] == "client-req-12345"
    bad = client.get("/healthz", headers={"x-request-id": "bad id\twith spaces<script>"})
    assert bad.headers["x-request-id"] != "bad id\twith spaces<script>"
    assert len(bad.headers["x-request-id"]) == 32


def test_log_event_refuses_fields_that_could_leak_data():
    with pytest.raises(ValueError):
        log_event("x", query="raw user text")
    with pytest.raises(ValueError):
        log_event("x", database_url="postgresql://...")


def test_cors_allows_only_the_configured_origin(make_client):
    client = make_client(cors_allowed_origins=["https://app.example.com"])
    ok = client.get("/healthz", headers={"origin": "https://app.example.com"})
    assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
    evil = client.get("/healthz", headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    preflight = client.options(
        "/api/v1/search",
        headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers


def test_error_responses_also_carry_cors_headers(make_client):
    client = make_client(
        search_rate_limit_per_minute=1, cors_allowed_origins=["https://app.example.com"]
    )
    headers = {"origin": "https://app.example.com"}
    client.post("/api/v1/search", json=SEARCH, headers=headers)
    limited = client.post("/api/v1/search", json=SEARCH, headers=headers)
    assert limited.status_code == 429
    assert limited.headers["access-control-allow-origin"] == "https://app.example.com"


@pytest.mark.parametrize("origin", ["*", "", "app.example.com", "ftp://x"])
def test_wildcard_or_malformed_origins_are_rejected_at_startup(origin):
    with pytest.raises(ValueError):
        Settings(_env_file=None, cors_allowed_origins=[origin])


def test_only_one_index_version_can_be_active(session_factory, api_bundle):
    with session_factory() as s:
        first = register_index_version(s, api_bundle)
        s.commit()
        assert first.status == "active"
        extra = IndexVersion(
            version="idx-other", encoder_name="e", encoder_revision="r", corpus_sha256="c" * 64,
            manifest_sha256="m" * 64, file_hashes={}, document_count=1, per_language_counts={},
            vector_dimension=1, status="active",
        )  # fmt: skip
        s.add(extra)
        with pytest.raises(IntegrityError):  # the database forbids a second active row
            s.flush()
        s.rollback()


def test_registering_a_new_version_retires_the_previous_one(
    session_factory, api_bundle, tmp_path, encoder
):

    with session_factory() as s:
        old = register_index_version(s, api_bundle)
        s.commit()
        api_bundle.manifest["version"] = "idx-newer"
        new = register_index_version(s, api_bundle)
        s.commit()
        s.refresh(old)
        assert (old.status, new.status) == ("retired", "active")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/search"),
        ("POST", "/api/v1/searches/abc/clicks"),
        ("PUT", "/api/v1/searches/abc/feedback/def"),  # browsers preflight PUT before sending it
        ("GET", "/api/v1/suggestions"),
    ],
)
def test_browser_preflight_is_allowed_for_every_method_the_ui_uses(make_client, method, path):
    client = make_client(cors_allowed_origins=["https://app.example.com"])
    r = client.options(
        path,
        headers={
            "origin": "https://app.example.com",
            "access-control-request-method": method,
            "access-control-request-headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://app.example.com"
    assert method in r.headers["access-control-allow-methods"]


def test_methods_the_ui_never_uses_are_not_allowed_by_cors(make_client):
    client = make_client(cors_allowed_origins=["https://app.example.com"])
    r = client.options(
        "/api/v1/search",
        headers={"origin": "https://app.example.com", "access-control-request-method": "DELETE"},
    )
    assert r.status_code == 400  # preflight refused
