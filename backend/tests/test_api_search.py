import re

import pytest
from api_support import SEARCH
from app.db.models import Document, IndexVersion, SearchResult, SearchRun
from sqlalchemy import func, select, update


def _post(client, **overrides):
    return client.post("/api/v1/search", json={**SEARCH, **overrides})


def test_search_returns_ranked_results_with_provenance_and_attribution(client):
    r = _post(client, page_size=5)
    assert r.status_code == 200
    body = r.json()
    assert [x["rank"] for x in body["results"]] == [1, 2, 3, 4, 5]
    top = body["results"][0]
    assert {"es:Volcán", "fr:Volcan"} >= {f"{top['language']}:{top['title']}"}  # cross-language hit
    assert top["source"] == "wikipedia"
    assert top["source_url"].startswith("https://") and top["source_revision"] == "42"
    licence = body["attribution"]["wikipedia"]
    assert licence["licence_name"].startswith("Creative Commons")
    assert licence["licence_url"].startswith("https://creativecommons.org/")
    assert body["index_version"].startswith("idx-") and body["total_results"] == 16
    assert r.headers["x-request-id"]


def test_default_mode_comes_from_the_index_manifest(make_client, api_bundle):
    api_bundle.manifest["default_mode"] = {"default_mode": "hybrid", "reason": "test"}
    assert _post(make_client()).json()["mode"] == "hybrid"
    api_bundle.manifest["default_mode"] = None
    assert _post(make_client()).json()["mode"] == "dense"


def test_language_and_mode_filters_are_applied(client):
    es_only = _post(client, target_languages=["es"], mode="lexical").json()
    assert {x["language"] for x in es_only["results"]} == {"es"}
    assert all(x["dense_rank"] is None for x in es_only["results"])
    hybrid = _post(client, mode="hybrid").json()
    assert hybrid["results"][0]["dense_rank"] and hybrid["results"][0]["lexical_rank"]


def test_pool_is_stored_with_a_hash_and_without_raw_query_text(client, session_factory):
    body = _post(client, query="  Volcano   lava  ").json()
    with session_factory() as s:
        run = s.get(SearchRun, body["search_id"])
        assert len(run.query_sha256) == 64 and run.analytics_query_text is None
        assert run.status == "ok" and run.cache_hit is False and run.requested_mode == "dense"
        assert sorted(run.target_languages) == ["es", "fr"]
        assert float(run.total_ms) > 0 and run.encoding_ms is not None
        ranks = s.scalars(
            select(SearchResult.rank).where(SearchResult.search_run_id == run.id).order_by("rank")
        ).all()
        assert ranks == list(range(1, 17))
        active = s.scalar(select(IndexVersion).where(IndexVersion.status == "active"))
        assert run.index_version_id == active.id


def test_query_text_is_stored_only_when_explicitly_enabled(make_client, session_factory):
    client = make_client(analytics_store_query_text=True)
    sid = _post(client, query="  Volcano   lava  ").json()["search_id"]
    with session_factory() as s:
        assert s.get(SearchRun, sid).analytics_query_text == "Volcano lava"  # normalised


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"query": "a"}, "QUERY_INVALID"),
        ({"query": "x" * 501}, "QUERY_INVALID"),
        ({"query": "bad\x00char"}, "QUERY_INVALID"),
        ({"query": "   "}, "QUERY_INVALID"),
        ({"target_languages": []}, "FILTER_INVALID"),
        ({"target_languages": ["es", "es"]}, "FILTER_INVALID"),
        ({"target_languages": ["es", "fr", "en", "en"]}, "FILTER_INVALID"),
        ({"target_languages": ["de"]}, "FILTER_INVALID"),
        ({"query_language": "xx"}, "FILTER_INVALID"),
        ({"sources": ["medium"]}, "FILTER_INVALID"),
        ({"sources": ["wikipedia", "wikipedia"]}, "FILTER_INVALID"),
        ({"mode": "magic"}, "MODE_INVALID"),
        ({"page_size": 4}, "PAGE_SIZE_INVALID"),
        ({"page_size": 21}, "PAGE_SIZE_INVALID"),
        ({"admin": True}, "REQUEST_INVALID"),
    ],
)
def test_invalid_requests_fail_safely_without_echoing_input(client, override, code):
    r = _post(client, **override)
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == code and error["request_id"] == r.headers["x-request-id"]
    assert "Traceback" not in r.text and "sqlalchemy" not in r.text.lower()
    submitted = str(override.get("query", ""))
    if len(submitted) > 5:
        assert submitted not in r.text  # the offending value is never reflected


def test_malformed_json_and_missing_content_type_are_rejected(client):
    assert client.post("/api/v1/search", content="{not json").status_code == 422
    r = client.post(
        "/api/v1/search", content="{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "REQUEST_INVALID"


def test_oversized_bodies_are_refused_before_processing(make_client):
    client = make_client(max_request_body_bytes=512)
    big = '{"query": "' + "x" * 2000 + '"}'
    declared = client.post(
        "/api/v1/search", content=big, headers={"content-type": "application/json"}
    )
    assert declared.status_code == 413 and declared.json()["error"]["code"] == "REQUEST_TOO_LARGE"

    def chunks():  # no Content-Length: sent chunked
        for i in range(0, len(big), 200):
            yield big[i : i + 200].encode()

    chunked = client.post(
        "/api/v1/search", content=chunks(), headers={"content-type": "application/json"}
    )
    assert chunked.status_code == 413


def test_snippets_and_titles_are_bounded_plain_text_even_if_the_database_holds_html(
    client, session_factory
):
    hostile = (
        "<script>alert(1)</script><img src=x onerror=alert(2)> Texto <b>claro</b> "
        "<scr<script>ipt and a lone <script src=//evil " + "palabra " * 200
    )
    with session_factory() as s:
        s.execute(
            update(Document)
            .where(Document.external_id.in_(["Volcán", "Volcan"]))
            .values(summary=hostile, title="<b>Volc</b><script>x</script>án")
        )
        s.commit()
    body = _post(client).json()
    tag_start = re.compile(r"<[A-Za-z/!?]")
    for item in body["results"]:
        for field in ("title", "snippet"):
            assert not tag_start.search(item[field]), (field, item[field][:80])
        assert len(item["snippet"]) <= 300
    top = next(x for x in body["results"] if x["source_url"].endswith("/Volcán"))
    assert top["snippet"].startswith("alert(1) Texto claro")  # tags gone, text kept as plain text
    assert top["title"] == "Volcxán"  # <b>/<script> removed; the inner text stays as inert text


def test_document_detail_and_not_found(client):
    item = _post(client).json()["results"][0]
    r = client.get(f"/api/v1/documents/{item['document_id']}")
    assert r.status_code == 200 and r.json()["attribution"]["licence_url"].startswith("https://")
    assert client.get("/api/v1/documents/not-a-uuid").status_code == 404
    missing = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
    assert missing.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_stats_exposes_only_public_summary(client):
    stats = client.get("/api/v1/stats").json()
    assert stats["documents_per_language"] == {"en": 8, "es": 8, "fr": 8}
    assert stats["index_version"].startswith("idx-") and stats["modes"] == [
        "dense",
        "lexical",
        "hybrid",
    ]
    text = str(stats)
    assert "sha256" not in text and "/tmp" not in text and "/private" not in text


def test_every_search_is_persisted_and_counted(client, session_factory):
    for _ in range(3):
        _post(client)
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(SearchRun)) == 3
