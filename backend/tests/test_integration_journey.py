"""The whole product path, through the public HTTP API, on a real PostgreSQL database.

ingest (fake MediaWiki) -> reindex (fake encoder artifacts) -> search -> page -> document ->
click -> feedback -> analytics reconciled against raw rows -> restart -> the data survives.
The only fakes are the network (MediaWiki) and the transformer; everything else is real.
"""

from __future__ import annotations

import pytest
from api_support import test_settings as make_settings
from app.config import Settings
from app.db.models import ClickEvent, Document, RelevanceFeedback, SearchResult, SearchRun
from app.ingestion.wikipedia import WikipediaSource
from app.main import create_app
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder
from fastapi.testclient import TestClient
from job_support import HEAD, KEY, get_job, start_index_job
from sqlalchemy import func, select

QUERY = {"query": "Medicine", "query_language": "en", "target_languages": ["es", "fr"]}


@pytest.fixture
def wiki_client(make_client, fake_wiki):
    def source_factory():
        return WikipediaSource(
            Settings(_env_file=None, http_max_retries=0),
            transport=fake_wiki.transport(),
            sleep=lambda _s: None,
            jitter=lambda: 0.0,
        )

    return make_client(admin_api_key=KEY, source_factory=source_factory)


def _ingest(client, language, titles):
    r = client.post(
        "/api/v1/admin/ingestion-jobs",
        json={"language": language, "mode": "seed", "titles": titles},
        headers=HEAD,
    )
    assert r.status_code == 202, r.text
    job = get_job(client, "ingestion", r.json()["job_id"])
    assert job["status"] == "completed", job
    return job


def _count(factory, model, *where) -> int:
    with factory() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


def test_the_complete_user_journey(wiki_client, session_factory, api_bundle):
    client = wiki_client
    assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200
    original = api_bundle.version
    assert client.post("/api/v1/search", json=QUERY).json()["index_version"] == original

    # 1. ingestion adds a Spanish article that the current index has never seen
    assert _ingest(client, "es", ["Medicina"])["counters"]["created"] == 1
    documents_before_reindex = _count(session_factory, Document)

    # 2. a reindex builds, validates and activates a new immutable version
    job = get_job(client, "index", start_index_job(client))
    assert (job["status"], job["step"]) == ("completed", "activated")
    new_version = job["index_version"]
    assert new_version != original and job["documents_total"] == documents_before_reindex
    assert client.get("/readyz").status_code == 200

    # 3. search finds the new article (cross-language: English query, Spanish page)
    first = client.post("/api/v1/search", json=QUERY).json()
    assert first["index_version"] == new_version and first["cache_hit"] is False
    top = first["results"][0]
    assert (top["title"], top["language"], top["rank"]) == ("Medicina", "es", 1)
    assert top["source_url"].startswith("https://es.wikipedia.org/")
    assert first["attribution"]["wikipedia"]["licence_name"].startswith("Creative Commons")
    assert client.post("/api/v1/search", json=QUERY).json()["cache_hit"] is True

    # 4. pagination walks one stored pool without repeats
    big = {**QUERY, "page_size": 5}
    page1 = client.post("/api/v1/search", json=big).json()
    page2 = client.get(
        f"/api/v1/searches/{page1['search_id']}/results",
        params={"cursor": page1["next_cursor"], "limit": 5},
    ).json()
    ids = [r["document_id"] for r in page1["results"] + page2["results"]]
    assert len(ids) == len(set(ids)) == 10
    assert [r["rank"] for r in page2["results"]] == list(range(6, 11))

    # 5. document detail, click, duplicate click, feedback
    detail = client.get(f"/api/v1/documents/{top['document_id']}").json()
    assert detail["title"] == "Medicina" and detail["source_revision"] == "21"
    sid = first["search_id"]
    click = client.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": top["document_id"]})
    assert click.status_code == 201 and click.json()["rank"] == 1
    client.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": top["document_id"]})
    fb = client.put(
        f"/api/v1/searches/{sid}/feedback/{top['document_id']}", json={"label": "relevant"}
    )
    assert fb.status_code == 200 and fb.json()["label"] == "relevant"

    # 6. analytics agree with the raw rows
    report = client.get("/api/v1/admin/analytics", headers=HEAD).json()
    runs = _count(session_factory, SearchRun)
    shown = _count(session_factory, SearchResult, SearchResult.shown_at.is_not(None))
    assert report["searches"]["total"] == runs == 4 and report["searches"]["ok"] == 4
    assert report["engagement"]["impressions"] == shown
    assert report["engagement"]["click_events"] == _count(session_factory, ClickEvent) == 2
    assert report["engagement"]["clicked_results"] == 1
    assert report["feedback"] == {"relevant": 1, "not_relevant": 0}
    assert _count(session_factory, RelevanceFeedback) == 1
    assert report["distribution"]["index_version"].get(new_version, 0) >= 3
    assert "Medicine" not in str(report)  # raw query text is not stored by default

    # 7. a restart (new process: only the config and the database survive) keeps all of it
    client.__exit__(None, None, None)
    restarted = TestClient(
        create_app(
            make_settings(
                admin_api_key=KEY,
                active_index_path=api_bundle.path,  # the env still names the ORIGINAL index
                artifact_root=api_bundle.path.parent,
            ),
            session_factory=session_factory,
            bundle=SearchBundle.load(api_bundle.path.parent / new_version, encoder=FakeEncoder()),
        )
    )
    restarted.__enter__()
    try:
        assert restarted.get("/readyz").status_code == 200
        assert restarted.get("/api/v1/stats").json()["index_version"] == new_version
        again = restarted.get(
            f"/api/v1/searches/{page1['search_id']}/results",
            params={"cursor": page1["next_cursor"], "limit": 5},
        )
        assert again.status_code == 200  # cursors and stored pools outlive the process
        assert [r["document_id"] for r in again.json()["results"]] == [
            r["document_id"] for r in page2["results"]
        ]
        after = restarted.get("/api/v1/admin/analytics", headers=HEAD).json()
        assert after["engagement"]["click_events"] == 2
    finally:
        restarted.__exit__(None, None, None)


def test_a_failed_reindex_leaves_the_journey_working_on_the_old_index(
    wiki_client, session_factory, api_bundle, monkeypatch
):
    client = wiki_client
    _ingest(client, "es", ["Medicina"])
    from app.jobs import reindex

    def explode(*args, **kwargs):
        raise RuntimeError("simulated encoder crash")

    monkeypatch.setattr(reindex, "build_index", explode)
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "failed" and "simulated" not in str(job)  # safe error code only
    assert client.get("/readyz").status_code == 200
    result = client.post("/api/v1/search", json=QUERY).json()
    assert result["index_version"] == api_bundle.version  # still the old, working index
    assert result["results"], "search keeps answering"
    assert all(r["title"] != "Medicina" for r in result["results"])  # new data not yet searchable
    click = client.post(
        f"/api/v1/searches/{result['search_id']}/clicks",
        json={"document_id": result["results"][0]["document_id"]},
    )
    assert click.status_code == 201
