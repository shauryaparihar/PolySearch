import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from api_support import SEARCH
from app.api.analytics import build_report
from app.api.logging_config import LOGGER_NAME, JsonFormatter
from app.db.models import (
    ClickEvent,
    Document,
    IndexVersion,
    RelevanceFeedback,
    SearchResult,
    SearchRun,
)
from sqlalchemy import func, select, text

KEY = "unit-test-admin-key-123"
HEAD = {"x-admin-key": KEY}


@pytest.fixture
def admin(make_client):
    return make_client(admin_api_key=KEY)


def _search(client, **kw):
    return client.post("/api/v1/search", json={**SEARCH, "page_size": 5, **kw}).json()


def _report(client, days=7):
    r = client.get("/api/v1/admin/analytics", params={"days": days}, headers=HEAD)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- admin access
def test_admin_api_does_not_exist_when_no_key_is_configured(client):
    r = client.get("/api/v1/admin/analytics", headers=HEAD)
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


def test_missing_wrong_and_malformed_keys_are_indistinguishable(admin):
    bodies = []
    for headers in (
        {},
        {"x-admin-key": "wrong"},
        {"x-admin-key": b"\xe9" * 40},
        {"x-admin-key": KEY + "x"},
    ):
        r = admin.get("/api/v1/admin/analytics", headers=headers)
        assert r.status_code == 401
        error = r.json()["error"]
        bodies.append((error["code"], error["message"], json.dumps(error["details"])))
    assert len(set(bodies)) == 1 and bodies[0][0] == "UNAUTHORIZED"


def test_the_correct_key_is_accepted(admin):
    assert admin.get("/api/v1/admin/analytics", headers=HEAD).status_code == 200


def test_the_key_is_compared_in_constant_time(admin, monkeypatch):
    calls = []
    import hmac

    real = hmac.compare_digest
    monkeypatch.setattr(
        "app.api.admin.hmac.compare_digest", lambda a, b: calls.append(1) or real(a, b)
    )
    admin.get("/api/v1/admin/analytics", headers={"x-admin-key": "nope"})
    assert calls == [1]


def test_failed_attempts_are_rate_limited_and_the_key_never_appears_in_output(make_client):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(JsonFormatter().format(record))

    handler = Capture()
    logging.getLogger(LOGGER_NAME).addHandler(handler)
    try:
        client = make_client(admin_api_key=KEY, admin_rate_limit_per_minute=3)
        codes = [
            client.get("/api/v1/admin/analytics", headers={"x-admin-key": "bad"}).status_code
            for _ in range(3)
        ]
        blocked = client.get(
            "/api/v1/admin/analytics", headers=HEAD
        )  # even the right key is throttled
    finally:
        logging.getLogger(LOGGER_NAME).removeHandler(handler)
    assert codes == [401, 401, 401] and blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert KEY not in blocked.text and KEY not in json.dumps(records)
    assert any("admin_auth_failed" in r for r in records)


@pytest.mark.parametrize("days", [0, 91, "x"])
def test_the_reporting_window_is_bounded(admin, days):
    assert (
        admin.get("/api/v1/admin/analytics", params={"days": days}, headers=HEAD).status_code == 422
    )


# ------------------------------------------------------ reconciliation with rows
def test_report_reconciles_with_the_database_rows(admin, session_factory):
    first = _search(admin)
    _search(admin)  # cache hit
    _search(admin, query="Glacier", target_languages=["es"], mode="lexical")
    sid, shown = first["search_id"], first["results"]
    admin.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": shown[0]["document_id"]})
    admin.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": shown[0]["document_id"]})
    admin.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": shown[1]["document_id"]})
    admin.put(
        f"/api/v1/searches/{sid}/feedback/{shown[0]['document_id']}", json={"label": "relevant"}
    )
    admin.put(
        f"/api/v1/searches/{sid}/feedback/{shown[1]['document_id']}", json={"label": "not_relevant"}
    )
    report = _report(admin)

    with session_factory() as s:
        runs = s.scalar(select(func.count()).select_from(SearchRun))
        hits = s.scalar(select(func.count()).select_from(SearchRun).where(SearchRun.cache_hit))
        shown_rows = s.scalar(
            select(func.count()).select_from(SearchResult).where(SearchResult.shown_at.is_not(None))
        )
        events = s.scalar(select(func.count()).select_from(ClickEvent))
        clicked = s.scalar(
            text(
                "select count(*) from "
                "(select distinct search_run_id, document_id from click_events) x"
            )
        )
        labels = dict(
            s.execute(
                select(RelevanceFeedback.label, func.count()).group_by(RelevanceFeedback.label)
            ).all()
        )
    assert report["searches"]["total"] == runs == 3 and report["searches"]["ok"] == 3
    assert (
        report["searches"]["cache_hit_rate"]
        == round(hits / runs, 6)
        == pytest.approx(1 / 3, abs=1e-6)
    )
    assert report["searches"]["zero_result_rate"] == 0.0
    assert report["engagement"]["impressions"] == shown_rows == 15
    assert (
        (report["engagement"]["click_events"], report["engagement"]["clicked_results"])
        == (events, clicked)
        == (3, 2)
    )
    assert report["engagement"]["ctr"] == round(2 / 15, 6)
    assert (
        report["feedback"]
        == {"relevant": labels["relevant"], "not_relevant": labels["not_relevant"]}
        == {"relevant": 1, "not_relevant": 1}
    )
    assert report["distribution"]["mode"] == {"dense": 2, "lexical": 1}
    assert report["distribution"]["query_language"] == {"en": 3}
    assert report["distribution"]["target_language"] == {"es": 3, "fr": 2}
    assert report["distribution"]["source"] == {"wikipedia": 3}
    assert sum(report["distribution"]["index_version"].values()) == 3
    assert sum(d["count"] for d in report["searches"]["per_day"]) == 3
    assert (
        report["latency_ms"]["median"] > 0
        and report["latency_ms"]["p95"] >= report["latency_ms"]["median"]
    )


def test_ctr_is_calculated_correctly_on_a_hand_built_fixture(session_factory, make_client):
    make_client()  # registers the index version and seeds documents
    with session_factory() as s:
        version = s.scalar(select(IndexVersion.id))
        docs = s.scalars(select(Document.id).order_by(Document.external_id).limit(4)).all()

        def run(shown_ranks):
            r = SearchRun(
                id=uuid.uuid4(), query_sha256="a" * 64, query_language="en",
                target_languages=["es"], sources=["wikipedia"], requested_mode="dense",
                index_version_id=version, result_pool_size=4, total_ms=1, status="ok",
            )  # fmt: skip
            s.add(r)
            s.flush()
            now = datetime.now(UTC)
            s.add_all(
                SearchResult(search_run_id=r.id, document_id=docs[i], rank=i + 1, score=0,
                             shown_at=now if i + 1 in shown_ranks else None)
                for i in range(4)
            )  # fmt: skip
            s.flush()
            return r

        a = run({1, 2, 3})
        run({1, 2})  # 5 impressions in total
        for run_, doc, n in ((a, docs[0], 2), (a, docs[1], 1)):  # rank 1 clicked twice, rank 2 once
            for _ in range(n):
                s.add(
                    ClickEvent(
                        search_run_id=run_.id, document_id=doc, rank=1 if doc == docs[0] else 2
                    )
                )
        s.commit()
        report = build_report(s, 7, {"search": 0}, False, 3)
    e = report["engagement"]
    assert (e["impressions"], e["clicked_results"], e["click_events"]) == (5, 2, 3)
    assert e["ctr"] == 0.4  # 2 distinct clicked results / 5 impressions
    by_rank = {row["rank"]: row for row in e["ctr_by_rank"]}
    assert (by_rank[1]["impressions"], by_rank[1]["clicked_results"], by_rank[1]["ctr"]) == (
        2,
        1,
        0.5,
    )
    assert (by_rank[2]["impressions"], by_rank[2]["clicked_results"], by_rank[2]["ctr"]) == (
        2,
        1,
        0.5,
    )
    assert (by_rank[3]["impressions"], by_rank[3]["ctr"]) == (1, 0.0) and 4 not in by_rank


def test_zero_result_rate_and_empty_windows(session_factory, make_client):
    client = make_client(admin_api_key=KEY)
    assert _report(client)["searches"] == {
        "total": 0,
        "ok": 0,
        "error": 0,
        "per_day": [],
        "cache_hit_rate": None,
        "zero_result_rate": None,
    }
    _search(client)
    with session_factory() as s:
        s.execute(text("update search_runs set result_pool_size = 0"))
        s.commit()
    assert _report(client)["searches"]["zero_result_rate"] == 1.0


def test_old_searches_fall_outside_the_window(admin, session_factory):
    _search(admin)
    with session_factory() as s:
        s.execute(
            text("update search_runs set created_at = :t"),
            {"t": datetime.now(UTC) - timedelta(days=10)},
        )
        s.commit()
    assert _report(admin, days=7)["searches"]["total"] == 0
    assert _report(admin, days=30)["searches"]["total"] == 1


def test_rate_limited_requests_are_counted_since_process_start(make_client):
    client = make_client(admin_api_key=KEY, search_rate_limit_per_minute=1)
    _search(client)
    client.post("/api/v1/search", json=SEARCH)
    client.post("/api/v1/search", json=SEARCH)
    assert _report(client)["rate_limited_since_process_start"]["search"] == 2


def test_query_text_is_absent_from_analytics_unless_enabled(admin, session_factory):
    for _ in range(3):
        _search(admin, query="private climate question")
    report = _report(admin)
    assert report["top_queries"]["text"] is None
    assert "private climate question" not in json.dumps(report)
    assert report["top_queries"]["by_hash"][0]["count"] == 3
    assert len(report["top_queries"]["by_hash"][0]["query_sha256_prefix"]) == 12
    with session_factory() as s:
        assert (
            s.scalar(
                select(func.count())
                .select_from(SearchRun)
                .where(SearchRun.analytics_query_text.is_not(None))
            )
            == 0
        )


def test_top_query_text_needs_the_flag_a_minimum_count_and_no_sensitive_text(make_client):
    client = make_client(
        admin_api_key=KEY, analytics_store_query_text=True, popular_query_min_count=3
    )
    for _ in range(3):
        _search(client, query="popular climate query")
        _search(client, query="write to jane@example.com")
    _search(client, query="rare one off query")
    top = _report(client)["top_queries"]["text"]
    assert top == [{"query": "popular climate query", "count": 3}]
