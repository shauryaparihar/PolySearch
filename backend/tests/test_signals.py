import threading
import uuid

import pytest
from api_support import SEARCH
from app.db.models import ClickEvent, RelevanceFeedback, SearchResult
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError


def _search(client, **kw):
    return client.post("/api/v1/search", json={**SEARCH, "page_size": 5, **kw}).json()


def _click(client, sid, doc, **body):
    return client.post(f"/api/v1/searches/{sid}/clicks", json={"document_id": doc, **body})


def _feedback(client, sid, doc, label):
    return client.put(f"/api/v1/searches/{sid}/feedback/{doc}", json={"label": label})


def _rows(factory, model):
    with factory() as s:
        return s.scalars(select(model)).all()


def test_click_on_a_shown_result_records_the_server_side_rank(client, session_factory):
    page = _search(client)
    doc = page["results"][2]["document_id"]
    r = _click(client, page["search_id"], doc)
    assert r.status_code == 201 and r.json()["rank"] == 3 and r.json()["duplicate"] is False
    [event] = _rows(session_factory, ClickEvent)
    assert (event.rank, str(event.document_id)) == (3, doc)


def test_the_client_cannot_choose_the_rank_or_send_unknown_fields(client):
    page = _search(client)
    r = _click(client, page["search_id"], page["results"][0]["document_id"], rank=1)
    assert r.status_code == 422


def test_idempotency_key_prevents_duplicate_events(client, session_factory):
    page = _search(client)
    sid, doc = page["search_id"], page["results"][0]["document_id"]
    first = _click(client, sid, doc, idempotency_key="browser-event-0001")
    again = _click(client, sid, doc, idempotency_key="browser-event-0001")
    assert (first.status_code, again.status_code) == (201, 200)
    assert (
        again.json()["duplicate"] is True and again.json()["click_id"] == first.json()["click_id"]
    )
    assert len(_rows(session_factory, ClickEvent)) == 1
    _click(client, sid, doc, idempotency_key="browser-event-0002")  # a genuinely new event
    assert len(_rows(session_factory, ClickEvent)) == 2


def test_concurrent_retries_with_one_key_create_a_single_event(client, session_factory):
    from fastapi.testclient import TestClient

    page = _search(client)
    sid, doc = page["search_id"], page["results"][0]["document_id"]
    codes = []

    def fire():
        with TestClient(client.app) as c:
            codes.append(_click(c, sid, doc, idempotency_key="race-event-0001").status_code)

    threads = [threading.Thread(target=fire) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes).count(201) == 1 and set(codes) <= {200, 201}
    assert len(_rows(session_factory, ClickEvent)) == 1


def test_invalid_idempotency_keys_are_rejected(client):
    page = _search(client)
    for key in ("short", "has spaces in it", "x" * 65, "bad;chars-here"):
        r = _click(
            client, page["search_id"], page["results"][0]["document_id"], idempotency_key=key
        )
        assert r.status_code == 422, key


def test_clicks_on_unrelated_or_unshown_documents_are_rejected(client, session_factory):
    page = _search(client)
    sid = page["search_id"]
    unrelated = _click(client, sid, str(uuid.uuid4()))
    assert unrelated.status_code == 404 and unrelated.json()["error"]["code"] == "RESULT_NOT_FOUND"
    with session_factory() as s:  # stored in the pool, but its page was never returned
        hidden = s.scalars(
            select(SearchResult).where(SearchResult.search_run_id == sid, SearchResult.rank == 12)
        ).one()
    assert _click(client, sid, str(hidden.document_id)).status_code == 404
    client.get(
        f"/api/v1/searches/{sid}/results", params={"cursor": page["next_cursor"], "limit": 20}
    )
    assert _click(client, sid, str(hidden.document_id)).status_code == 201  # now shown


def test_unknown_searches_and_bad_ids(client):
    doc = str(uuid.uuid4())
    for sid in ("not-a-uuid", str(uuid.uuid4())):
        r = _click(client, sid, doc)
        assert r.status_code == 404 and r.json()["error"]["code"] == "SEARCH_NOT_FOUND"
    page = _search(client)
    bad = client.post(f"/api/v1/searches/{page['search_id']}/clicks", json={"document_id": "nope"})
    assert bad.status_code == 422


def test_the_database_refuses_signals_for_documents_outside_the_search(client, session_factory):
    page = _search(client)
    sid = page["search_id"]
    with session_factory() as s:
        other = s.scalar(
            text(
                "select id from documents where id not in "
                "(select document_id from search_results where search_run_id = :sid) limit 1"
            ),
            {"sid": sid},
        )
        for table, extra in ((ClickEvent, {"rank": 1}), (RelevanceFeedback, {"label": "relevant"})):
            with pytest.raises(IntegrityError):
                s.execute(insert(table).values(search_run_id=sid, document_id=other, **extra))
            s.rollback()


def test_feedback_is_an_upsert_with_one_row_per_result(client, session_factory):
    page = _search(client)
    sid, doc = page["search_id"], page["results"][0]["document_id"]
    first = _feedback(client, sid, doc, "relevant")
    second = _feedback(client, sid, doc, "not_relevant")
    assert first.status_code == second.status_code == 200
    [row] = _rows(session_factory, RelevanceFeedback)
    assert row.label == "not_relevant" and row.updated_at >= row.created_at
    other = _feedback(client, sid, page["results"][1]["document_id"], "relevant")
    assert other.status_code == 200 and len(_rows(session_factory, RelevanceFeedback)) == 2


def test_feedback_validation_and_ownership(client, session_factory):
    page = _search(client)
    sid, doc = page["search_id"], page["results"][0]["document_id"]
    assert _feedback(client, sid, doc, "great").status_code == 422
    assert _feedback(client, sid, str(uuid.uuid4()), "relevant").status_code == 404
    assert _feedback(client, str(uuid.uuid4()), doc, "relevant").status_code == 404
    assert (
        client.put(
            f"/api/v1/searches/{sid}/feedback/not-a-uuid", json={"label": "relevant"}
        ).status_code
        == 422
    )
    assert _rows(session_factory, RelevanceFeedback) == []


def test_only_returned_results_count_as_impressions(client, session_factory):
    page = _search(client)
    sid = page["search_id"]

    def shown():
        with session_factory() as s:
            return s.scalar(
                select(func.count())
                .select_from(SearchResult)
                .where(SearchResult.search_run_id == sid, SearchResult.shown_at.is_not(None))
            )

    assert shown() == 5  # the first page
    client.get(
        f"/api/v1/searches/{sid}/results", params={"cursor": page["next_cursor"], "limit": 5}
    )
    assert shown() == 10
    client.get(
        f"/api/v1/searches/{sid}/results", params={"cursor": page["next_cursor"], "limit": 5}
    )
    assert shown() == 10  # re-fetching the same page adds nothing


def test_writes_have_their_own_stricter_rate_limit(make_client):
    client = make_client(write_rate_limit_per_minute=2)
    page = _search(client)
    sid, doc = page["search_id"], page["results"][0]["document_id"]
    codes = [_click(client, sid, doc).status_code for _ in range(4)]
    assert codes == [201, 201, 429, 429]
    limited = _click(client, sid, doc)
    assert int(limited.headers["retry-after"]) >= 1
    assert _search(client)["search_id"]  # searching is a separate bucket
