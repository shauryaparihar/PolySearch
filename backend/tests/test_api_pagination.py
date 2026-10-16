import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from api_support import SEARCH
from app.api.service import decode_cursor, encode_cursor, purge_expired_pools
from app.db.models import SearchResult, SearchRun
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError


def _search(client, **kw):
    return client.post("/api/v1/search", json={**SEARCH, "page_size": 5, **kw}).json()


def _all_pages(client, first):
    seen, page = list(first["results"]), first
    while page["next_cursor"]:
        r = client.get(
            f"/api/v1/searches/{first['search_id']}/results",
            params={"cursor": page["next_cursor"], "limit": 5},
        )
        assert r.status_code == 200
        page = r.json()
        seen += page["results"]
    return seen


def test_pages_form_one_stable_ranking_without_gaps_or_duplicates(client):
    first = _search(client)
    seen = _all_pages(client, first)
    assert [x["rank"] for x in seen] == list(range(1, 17))
    assert len({x["document_id"] for x in seen}) == 16
    again = _all_pages(client, first)  # same stored pool, same answer
    assert [x["document_id"] for x in again] == [x["document_id"] for x in seen]


def test_paging_does_not_encode_the_query_again(client, encoder):
    first = _search(client)
    calls_after_search = encoder.calls
    _all_pages(client, first)
    assert encoder.calls == calls_after_search


def test_stored_pages_survive_the_index_becoming_unavailable(client):
    first = _search(client)
    client.app.state.polysearch.bundle = None  # e.g. mid-activation or artifacts gone
    r = client.get(
        f"/api/v1/searches/{first['search_id']}/results", params={"cursor": first["next_cursor"]}
    )
    assert r.status_code == 200 and r.json()["results"][0]["rank"] == 6


def test_the_last_page_has_no_cursor_and_limit_controls_page_length(client):
    first = _search(client)
    r = client.get(f"/api/v1/searches/{first['search_id']}/results", params={"limit": 20}).json()
    assert len(r["results"]) == 16 and r["next_cursor"] is None and r["results"][0]["rank"] == 1


def _forge(secret, search_id, rank):
    return encode_cursor(secret.encode(), search_id, rank)


def test_cursors_are_signed_and_bound_to_their_search(client):
    first = _search(client)
    sid = first["search_id"]
    url = f"/api/v1/searches/{sid}/results"
    payload = base64.urlsafe_b64encode(json.dumps({"s": sid, "r": 3}).encode()).decode()
    bad = {
        "garbage": "garbage",
        "unsigned": payload + ".0123456789abcdef0123456789abcdef",
        "no_signature": payload,
        "wrong_secret": _forge("not-the-secret", sid, 3),
        "other_search": _forge("test-secret", "00000000-0000-0000-0000-000000000001", 3),
        "rank_beyond_pool": _forge("test-secret", sid, 17),
        "rank_zero": _forge("test-secret", sid, 0),
        "oversized": "x" * 300,
    }
    for name, cursor in bad.items():
        r = client.get(url, params={"cursor": cursor})
        assert r.status_code in (400, 422), name
        assert r.json()["error"]["code"] in ("CURSOR_INVALID",), name
    ok = client.get(url, params={"cursor": _forge("test-secret", sid, 3)})
    assert ok.status_code == 200 and ok.json()["results"][0]["rank"] == 3


def test_a_cursor_cannot_be_replayed_against_another_search(client):
    a, b = _search(client), _search(client, query="Glacier")
    r = client.get(
        f"/api/v1/searches/{b['search_id']}/results", params={"cursor": a["next_cursor"]}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "CURSOR_INVALID"


@pytest.mark.parametrize("search_id", ["not-a-uuid", "00000000-0000-0000-0000-000000000000"])
def test_unknown_searches_are_not_found(client, search_id):
    r = client.get(f"/api/v1/searches/{search_id}/results")
    assert r.status_code == 404 and r.json()["error"]["code"] == "SEARCH_NOT_FOUND"


@pytest.mark.parametrize("limit", [0, 4, 21])
def test_page_limit_is_bounded(client, limit):
    first = _search(client)
    r = client.get(f"/api/v1/searches/{first['search_id']}/results", params={"limit": limit})
    assert r.status_code == 422 and r.json()["error"]["code"] == "PAGE_SIZE_INVALID"


def test_decode_cursor_roundtrip():
    secret = b"s"
    assert decode_cursor(secret, encode_cursor(secret, "abc", 7), "abc", 20) == 7


def test_database_constraints_protect_the_stored_pool(client, session_factory):
    first = _search(client)
    with session_factory() as s:
        run = s.get(SearchRun, first["search_id"])
        row = s.scalars(select(SearchResult).where(SearchResult.search_run_id == run.id)).first()
        with pytest.raises(IntegrityError):  # the same document twice in one pool
            s.execute(
                insert(SearchResult).values(
                    search_run_id=run.id, document_id=row.document_id, rank=99, score=0
                )
            )
        s.rollback()
        other = s.scalars(
            select(SearchResult).where(SearchResult.search_run_id == run.id, SearchResult.rank == 2)
        ).one()
        with pytest.raises(IntegrityError):  # the same rank twice in one pool
            s.execute(
                insert(SearchResult).values(
                    search_run_id=run.id,
                    document_id=row.document_id,
                    rank=other.rank,
                    score=0,
                )
            )
        s.rollback()


def test_old_search_pools_are_removed_by_the_retention_job(client, session_factory):
    old, new = _search(client), _search(client, query="Glacier")
    with session_factory() as s:
        s.execute(
            text("update search_runs set created_at = :t where id = :id"),
            {"t": datetime.now(UTC) - timedelta(days=40), "id": old["search_id"]},
        )
        s.commit()
        assert purge_expired_pools(s, retention_days=30) == 1
        assert s.scalar(select(func.count()).select_from(SearchRun)) == 1
        assert s.scalar(select(func.count()).select_from(SearchResult)) == 16  # old pool gone too
    assert client.get(f"/api/v1/searches/{old['search_id']}/results").status_code == 404
    assert client.get(f"/api/v1/searches/{new['search_id']}/results").status_code == 200
