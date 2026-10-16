from api_support import SEARCH
from app.api.cache import QueryCache, cache_key, canonical_query, query_sha256

BASE = dict(
    query="Volcano", query_language="en", target_languages=["es", "fr"], sources=["wikipedia"],
    mode="dense", pool_size=100, index_version="idx-a",
)  # fmt: skip


def test_cache_key_ignores_whitespace_and_filter_order_only():
    same = cache_key(**{**BASE, "query": "  Volcano  ", "target_languages": ["fr", "es"]})
    assert same == cache_key(**BASE)
    assert len(same) == 64 and "volcano" not in same.lower()


def test_cache_key_separates_filters_mode_version_and_pool():
    base = cache_key(**BASE)
    variants = [
        {"query": "Volcanoes"},
        {"query_language": "es"},
        {"target_languages": ["es"]},
        {"sources": ["other"]},
        {"mode": "hybrid"},
        {"pool_size": 50},
        {"index_version": "idx-b"},  # a new index makes old entries unreachable
    ]
    keys = {cache_key(**{**BASE, **v}) for v in variants}
    assert base not in keys and len(keys) == len(variants)


def test_query_fingerprint_is_stable_and_normalised():
    assert query_sha256("a   b") == query_sha256(" a b ")
    assert canonical_query("  a \n b ") == "a b"


def test_cache_entries_expire_and_are_bounded():
    now = [0.0]
    cache = QueryCache(max_entries=2, ttl_seconds=10, timer=lambda: now[0])
    cache.set("a", 1)
    assert cache.get("a") == 1
    now[0] = 11
    assert cache.get("a") is None  # expired
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 2


def test_repeat_search_is_a_cache_hit_without_running_the_encoder_again(client, encoder):
    first = client.post("/api/v1/search", json=SEARCH).json()
    calls = encoder.calls
    second = client.post("/api/v1/search", json={**SEARCH, "query": "  Volcano "}).json()
    assert (first["cache_hit"], second["cache_hit"]) == (False, True)
    assert encoder.calls == calls
    assert first["search_id"] != second["search_id"]  # each search gets its own stored pool
    assert [r["document_id"] for r in first["results"]] == [
        r["document_id"] for r in second["results"]
    ]


def test_a_different_filter_or_mode_is_a_cache_miss(client):
    client.post("/api/v1/search", json=SEARCH)
    assert (
        client.post("/api/v1/search", json={**SEARCH, "target_languages": ["es"]}).json()[
            "cache_hit"
        ]
        is False
    )
    assert (
        client.post("/api/v1/search", json={**SEARCH, "mode": "hybrid"}).json()["cache_hit"]
        is False
    )
    assert client.post("/api/v1/search", json=SEARCH).json()["cache_hit"] is True


def test_cache_entries_from_an_older_index_are_not_served(make_client, api_bundle):
    client = make_client()
    client.post("/api/v1/search", json=SEARCH)
    api_bundle.manifest["version"] = "idx-newer"  # as if a new version were activated
    assert client.post("/api/v1/search", json=SEARCH).json()["cache_hit"] is False
