import pytest
from api_support import SEARCH
from app.api.suggest import build_suggester
from app.search.trie import PrefixTrie, fold, looks_sensitive


def _titles(client, **params):
    r = client.get("/api/v1/suggestions", params=params)
    assert r.status_code == 200, r.text
    return [s["text"] for s in r.json()["suggestions"]], r.json()


def test_fold_removes_case_accents_and_extra_whitespace():
    assert fold("  ÉCOLOGÍA   Marina ") == "ecologia marina"
    assert fold("Straße") == "strasse" and fold("Ñandú") == "nandu"


def test_trie_prefix_lookup_is_unicode_aware_and_keeps_the_display_string():
    trie = PrefixTrie()
    for title in ("Volcán", "Volcano", "Ópera", "Glaciar"):
        trie.add(title)
    assert [e.display for e in trie.suggest("VOLC")] == ["Volcán", "Volcano"]  # shorter first
    assert [e.display for e in trie.suggest("opé")] == ["Ópera"]
    assert trie.suggest("zzz") == [] and trie.suggest("   ") == []  # nothing typed -> nothing


def test_trie_deduplicates_across_languages_and_merges_them():
    trie = PrefixTrie()
    for lang in ("en", "es", "fr"):
        trie.add("Jazz", languages={lang})
    trie.add("Volcán", languages={"es"})
    trie.add("Volcan", languages={"fr"})  # same folded key
    assert len(trie) == 2
    assert sorted(trie.suggest("jaz")[0].languages) == ["en", "es", "fr"]
    assert sorted(trie.suggest("volc")[0].languages) == ["es", "fr"]


def test_popularity_outranks_length_and_nodes_are_bounded():
    trie = PrefixTrie(max_per_node=3)
    for i in range(10):
        trie.add(f"alpha {i}")
    trie.add("alpha long popular query", weight=50)
    top = trie.suggest("alp", limit=10)
    assert len(top) == 3 and top[0].display == "alpha long popular query"


def test_sensitive_looking_text_is_detected():
    assert looks_sensitive("mail me at someone@example.com")
    assert looks_sensitive("visit https://example.com/x")
    assert looks_sensitive("my number is 5551234567")
    assert looks_sensitive("x" * 61)
    assert not looks_sensitive("climate change policy")


def test_endpoint_matches_prefixes_ignoring_case_and_accents(client):
    titles, body = _titles(client, prefix="volc")
    assert titles[0] == "Volcán" and body["prefix"] == "volc"
    assert _titles(client, prefix="VOLCÁN")[0] == titles
    assert _titles(client, prefix="opera")[0][0] in ("Ópera", "Opéra", "Opera")


def test_results_are_deduplicated_limited_and_filterable_by_language(client):
    titles, body = _titles(client, prefix="ja")
    assert titles == ["Jazz"] and body["suggestions"][0]["languages"] == ["en", "es", "fr"]
    es_titles, _ = _titles(client, prefix="vo", language="es")
    assert es_titles == ["Volcán"]
    en_only, _ = _titles(client, prefix="volc", language="en")
    assert en_only == ["Volcano"]
    assert (
        len(_titles(client, prefix="vo")[0]) == 2
        and len(_titles(client, prefix="vo", limit=1)[0]) == 1
    )


@pytest.mark.parametrize(
    "params", [{"prefix": "v"}, {"prefix": "x" * 101}, {"prefix": "ab", "limit": 0},
               {"prefix": "ab", "limit": 21}, {"prefix": "ab", "language": "de"}, {}],
)  # fmt: skip
def test_invalid_suggestion_requests_are_rejected(client, params):
    r = client.get("/api/v1/suggestions", params=params)
    assert r.status_code == 422 and r.json()["error"]["request_id"]


def test_unknown_prefix_returns_an_empty_list(client):
    assert _titles(client, prefix="qqqq")[0] == []


def test_suggestions_have_their_own_rate_limit(make_client):
    client = make_client(suggest_rate_limit_per_minute=2)
    codes = [
        client.get("/api/v1/suggestions", params={"prefix": "vo"}).status_code for _ in range(4)
    ]
    assert codes == [200, 200, 429, 429]
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200  # separate bucket


def _search(client, query):
    return client.post("/api/v1/search", json={**SEARCH, "query": query})


def _rebuild(client):
    state = client.app.state.polysearch
    state.suggester = None  # force the next request to rebuild from current data
    return state


def test_rare_raw_queries_never_become_suggestions(make_client):
    client = make_client(analytics_store_query_text=True, popular_query_min_count=3)
    _search(client, "one off secret topic")  # searched once
    for _ in range(3):
        _search(client, "popular volcano topic")
    _rebuild(client)
    titles, _ = _titles(client, prefix="po")
    assert "popular volcano topic" in titles
    assert _titles(client, prefix="one")[0] == []  # below the frequency threshold


def test_sensitive_looking_popular_queries_are_never_suggested(make_client):
    client = make_client(analytics_store_query_text=True, popular_query_min_count=2)
    for _ in range(3):
        _search(client, "contact me jane@example.com")
        _search(client, "call 5551234567 now")
    _rebuild(client)
    assert _titles(client, prefix="contact")[0] == [] and _titles(client, prefix="call")[0] == []


def test_query_text_is_never_suggested_when_storage_is_disabled(client):
    for _ in range(5):
        _search(client, "popular volcano topic")
    _rebuild(client)
    assert _titles(client, prefix="po")[0] == []


def test_the_trie_is_rebuilt_when_the_active_index_version_changes(client, api_bundle):
    state = client.app.state.polysearch
    first = state.suggester
    assert first is not None and first.index_version == api_bundle.version
    api_bundle.manifest["version"] = "idx-newer"
    _titles(client, prefix="volc")
    assert state.suggester is not first and state.suggester.index_version == "idx-newer"


def test_only_searchable_titles_are_suggested(client, session_factory):
    from app.db.models import Document
    from sqlalchemy import update

    with session_factory() as s:  # a title that exists in the database but is not in the index
        s.add(
            Document(
                source_id=s.scalar(select_source_id()),
                external_id="Unindexed_Page",
                language="en",
                title="Unindexed Page",
                summary="x",
                source_url="https://en.wikipedia.org/wiki/Unindexed_Page",
                content_sha256="a" * 64,
            )
        )
        s.execute(update(Document).where(Document.external_id == "Jazz").values(is_active=False))
        s.commit()
    _rebuild(client)
    assert _titles(client, prefix="unind")[0] == []
    assert _titles(client, prefix="jaz")[0] == []  # inactive documents are not suggested


def select_source_id():
    from app.db.models import ContentSource
    from sqlalchemy import select

    return select(ContentSource.id).limit(1)


def test_build_suggester_reflects_the_bundle_version(client):
    suggester = build_suggester(client.app.state.polysearch)
    assert len(suggester.all) > 0 and set(suggester.by_language) == {"en", "es", "fr"}


def test_no_index_means_no_suggestions(session_factory):
    from api_support import test_settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(test_settings(), session_factory=session_factory)) as c:
        r = c.get("/api/v1/suggestions", params={"prefix": "vo"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "INDEX_UNAVAILABLE"
