import httpx
import pytest
from app.config import Settings
from app.ingestion.sources import RawRecord, SourceError
from app.ingestion.wikipedia import WikipediaSource, assert_allowed_url


def _source(transport, **settings):
    return WikipediaSource(
        Settings(**settings), transport=transport, sleep=lambda _s: None, jitter=lambda: 0.0
    )


def test_continuation_is_followed_across_pages(source, fake_wiki):
    seen, token = [], None
    while True:
        stubs, token = source.fetch_updates("en", token, 50)
        seen += [s.title for s in stubs]
        if token is None:
            break
    assert seen == ["Mathematics", "Physics", "Zebra"]  # fake wiki pages in chunks of 2
    assert len(fake_wiki.requests) == 2


def test_transient_errors_are_retried_a_bounded_number_of_times(fake_wiki):
    fake_wiki.fail_next = [httpx.Response(503)] * 10
    src = _source(fake_wiki.transport(), http_max_retries=2)
    with pytest.raises(SourceError) as exc:
        src.fetch_records("en", ["Mathematics"])
    assert exc.value.transient
    assert len(fake_wiki.requests) == 3  # 1 try + 2 retries, then give up


def test_transient_error_then_success(fake_wiki):
    fake_wiki.fail_next = [httpx.ReadTimeout("slow")]
    src = _source(fake_wiki.transport(), http_max_retries=2)
    records, _ = src.fetch_records("en", ["Mathematics"])
    assert [r.title for r in records] == ["Mathematics"]
    assert len(fake_wiki.requests) == 2


def test_non_transient_error_is_not_retried(fake_wiki):
    fake_wiki.fail_next = [httpx.Response(404)]
    src = _source(fake_wiki.transport())
    with pytest.raises(SourceError) as exc:
        src.fetch_records("en", ["Mathematics"])
    assert exc.value.code == "http_404"
    assert len(fake_wiki.requests) == 1


def test_api_error_body_is_a_safe_failure(fake_wiki):
    fake_wiki.fail_next = [
        httpx.Response(200, json={"error": {"code": "badvalue", "info": "secret"}})
    ]
    with pytest.raises(SourceError) as exc:
        _source(fake_wiki.transport()).fetch_records("en", ["Mathematics"])
    assert exc.value.code == "api_badvalue"
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://en.wikipedia.org/w/api.php",
        "https://evil.example/w/api.php",
        "https://en.wikipedia.org.evil.example/w/api.php",
        "https://user:pw@en.wikipedia.org/w/api.php",
        "https://en.wikipedia.org:8443/w/api.php",
    ],
)
def test_url_allowlist_rejects_unapproved_urls(url):
    with pytest.raises(SourceError) as exc:
        assert_allowed_url(url)
    assert exc.value.code == "url_not_allowed"


def test_redirect_to_unapproved_host_is_blocked_without_a_follow_up_request():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})

    with pytest.raises(SourceError) as exc:
        _source(httpx.MockTransport(handler)).fetch_records("en", ["Mathematics"])
    assert exc.value.code == "redirect_blocked"
    assert len(seen) == 1 and "evil.example" not in seen[0]


def test_only_supported_languages_and_valid_titles_are_accepted(source):
    for call in (
        lambda: source.fetch_records("de", ["X"]),
        lambda: source.fetch_records("en", ["A|B"]),
        lambda: source.fetch_records("en", [f"T{i}" for i in range(21)]),
        lambda: source.fetch_updates("en", {"evil": "x"}, 10),
    ):
        with pytest.raises(SourceError):
            call()


def test_normalise_keeps_provenance_and_rejects_foreign_urls(source):
    raw = RawRecord(
        "en", "Order  (math)", "Text <b>here</b>.", "https://evil.example/x", "77", None
    )
    rec = source.normalise_record(raw)
    assert rec.source_revision == "77"
    assert rec.source_url == "https://en.wikipedia.org/wiki/Order_(math)"  # never the foreign URL
    assert rec.summary == "Text here."


def test_empty_summary_is_a_safe_failure(source):
    with pytest.raises(SourceError) as exc:
        source.normalise_record(RawRecord("en", "T", "   ", None, "1", None))
    assert exc.value.code == "empty_summary"


def test_rights_info_is_read_from_the_source_for_every_language(source):
    rights = source.fetch_rights_info()
    assert set(rights) == {"en", "es", "fr"}
    assert rights["es"]["url"].endswith("deed.es")
