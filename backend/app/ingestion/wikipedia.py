"""MediaWiki (Wikipedia) connector.

Security rules enforced here:
* Only the fixed per-language HTTPS hosts in ``ALLOWED_HOSTS`` are ever contacted.
  Callers pass a language code and page titles, never a URL.
* HTTP redirects are never followed. A 3xx response is an error and no follow-up
  request is made, which prevents redirect-based SSRF to unapproved hosts.
* Timeouts, bounded retries with exponential backoff + jitter, and bounded batch sizes.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from datetime import datetime
from urllib.parse import quote, urlsplit

import httpx

from app.config import SUPPORTED_LANGUAGES, Settings, get_settings
from app.ingestion.sources import NormalizedRecord, PageStub, RawRecord, SourceError
from app.ingestion.text import bound_text, clean_text, content_sha256

ALLOWED_HOSTS: dict[str, str] = {lang: f"{lang}.wikipedia.org" for lang in SUPPORTED_LANGUAGES}
API_PATH = "/w/api.php"
_CONTINUE_KEYS = {"gapcontinue", "continue"}
MAX_CONTINUATIONS = 20
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def api_url(language: str) -> str:
    if language not in ALLOWED_HOSTS:
        raise SourceError("unsupported_language", "Language is not supported")
    return f"https://{ALLOWED_HOSTS[language]}{API_PATH}"


def assert_allowed_url(url: str) -> None:
    """Raise unless ``url`` is an HTTPS API URL on an approved host."""
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in ALLOWED_HOSTS.values()
        or parts.port not in (None, 443)
        or parts.username
        or parts.password
        or parts.path != API_PATH
    ):
        raise SourceError("url_not_allowed", "URL is not an approved HTTPS MediaWiki endpoint")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _validate_titles(titles: list[str], limit: int) -> list[str]:
    if not titles or len(titles) > limit:
        raise SourceError("batch_size_invalid", f"Between 1 and {limit} titles are allowed")
    cleaned = []
    for title in titles:
        title = title.strip()
        if (
            not title
            or len(title.encode("utf-8")) > 255
            or "|" in title
            or _CONTROL_RE.search(title)
        ):
            raise SourceError("title_invalid", "Invalid page title")
        cleaned.append(title)
    return cleaned


def _validate_token(token: dict | None) -> dict[str, str] | None:
    if token is None:
        return None
    if (
        not isinstance(token, dict)
        or not token
        or set(token) - _CONTINUE_KEYS
        or not all(isinstance(v, str) and len(v) <= 512 for v in token.values())
    ):
        raise SourceError("continuation_invalid", "Invalid continuation token")
    return dict(token)


class WikipediaSource:
    name = "wikipedia"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings or get_settings()
        self._sleep = sleep
        self._jitter = jitter
        self._client = httpx.Client(
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": self.settings.wikipedia_user_agent},
            timeout=httpx.Timeout(
                self.settings.http_read_timeout, connect=self.settings.http_connect_timeout
            ),
        )

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- transport
    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        s = self.settings
        delay = min(s.http_backoff_max, s.http_backoff_base * (2**attempt))
        delay += self._jitter() * s.http_backoff_base
        if retry_after is not None:
            delay = max(delay, min(retry_after, s.http_backoff_max))
        return min(delay, s.http_backoff_max)

    def _request(self, language: str, params: dict[str, str | int]) -> dict:
        url = api_url(language)
        assert_allowed_url(url)
        query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        last: SourceError | None = None
        for attempt in range(self.settings.http_max_retries + 1):
            if attempt:
                self._sleep(self._backoff(attempt - 1, last.retry_after if last else None))
            try:
                response = self._client.get(url, params=query)
            except (httpx.TimeoutException, httpx.TransportError):
                last = SourceError(
                    "network_error", "Network error contacting source", transient=True
                )
                continue
            if 300 <= response.status_code < 400:
                # Never follow: the Location header is not trusted or even read.
                raise SourceError("redirect_blocked", "Source redirect was blocked")
            if response.status_code == 429 or response.status_code >= 500:
                last = SourceError(
                    f"http_{response.status_code}", "Source unavailable", transient=True
                )
                retry_after = response.headers.get("Retry-After", "")
                last.retry_after = float(retry_after) if retry_after.isdigit() else None
                continue
            if response.status_code != 200:
                raise SourceError(f"http_{response.status_code}", "Source rejected the request")
            try:
                body = response.json()
            except ValueError:
                raise SourceError("bad_json", "Source returned an invalid response") from None
            if not isinstance(body, dict):
                raise SourceError("bad_json", "Source returned an invalid response")
            error = body.get("error")
            if error:
                code = (
                    str(error.get("code", "api_error")) if isinstance(error, dict) else "api_error"
                )
                if code == "maxlag":
                    last = SourceError("maxlag", "Source is lagging", transient=True)
                    continue
                raise SourceError(f"api_{code}", "Source API reported an error")
            return body
        assert last is not None
        raise last

    # ------------------------------------------------------------------ queries
    def fetch_rights_info(self) -> dict[str, dict[str, str]]:
        """Licence metadata per language, exactly as the live API reports it."""
        info = {}
        for language in SUPPORTED_LANGUAGES:
            body = self._request(
                language, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"}
            )
            rights = body.get("query", {}).get("rightsinfo", {})
            if not rights.get("text") or not rights.get("url"):
                raise SourceError("rights_missing", "Source did not report licence information")
            info[language] = {"name": rights["text"], "url": rights["url"]}
        return info

    def fetch_revisions(self, language: str, titles: list[str]) -> list[PageStub | str]:
        """Cheap revision lookup. Returns a PageStub per existing page, or the title if missing."""
        titles = _validate_titles(titles, self.settings.titles_per_request * 2)
        body = self._request(
            language,
            {"action": "query", "prop": "info", "titles": "|".join(titles), "redirects": 1},
        )
        out: list[PageStub | str] = []
        for page in body.get("query", {}).get("pages", []):
            if page.get("missing") or page.get("invalid"):
                out.append(page.get("title", ""))
            else:
                out.append(self._stub(language, page))
        return out

    def fetch_records(self, language: str, titles: list[str]) -> tuple[list[RawRecord], list[str]]:
        """Fetch lead-section plain text for up to ``titles_per_request`` titles."""
        titles = _validate_titles(titles, self.settings.titles_per_request)
        body = self._request(
            language,
            {
                "action": "query",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": "max",
                "inprop": "url",
                "titles": "|".join(titles),
                "redirects": 1,
            },
        )
        records, missing = [], []
        for page in body.get("query", {}).get("pages", []):
            if page.get("missing") or page.get("invalid"):
                missing.append(page.get("title", ""))
                continue
            records.append(
                RawRecord(
                    language=language,
                    title=page.get("title", ""),
                    extract=page.get("extract") or "",
                    url=page.get("canonicalurl"),
                    revision=str(page["lastrevid"]) if page.get("lastrevid") is not None else None,
                    updated_at=_parse_ts(page.get("touched")),
                )
            )
        return records, missing

    def fetch_langlinks(
        self, language: str, titles: list[str], target_language: str
    ) -> dict[str, str | None]:
        """Map each existing page in ``language`` to its linked title in ``target_language``.

        Keys are canonical (normalised/redirect-resolved) titles of the *source* pages; the
        value is the interlanguage link title, or None if the page has no such link.
        """
        if target_language not in ALLOWED_HOSTS:
            raise SourceError("unsupported_language", "Language is not supported")
        titles = _validate_titles(titles, self.settings.titles_per_request * 2)
        links: dict[str, str | None] = {}
        token: dict[str, str] = {}
        for _ in range(MAX_CONTINUATIONS):
            body = self._request(
                language,
                {
                    "action": "query",
                    "prop": "langlinks",
                    "titles": "|".join(titles),
                    "lllang": target_language,
                    "lllimit": "max",
                    "redirects": 1,
                    **token,
                },
            )
            for page in body.get("query", {}).get("pages", []):
                if page.get("missing") or page.get("invalid"):
                    continue
                found = page.get("langlinks") or []
                title = page.get("title", "")
                links[title] = found[0]["title"] if found else links.get(title)
            cont = body.get("continue")
            if not cont:
                return links
            token = {k: v for k, v in cont.items() if isinstance(v, str) and len(v) <= 512}
        raise SourceError("continuation_limit", "Too many continuation steps")

    def resolve_titles(self, language: str, titles: list[str]) -> dict[str, str | None]:
        """Map requested titles to canonical titles (following normalisation and redirects).

        Missing pages map to None. This is how an interlanguage link that points at a redirect
        is tied to the page that actually holds the article.
        """
        titles = _validate_titles(titles, self.settings.titles_per_request * 2)
        body = self._request(
            language,
            {"action": "query", "prop": "info", "titles": "|".join(titles), "redirects": 1},
        )
        query = body.get("query", {})
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        existing = {
            p["title"] for p in query.get("pages", []) if not (p.get("missing") or p.get("invalid"))
        }
        resolved: dict[str, str | None] = {}
        for requested in titles:
            title = normalized.get(requested, requested)
            for _ in range(5):  # bounded redirect chain
                if title not in redirects:
                    break
                title = redirects[title]
            resolved[requested] = title if title in existing else None
        return resolved

    def fetch_page(self, external_id: str, language: str) -> RawRecord | None:
        records, _ = self.fetch_records(language, [external_id.replace("_", " ")])
        return records[0] if records else None

    def fetch_updates(
        self, language: str, continuation_token: dict | None, limit: int
    ) -> tuple[list[PageStub], dict | None]:
        """Enumerate main-namespace non-redirect pages (alphabetically) with continuation."""
        limit = max(1, min(limit, 50))
        params: dict[str, str | int] = {
            "action": "query",
            "generator": "allpages",
            "gapnamespace": 0,
            "gapfilterredir": "nonredirects",
            "gaplimit": limit,
            "prop": "info",
        }
        token = _validate_token(continuation_token)
        if token:
            params.update(token)
        body = self._request(language, params)
        stubs = [
            self._stub(language, p)
            for p in body.get("query", {}).get("pages", [])
            if not p.get("missing")
        ]
        stubs.sort(key=lambda s: s.title)
        next_token = body.get("continue")
        return stubs, (_validate_token(next_token) if next_token else None)

    def _stub(self, language: str, page: dict) -> PageStub:
        revision = str(page["lastrevid"]) if page.get("lastrevid") is not None else None
        return PageStub(language, page.get("title", ""), revision, _parse_ts(page.get("touched")))

    # ------------------------------------------------------------ normalisation
    def canonical_url(self, language: str, title: str, candidate: str | None) -> str:
        host = ALLOWED_HOSTS[language]
        if candidate:
            parts = urlsplit(candidate)
            if parts.scheme == "https" and parts.hostname == host and not parts.username:
                return candidate
        return f"https://{host}/wiki/{quote(title.replace(' ', '_'), safe='()_,:')}"

    def normalise_record(self, raw: RawRecord) -> NormalizedRecord:
        if raw.language not in ALLOWED_HOSTS:
            raise SourceError("unsupported_language", "Language is not supported")
        title = clean_text(raw.title)
        summary = bound_text(clean_text(raw.extract), self.settings.summary_max_chars)
        if not title:
            raise SourceError("empty_title", "Record has no title")
        if not summary:
            raise SourceError("empty_summary", "Record has no summary")
        return NormalizedRecord(
            external_id=title.replace(" ", "_"),
            language=raw.language,
            title=title,
            summary=summary,
            source_url=self.canonical_url(raw.language, title, raw.url),
            source_revision=raw.revision,
            source_updated_at=raw.updated_at,
            content_sha256=content_sha256(title, summary),
        )
