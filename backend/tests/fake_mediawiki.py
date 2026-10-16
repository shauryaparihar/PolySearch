"""A tiny in-memory MediaWiki API for tests (served through httpx.MockTransport)."""

from __future__ import annotations

import httpx

RIGHTS = {"text": "Creative Commons Attribution-Share Alike 4.0"}


class FakeMediaWiki:
    def __init__(self, pages: dict[str, dict[str, dict]], page_size: int = 2):
        # pages[language][title] = {"revid": int, "extract": str}
        self.pages = pages
        self.page_size = page_size
        self.requests: list[httpx.Request] = []
        self.fail_next: list[httpx.Response | Exception] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_next:
            outcome = self.fail_next.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        language = request.url.host.split(".")[0]
        q = dict(request.url.params)
        wiki = self.pages.get(language, {})
        if q.get("meta") == "siteinfo":
            rights = {
                **RIGHTS,
                "url": f"https://creativecommons.org/licenses/by-sa/4.0/deed.{language}",
            }
            return httpx.Response(200, json={"query": {"rightsinfo": rights}})
        if q.get("generator") == "allpages":
            titles = sorted(wiki)
            start = q.get("gapcontinue")
            if start:
                titles = [t for t in titles if t.replace(" ", "_") >= start]
            chunk = titles[: min(int(q["gaplimit"]), self.page_size)]
            body: dict = {"query": {"pages": [self._info(language, t) for t in chunk]}}
            rest = titles[len(chunk) :]
            if rest:
                body["continue"] = {
                    "gapcontinue": rest[0].replace(" ", "_"),
                    "continue": "gapcontinue||",
                }
            return httpx.Response(200, json=body)
        pages = []
        for title in q["titles"].split("|"):
            if title not in wiki:
                pages.append({"title": title, "missing": True})
            elif "extracts" in q.get("prop", ""):
                pages.append({**self._info(language, title), "extract": wiki[title]["extract"]})
            else:
                pages.append(self._info(language, title))
        return httpx.Response(200, json={"query": {"pages": pages}})

    def _info(self, language: str, title: str) -> dict:
        slug = title.replace(" ", "_")
        return {
            "title": title,
            "lastrevid": self.pages[language][title]["revid"],
            "touched": "2026-10-01T00:00:00Z",
            "canonicalurl": f"https://{language}.wikipedia.org/wiki/{slug}",
        }
