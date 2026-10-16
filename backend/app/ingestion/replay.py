"""Record/replay HTTP transport so benchmark construction is reproducible offline.

The first run hits the real (allow-listed) API and stores each successful JSON response;
later runs replay from disk. Only 200 JSON responses are stored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx


class RecordReplayTransport(httpx.BaseTransport):
    def __init__(
        self, cache_dir: Path, inner: httpx.BaseTransport | None = None, offline: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.inner = inner if inner is not None else httpx.HTTPTransport()
        self.offline = offline
        self.hits = 0
        self.misses = 0

    def _path(self, request: httpx.Request) -> Path:
        key = hashlib.sha256(f"{request.method} {request.url}".encode()).hexdigest()
        return self.cache_dir / f"{key[:24]}.json"

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = self._path(request)
        if path.exists():
            self.hits += 1
            stored = json.loads(path.read_text(encoding="utf-8"))
            return httpx.Response(stored["status"], json=stored["body"], request=request)
        if self.offline:
            raise httpx.ConnectError("offline mode: response not in cache", request=request)
        self.misses += 1
        response = self.inner.handle_request(request)
        response.read()
        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                return response
            path.write_text(
                json.dumps(
                    {"url": str(request.url), "status": 200, "body": body},
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=1,
                ),
                encoding="utf-8",
            )
        return response
