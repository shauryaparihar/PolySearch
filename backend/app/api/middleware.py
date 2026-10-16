"""Request context: request IDs, body-size limit and one access-log line per request.

Pure ASGI (no BaseHTTPMiddleware) so streaming and error responses behave predictably.
"""

from __future__ import annotations

import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response
from app.api.logging_config import log_event

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{8,64}$")


class BodyTooLarge(Exception):
    pass


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        supplied = headers.get("x-request-id", "")
        request_id = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started, status = time.perf_counter(), 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    scope["state"]["body_too_large"] = True  # lets the error handler answer 413
                    raise BodyTooLarge
            return message

        declared = headers.get("content-length", "")
        try:
            if declared.isdigit() and int(declared) > self.max_body_bytes:
                raise BodyTooLarge
            await self.app(scope, limited_receive, send_wrapper)
        except BodyTooLarge:
            if not response_started:
                response = error_response(
                    413, "REQUEST_TOO_LARGE", "The request body is too large.", request_id
                )
                await response(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            log_event(
                "request",
                request_id=request_id,
                method=scope["method"],
                route=getattr(route, "path", "unmatched"),  # route template, never the raw URL
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
