"""Admin authentication (one environment-provided key; not user authentication)."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Request

from app.api.errors import ApiError, request_id_of
from app.api.logging_config import log_event
from app.api.ratelimit import client_id
from app.api.state import AppState


def require_admin(request: Request) -> None:
    """Raise unless the request carries the configured admin key.

    * No key configured: the admin API does not exist (404), so it cannot be probed.
    * Every attempt, including wrong keys, counts against a strict per-client limit.
    * Comparison hashes both sides and uses ``hmac.compare_digest`` (constant time).
    * Failures say nothing about *why* (missing, wrong or malformed key look identical).
    """
    state: AppState = request.app.state.polysearch
    configured = state.settings.admin_api_key
    if not configured:
        raise ApiError(404, "NOT_FOUND", "Request could not be served.")
    peer = request.client.host if request.client else None
    who = client_id(peer, request.headers.get("x-forwarded-for"), state.settings.trusted_proxy_ips)
    allowed, retry_after = state.admin_limiter.check(who)
    if not allowed:
        state.metrics["rate_limited"]["admin"] += 1
        raise ApiError(
            429, "RATE_LIMITED", "Too many requests. Please slow down.",
            headers={"Retry-After": str(retry_after)},
        )  # fmt: skip
    supplied = request.headers.get("x-admin-key", "")
    ok = hmac.compare_digest(
        hashlib.sha256(supplied.encode("utf-8", "ignore")).digest(),
        hashlib.sha256(configured.encode("utf-8")).digest(),
    )
    if not ok:
        log_event(
            "admin_auth_failed",
            level=30,
            request_id=request_id_of(request),
            error_code="UNAUTHORIZED",
        )
        raise ApiError(401, "UNAUTHORIZED", "Authentication failed.")
