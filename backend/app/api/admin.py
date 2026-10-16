"""Admin authentication (one environment-provided key; not user authentication)."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Request

from app.api.errors import ApiError, request_id_of
from app.api.logging_config import log_event
from app.api.ratelimit import client_id
from app.api.state import AppState


def _throttled(retry_after: int) -> ApiError:
    return ApiError(
        429,
        "RATE_LIMITED",
        "Too many requests. Please slow down.",
        details={"retry_after_seconds": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


def require_admin(request: Request) -> None:
    """Raise unless the request carries the configured admin key.

    * No key configured: the admin API does not exist (404), so it cannot be probed.
    * Wrong keys are throttled tightly (``admin_failed_attempts_per_minute`` per client): once a
      client has failed that often it is locked out for the rest of the window, even with the
      correct key. This slows guessing without slowing legitimate use.
    * Authenticated calls have a separate, generous limit (status polling is normal).
    * Comparison hashes both sides and uses ``hmac.compare_digest`` (constant time).
    * Failures say nothing about *why* (missing, wrong or malformed key look identical).
    """
    state: AppState = request.app.state.polysearch
    configured = state.settings.admin_api_key
    if not configured:
        raise ApiError(404, "NOT_FOUND", "Request could not be served.")
    peer = request.client.host if request.client else None
    who = client_id(peer, request.headers.get("x-forwarded-for"), state.settings.trusted_proxy_ips)

    locked, retry_after = state.admin_failure_limiter.blocked(who)
    if locked:
        state.metrics["rate_limited"]["admin"] += 1
        raise _throttled(retry_after)

    supplied = request.headers.get("x-admin-key", "")
    ok = hmac.compare_digest(
        hashlib.sha256(supplied.encode("utf-8", "ignore")).digest(),
        hashlib.sha256(configured.encode("utf-8")).digest(),
    )
    if not ok:
        state.admin_failure_limiter.record(who)
        log_event(
            "admin_auth_failed",
            level=30,
            request_id=request_id_of(request),
            error_code="UNAUTHORIZED",
        )
        raise ApiError(401, "UNAUTHORIZED", "Authentication failed.")

    allowed, retry_after = state.admin_limiter.check(who)
    if not allowed:
        state.metrics["rate_limited"]["admin"] += 1
        raise _throttled(retry_after)
