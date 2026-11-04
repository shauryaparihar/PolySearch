"""Readiness of answer mode, kept apart from search readiness (``/readyz`` is unchanged)."""

from __future__ import annotations

from sqlalchemy import func, select

from app.api.state import AppState
from app.db.models import DocumentPassage, IndexVersion


def run_rag_readiness(state: AppState) -> dict:
    settings = state.settings
    bundle, _ = state.snapshot()
    store = state.passages
    checks = {
        "enabled": settings.rag_enabled,
        "provider_configured": settings.rag_configured,
        "passages_loaded": store is not None,
        "passages_match_index": bool(store and bundle and store.index_version == bundle.version),
        "passages_in_database": False,
    }
    if store is not None and checks["passages_match_index"]:
        try:
            with state.session_factory() as session:
                stored = session.scalar(
                    select(func.count())
                    .select_from(DocumentPassage)
                    .join(IndexVersion, IndexVersion.id == DocumentPassage.index_version_id)
                    .where(IndexVersion.version == store.index_version)
                )
            checks["passages_in_database"] = stored == store.count
        except Exception:  # noqa: BLE001  (a database problem is "not ready", never a 500)
            checks["passages_in_database"] = False
    body = {"ready": all(checks.values()), "checks": checks}
    if not body["ready"] and state.passage_error:
        body["reason"] = state.passage_error  # a stable code, never an exception message
    return body
