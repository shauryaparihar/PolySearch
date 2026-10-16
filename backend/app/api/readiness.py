"""Readiness: is this instance able to serve correct searches right now?

``/healthz`` only says the process is alive. ``/readyz`` checks the things that make results
trustworthy: database reachable, an active index row that matches the loaded bundle, artifact
checksums and counts, the encoder loaded, and every indexed document resolving to a database
row. The (slightly expensive) result is cached for ``readiness_cache_seconds``.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import func, select, text

from app.api.state import AppState
from app.db.models import ContentSource, Document, IndexVersion
from app.search.artifacts import ArtifactError


def _check_database(state: AppState) -> bool:
    with state.session_factory() as session:
        session.execute(text("select 1"))
    return True


ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _check_schema(state: AppState) -> bool:
    """The database is at the newest migration this code expects (no half-upgraded schema)."""
    head = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()
    with state.session_factory() as session:
        current = session.scalar(text("select version_num from alembic_version"))
    return current == head


def _check_index_row(state: AppState) -> bool:
    with state.session_factory() as session:
        row = session.scalar(select(IndexVersion).where(IndexVersion.status == "active"))
        return row is not None and state.bundle is not None and row.version == state.bundle.version


def _check_artifacts(state: AppState) -> bool:
    state.bundle.verify()
    return True


def _check_encoder(state: AppState) -> bool:
    return state.bundle._encoder is not None  # noqa: SLF001 (loaded at startup)


def _check_suggester(state: AppState) -> bool:
    """The autocomplete trie exists for the bundle being served (built lazily if needed)."""
    from app.api.suggest import get_suggester

    suggester = get_suggester(state)
    return suggester is not None and suggester.index_version == state.bundle.version


def _check_mappings(state: AppState) -> bool:
    """Every indexed document key must resolve to an active database document."""
    indexed = state.bundle.document_keys()
    with state.session_factory() as session:
        stored = {
            f"{lang}:{ext}"
            for lang, ext in session.execute(
                select(Document.language, Document.external_id)
                .join(ContentSource, ContentSource.id == Document.source_id)
                .where(Document.is_active.is_(True))
            )
        }
        session.scalar(select(func.count()).select_from(Document))  # keep the query path warm
    return indexed <= stored


CHECKS = {
    "database": _check_database,
    "schema": _check_schema,
    "active_index": _check_index_row,
    "artifacts": _check_artifacts,
    "encoder": _check_encoder,
    "suggester": _check_suggester,
    "document_mappings": _check_mappings,
}


def run_readiness(state: AppState) -> dict:
    cached = state.readiness.get("result")
    now = state.clock()
    if cached and now - state.readiness["at"] < state.settings.readiness_cache_seconds:
        # The expensive checks may be cached, but losing the database must show up immediately.
        try:
            _check_database(state)
        except Exception:
            state.readiness.clear()
            return {"ready": False, "checks": {**cached["checks"], "database": False}}
        return cached
    results: dict[str, bool] = {}
    for name, check in CHECKS.items():
        if state.bundle is None and name not in ("database", "schema"):
            results[name] = False
            continue
        try:
            results[name] = bool(check(state))
        except (ArtifactError, Exception):  # a failing dependency means "not ready", never a 500
            results[name] = False
    outcome = {"ready": all(results.values()), "checks": results}
    state.readiness["result"], state.readiness["at"] = outcome, now
    return outcome
