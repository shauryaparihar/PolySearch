"""Keeps the optional passage artifact in step with the served search index.

Every function here returns a result instead of raising: a missing or broken passage artifact
must only ever switch answer mode off, never affect search or a re-index.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from app.api.logging_config import log_event
from app.config import Settings
from app.db.models import DocumentPassage, IndexVersion
from app.rag.passage_store import (
    PassageArtifactError,
    PassageStore,
    build_passage_artifact,
    passage_dir,
    sync_passages,
)
from app.rag.passages import ChunkConfig
from app.search.bundle import SearchBundle

# Stable codes explaining why answer mode has no passages (shown by /readyz/rag).
RAG_DISABLED = "RAG_DISABLED"
PASSAGES_MISSING = "PASSAGES_MISSING"
PASSAGES_INVALID = "PASSAGES_INVALID"
PASSAGES_PENDING = "PASSAGES_PENDING"  # an index was activated and its passages are not built
PASSAGES_BUILD_FAILED = "PASSAGES_BUILD_FAILED"
PASSAGES_SYNC_FAILED = "PASSAGES_SYNC_FAILED"


def artifact_root(settings: Settings, bundle: SearchBundle) -> Path:
    return settings.artifact_root or Path(bundle.path).parent


def chunk_config(settings: Settings) -> ChunkConfig:
    return ChunkConfig.from_max_tokens(settings.rag_max_passage_tokens)


def _encoder_identity(bundle: SearchBundle) -> tuple[str, str]:
    enc = bundle.manifest["encoder"]
    return enc["name"], enc["revision"]


def load_passages(
    settings: Settings, bundle: SearchBundle
) -> tuple[PassageStore | None, str | None]:
    """Load the passages of the served index: ``(store, None)`` or ``(None, reason_code)``."""
    if not settings.rag_enabled:
        return None, RAG_DISABLED
    path = passage_dir(artifact_root(settings, bundle), bundle.version)
    if not (path / "manifest.json").is_file():
        return None, PASSAGES_MISSING
    try:
        store = PassageStore.load(
            path, expect_index_version=bundle.version, expect_encoder=_encoder_identity(bundle)
        )
    except (PassageArtifactError, OSError):
        log_event("passages_load_failed", level=40, error_code=PASSAGES_INVALID)
        return None, PASSAGES_INVALID
    return store, None


def build_passages_for(
    settings: Settings, bundle: SearchBundle, corpus, encoder, previous: PassageStore | None = None
) -> tuple[PassageStore | None, str | None]:
    """Build, validate and load the passages of a *candidate* index. Never raises.

    ``previous`` is the currently served artifact; its vectors are reused for unchanged text."""
    if not settings.rag_enabled:
        return None, RAG_DISABLED
    stats: dict = {}
    try:
        path = build_passage_artifact(
            corpus,
            encoder,
            bundle.version,
            bundle.manifest["corpus"]["sha256"],
            artifact_root(settings, bundle),
            chunk_config(settings),
            reuse=previous,
            stats=stats,
        )
        log_event(
            "passages_built",
            passages_encoded=stats.get("encoded", 0),
            passages_reused=stats.get("reused", 0),
        )
        store = PassageStore.load(
            path, expect_index_version=bundle.version, expect_encoder=_encoder_identity(bundle)
        )
        return store, None
    except Exception:  # noqa: BLE001  (any failure here only disables answer mode)
        log_event("passages_build_failed", level=40, error_code=PASSAGES_BUILD_FAILED)
        return None, PASSAGES_BUILD_FAILED


def sync_to_database(session_factory, store: PassageStore) -> bool:
    """Write the store's passages to PostgreSQL for the registered index version."""
    try:
        with session_factory() as session:
            version = session.scalar(
                select(IndexVersion).where(IndexVersion.version == store.index_version)
            )
            if version is None:
                return False
            sync_passages(session, store, version)
            session.commit()
        return True
    except Exception:  # noqa: BLE001
        log_event("passages_sync_failed", level=40, error_code=PASSAGES_SYNC_FAILED)
        return False


def sync_if_missing(session_factory, store: PassageStore) -> str:
    """At startup: copy the passages to PostgreSQL when none are stored for this index version.

    Returns ``"present"``, ``"synced"`` or ``"failed"``. Never raises; a failure only leaves
    answer mode falling back, which ``/readyz/rag`` reports."""
    try:
        with session_factory() as session:
            version = session.scalar(
                select(IndexVersion).where(IndexVersion.version == store.index_version)
            )
            if version is None:
                return "failed"
            stored = session.scalar(
                select(func.count())
                .select_from(DocumentPassage)
                .where(DocumentPassage.index_version_id == version.id)
            )
            if stored:
                return "present"
    except Exception:  # noqa: BLE001
        log_event("passages_sync_failed", level=40, error_code=PASSAGES_SYNC_FAILED)
        return "failed"
    return "synced" if sync_to_database(session_factory, store) else "failed"


def prune_passage_artifacts(root: Path, keep: set[str], delete: bool = False) -> list[str]:
    """Passage artifacts whose index is gone (or not in ``keep``). Dry run unless ``delete``.

    A passage directory is kept when its index version directory still exists under
    ``root`` or the version is in ``keep`` (always include the served version)."""
    import shutil

    base = Path(root) / "passages"
    if not base.is_dir():
        return []
    removable = [
        d.name
        for d in sorted(base.iterdir())
        if d.is_dir() and d.name not in keep and not (Path(root) / d.name).is_dir()
    ]
    if delete:
        for name in removable:
            shutil.rmtree(base / name)
    return removable
