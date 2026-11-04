"""Keeps the optional passage artifact in step with the served search index.

Every function here returns a result instead of raising: a missing or broken passage artifact
must only ever switch answer mode off, never affect search or a re-index.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app.api.logging_config import log_event
from app.config import Settings
from app.db.models import IndexVersion
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
    settings: Settings, bundle: SearchBundle, corpus, encoder
) -> tuple[PassageStore | None, str | None]:
    """Build, validate and load the passages of a *candidate* index. Never raises."""
    if not settings.rag_enabled:
        return None, RAG_DISABLED
    try:
        path = build_passage_artifact(
            corpus,
            encoder,
            bundle.version,
            bundle.manifest["corpus"]["sha256"],
            artifact_root(settings, bundle),
            chunk_config(settings),
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
