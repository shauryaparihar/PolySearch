"""Database models for the corpus (Phase 1: sources, documents, ingestion jobs)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

LANGUAGE_CHECK = "language IN ('en', 'es', 'fr')"
JOB_STATUSES = ("queued", "running", "completed", "partial", "failed")
JOB_MODES = ("seed", "incremental")


class ContentSource(Base):
    __tablename__ = "content_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    base_url: Mapped[str] = mapped_column(String(255))
    licence_name: Mapped[str] = mapped_column(String(255))
    licence_url: Mapped[str] = mapped_column(String(512))
    attribution_text: Mapped[str] = mapped_column(Text)
    licence_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enabled: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "source_id", "external_id", "language", name="uq_documents_source_ext_lang"
        ),
        CheckConstraint(LANGUAGE_CHECK, name="ck_documents_language"),
        CheckConstraint("char_length(content_sha256) = 64", name="ck_documents_sha256_len"),
        Index("ix_documents_language_active", "language", "is_active"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[int] = mapped_column(ForeignKey("content_sources.id"))
    external_id: Mapped[str] = mapped_column(String(512))
    language: Mapped[str] = mapped_column(String(2))
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(String(1024))
    # NULL for documents imported from the legacy dataset, which stored no revision.
    source_revision: Mapped[str | None] = mapped_column(String(64))
    content_sha256: Mapped[str] = mapped_column(String(64))
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'partial', 'failed')",
            name="ck_ingestion_jobs_status",
        ),
        CheckConstraint("mode IN ('seed', 'incremental')", name="ck_ingestion_jobs_mode"),
        # At most one queued/running ingestion job (the single worker runs one job at a time).
        Index(
            "uq_ingestion_jobs_one_active",
            text("(true)"),
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[int] = mapped_column(ForeignKey("content_sources.id"))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    mode: Mapped[str] = mapped_column(String(16))
    language: Mapped[str | None] = mapped_column(String(2))
    continuation_state: Mapped[dict | None] = mapped_column(JSONB)
    fetched_count: Mapped[int] = mapped_column(Integer, default=0)
    created_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_count: Mapped[int] = mapped_column(Integer, default=0)
    unchanged_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    # Bounded list of {"item": ..., "code": ...}; codes are sanitized, never raw exceptions.
    failed_items: Mapped[list | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


INDEX_STATUSES = ("building", "validated", "active", "failed", "retired")
SEARCH_MODES = ("dense", "lexical", "hybrid")


class IndexVersion(Base):
    __tablename__ = "index_versions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('building', 'validated', 'active', 'failed', 'retired')",
            name="ck_index_versions_status",
        ),
        # At most one active index version, enforced by the database itself.
        Index(
            "uq_index_versions_one_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    encoder_name: Mapped[str] = mapped_column(String(255))
    encoder_revision: Mapped[str] = mapped_column(String(64))
    corpus_sha256: Mapped[str] = mapped_column(String(64))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    file_hashes: Mapped[dict] = mapped_column(JSONB)
    document_count: Mapped[int] = mapped_column(Integer)
    per_language_counts: Mapped[dict] = mapped_column(JSONB)
    vector_dimension: Mapped[int] = mapped_column(Integer)
    default_mode: Mapped[str | None] = mapped_column(String(16))
    benchmark_version: Mapped[str | None] = mapped_column(String(32))
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="building")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchRun(Base):
    __tablename__ = "search_runs"
    __table_args__ = (
        CheckConstraint("status IN ('ok', 'error')", name="ck_search_runs_status"),
        CheckConstraint(
            "requested_mode IN ('dense', 'lexical', 'hybrid')", name="ck_search_runs_mode"
        ),
        Index("ix_search_runs_query_sha256", "query_sha256"),
        Index("ix_search_runs_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_sha256: Mapped[str] = mapped_column(String(64))
    # Only filled when ANALYTICS_STORE_QUERY_TEXT=true (normalised, at most 500 characters).
    analytics_query_text: Mapped[str | None] = mapped_column(String(500))
    query_language: Mapped[str] = mapped_column(String(2))
    target_languages: Mapped[list] = mapped_column(JSONB)
    sources: Mapped[list] = mapped_column(JSONB)
    requested_mode: Mapped[str] = mapped_column(String(16))
    index_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("index_versions.id"))
    result_pool_size: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[float] = mapped_column(Numeric(10, 3), default=0)
    encoding_ms: Mapped[float | None] = mapped_column(Numeric(10, 3))
    lexical_ms: Mapped[float | None] = mapped_column(Numeric(10, 3))
    dense_ms: Mapped[float | None] = mapped_column(Numeric(10, 3))
    fusion_ms: Mapped[float | None] = mapped_column(Numeric(10, 3))
    database_ms: Mapped[float | None] = mapped_column(Numeric(10, 3))
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SearchResult(Base):
    __tablename__ = "search_results"
    __table_args__ = (
        UniqueConstraint("search_run_id", "rank", name="uq_search_results_run_rank"),
        CheckConstraint("rank >= 1", name="ck_search_results_rank_positive"),
    )

    search_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("search_runs.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Numeric(12, 8))
    dense_rank: Mapped[int | None] = mapped_column(Integer)
    dense_score: Mapped[float | None] = mapped_column(Numeric(12, 8))
    lexical_rank: Mapped[int | None] = mapped_column(Integer)
    lexical_score: Mapped[float | None] = mapped_column(Numeric(12, 8))
    # Set when the result was actually returned to a client (an impression).
    shown_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


FEEDBACK_LABELS = ("relevant", "not_relevant")


def _result_fk() -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        ["search_run_id", "document_id"],
        ["search_results.search_run_id", "search_results.document_id"],
        ondelete="CASCADE",
    )


class ClickEvent(Base):
    """A click on a result of a specific search. The composite foreign key means the database
    itself rejects clicks on documents that search never returned."""

    __tablename__ = "click_events"
    __table_args__ = (
        _result_fk(),
        UniqueConstraint("search_run_id", "idempotency_key", name="uq_click_events_run_key"),
        Index("ix_click_events_run_doc", "search_run_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    rank: Mapped[int] = mapped_column(Integer)  # rank at click time, read from the stored pool
    idempotency_key: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RelevanceFeedback(Base):
    __tablename__ = "relevance_feedback"
    __table_args__ = (
        _result_fk(),
        UniqueConstraint("search_run_id", "document_id", name="uq_relevance_feedback_run_doc"),
        CheckConstraint(
            "label IN ('relevant', 'not_relevant')", name="ck_relevance_feedback_label"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    label: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class IndexJob(Base):
    """A rebuild: export corpus -> encode -> validate -> atomically activate."""

    __tablename__ = "index_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'partial', 'failed')",
            name="ck_index_jobs_status",
        ),
        Index(
            "uq_index_jobs_one_active",
            text("(true)"),
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    step: Mapped[str] = mapped_column(String(24), default="queued")
    corpus_sha256: Mapped[str | None] = mapped_column(String(64))
    documents_total: Mapped[int] = mapped_column(Integer, default=0)
    documents_encoded: Mapped[int] = mapped_column(Integer, default=0)
    staged_version: Mapped[str | None] = mapped_column(String(64))
    index_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("index_versions.id"))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
