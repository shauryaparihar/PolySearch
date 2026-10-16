"""Database models for the corpus (Phase 1: sources, documents, ingestion jobs)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
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
