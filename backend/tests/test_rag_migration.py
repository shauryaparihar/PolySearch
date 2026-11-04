"""Migration 0005: upgrade, downgrade, and the database-level rules for answer data."""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from alembic import command
from api_support import seed_documents
from app.db.models import (
    Document,
    DocumentPassage,
    GenerationCitation,
    GenerationFeedback,
    GenerationRun,
    IndexVersion,
    LlmUsageDay,
)
from app.search.artifacts import file_sha256  # noqa: F401  (keeps import order stable)
from conftest import reset_schema
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

RAG_TABLES = {
    "document_passages", "generation_runs", "generation_citations", "generation_feedback",
    "llm_usage_days", "llm_reservations",
}  # fmt: skip


def test_0005_upgrades_and_downgrades_cleanly(engine, alembic_config):
    reset_schema(engine)
    command.upgrade(alembic_config, "head")
    assert RAG_TABLES <= set(inspect(engine).get_table_names())
    command.downgrade(alembic_config, "0004")
    names = set(inspect(engine).get_table_names())
    assert not RAG_TABLES & names and {"documents", "index_versions", "index_jobs"} <= names
    command.upgrade(alembic_config, "head")  # leave the schema usable for other tests
    assert RAG_TABLES <= set(inspect(engine).get_table_names())


@pytest.fixture
def rag_db(session_factory, api_bundle):
    from app.api.service import register_index_version

    seed_documents(session_factory)
    with session_factory() as session:
        version = register_index_version(session, api_bundle)
        document = session.scalars(select(Document)).first()
        session.commit()
        return session_factory, version.id, document.id


def _passage(version_id, document_id, ordinal=0, **overrides):
    values = dict(
        index_version_id=version_id, document_id=document_id, ordinal=ordinal,
        text="A passage.", language="en", estimated_tokens=3, content_sha256="a" * 64,
    )  # fmt: skip
    values.update(overrides)
    return DocumentPassage(**values)


def _run(**overrides):
    values = dict(
        question_sha256="b" * 64, languages=["en"], prompt_version="p1",
        context_builder_version="c1", status="answered", abstained=False, fallback_used=False,
        cache_status="miss",
    )  # fmt: skip
    values.update(overrides)
    return GenerationRun(**values)


def _rejected(factory, *rows):
    with factory() as session:
        session.add_all(rows)
        with pytest.raises(IntegrityError):
            session.commit()


def test_passages_are_unique_per_version_document_and_ordinal(rag_db):
    factory, version_id, document_id = rag_db
    with factory() as session:
        session.add(_passage(version_id, document_id, 0))
        session.commit()
    _rejected(factory, _passage(version_id, document_id, 0))
    with factory() as session:  # another ordinal is fine
        session.add(_passage(version_id, document_id, 1))
        session.commit()


@pytest.mark.parametrize("bad", [{"language": "de"}, {"ordinal": -1}, {"content_sha256": "short"}])
def test_passage_rules_are_enforced_by_the_database(rag_db, bad):
    factory, version_id, document_id = rag_db
    _rejected(factory, _passage(version_id, document_id, **bad))


def test_abstention_and_fallback_flags_must_match_status(rag_db):
    factory, *_ = rag_db
    with factory() as session:  # the three valid shapes
        session.add_all([
            _run(),
            _run(status="abstained", abstained=True, abstention_reason="INSUFFICIENT_EVIDENCE"),
            _run(status="fallback", fallback_used=True, fallback_reason="PROVIDER_ERROR"),
        ])  # fmt: skip
        session.commit()
    for bad in (
        dict(status="fallback", fallback_used=False),
        dict(status="abstained", abstained=True, fallback_used=True),
        dict(status="answered", abstained=True),
        dict(status="unknown"),
        dict(cache_status="stale"),
        dict(retrieval_mode="magic"),
    ):
        _rejected(factory, _run(**bad))


def test_citations_are_unique_and_cascade_with_their_run(rag_db):
    factory, version_id, document_id = rag_db
    with factory() as session:
        passage, run = _passage(version_id, document_id), _run()
        session.add_all([passage, run])
        session.flush()
        cite = dict(generation_run_id=run.id, passage_id=passage.id, label="P1",
                    claim_ordinal=0, retrieval_rank=1, retrieval_score=0.5)  # fmt: skip
        session.add(GenerationCitation(**cite))
        session.commit()
        run_id = run.id
    _rejected(factory, GenerationCitation(**cite))  # same run, claim and label
    _rejected(factory, GenerationCitation(**{**cite, "claim_ordinal": -1, "label": "P2"}))
    _rejected(factory, GenerationCitation(**{**cite, "retrieval_rank": 0, "label": "P3"}))
    with factory() as session:
        session.delete(session.get(GenerationRun, run_id))
        session.commit()
        assert session.scalar(text("select count(*) from generation_citations")) == 0


def test_feedback_is_one_closed_rating_per_run(rag_db):
    factory, *_ = rag_db
    with factory() as session:
        run = _run()
        session.add(run)
        session.flush()
        session.add(GenerationFeedback(generation_run_id=run.id, rating="helpful"))
        session.commit()
        run_id = run.id
    _rejected(factory, GenerationFeedback(generation_run_id=run_id, rating="not_helpful"))
    with factory() as session:
        other = _run()
        session.add(other)
        session.flush()
        session.add(GenerationFeedback(generation_run_id=other.id, rating="meh"))
        with pytest.raises(IntegrityError):
            session.commit()
    _rejected(
        factory, GenerationFeedback(generation_run_id=uuid.uuid4(), rating="helpful")
    )  # no such run


def test_deleting_an_index_version_removes_its_passages_but_not_documents(rag_db):
    factory, version_id, document_id = rag_db
    with factory() as session:
        session.add(_passage(version_id, document_id))
        session.commit()
        session.delete(session.get(IndexVersion, version_id))
        session.commit()
        assert session.scalar(text("select count(*) from document_passages")) == 0
        assert session.get(Document, document_id) is not None


def test_daily_usage_rejects_negative_counters(rag_db):
    factory, *_ = rag_db
    with factory() as session:
        session.add(LlmUsageDay(usage_date=date(2026, 1, 1), reserved_tokens=0, used_tokens=0))
        session.commit()
    _rejected(factory, LlmUsageDay(usage_date=date(2026, 1, 2), reserved_tokens=-1, used_tokens=0))
