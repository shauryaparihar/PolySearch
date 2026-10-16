import json

import httpx
from app.db.models import ContentSource, Document, IngestionJob
from app.ingestion import service
from app.ingestion.sources import NormalizedRecord
from app.ingestion.text import content_sha256
from sqlalchemy import select


def _rec(title="Mathematics", summary="Numbers.", revision="1", language="en"):
    return NormalizedRecord(
        external_id=title.replace(" ", "_"),
        language=language,
        title=title,
        summary=summary,
        source_url=f"https://{language}.wikipedia.org/wiki/{title}",
        source_revision=revision,
        source_updated_at=None,
        content_sha256=content_sha256(title, summary),
    )


def _src_id(session):
    src = ContentSource(
        name="wikipedia", base_url="x", licence_name="l", licence_url="u", attribution_text="a"
    )
    session.add(src)
    session.flush()
    return src.id


def test_upsert_is_idempotent(session_factory):
    with session_factory() as s:
        sid = _src_id(s)
        assert service.upsert_document(s, sid, _rec()) == "created"
        assert service.upsert_document(s, sid, _rec()) == "unchanged"
        assert len(s.scalars(select(Document)).all()) == 1


def test_unchanged_revision_does_not_write_the_row(session_factory):
    with session_factory() as s:
        sid = _src_id(s)
        service.upsert_document(s, sid, _rec())
        before = s.scalar(select(Document.fetched_at))
        service.upsert_document(s, sid, _rec())
        assert s.scalar(select(Document.fetched_at)) == before


def test_changed_content_or_revision_updates_and_reactivates(session_factory):
    with session_factory() as s:
        sid = _src_id(s)
        service.upsert_document(s, sid, _rec())
        assert service.upsert_document(s, sid, _rec(summary="New text.", revision="2")) == "updated"
        doc = s.scalar(select(Document))
        assert (doc.summary, doc.source_revision) == ("New text.", "2")
        doc.is_active = False
        assert service.upsert_document(s, sid, _rec(summary="New text.", revision="2")) == "updated"
        assert doc.is_active


def test_sync_titles_stores_provenance_and_licence_end_to_end(session_factory, source):
    status = service.sync_titles(session_factory, source, "en", ["Mathematics", "Missing page"])
    assert status == "partial"  # one title does not exist
    with session_factory() as s:
        doc = s.scalar(select(Document).where(Document.external_id == "Mathematics"))
        assert doc.source_revision == "11"
        assert doc.source_url == "https://en.wikipedia.org/wiki/Mathematics"
        src = s.get(ContentSource, doc.source_id)
        assert src.licence_name == "Creative Commons Attribution-Share Alike 4.0"
        assert src.licence_url.startswith("https://creativecommons.org/")
        assert src.licence_verified_at is not None
        job = s.scalar(select(IngestionJob))
        assert (job.created_count, job.failed_count) == (1, 1)
        assert job.failed_items == [{"item": "Missing page", "code": "page_not_found"}]


def test_second_sync_skips_unchanged_revisions_without_fetching_text(
    session_factory, source, fake_wiki
):
    service.sync_titles(session_factory, source, "en", ["Mathematics", "Physics"])
    fake_wiki.requests.clear()
    service.sync_titles(session_factory, source, "en", ["Mathematics", "Physics"])
    with session_factory() as s:
        job = s.scalar(select(IngestionJob).order_by(IngestionJob.created_at.desc()))
        assert (job.fetched_count, job.unchanged_count) == (0, 2)
    assert all("extracts" not in str(r.url) for r in fake_wiki.requests)


def test_enumeration_follows_continuation_and_respects_the_page_budget(session_factory, source):
    assert service.sync_enumerate(session_factory, source, "en", max_pages=2) == "completed"
    with session_factory() as s:
        assert {d.title for d in s.scalars(select(Document))} == {"Mathematics", "Physics"}
        token = service.last_continuation(session_factory, "en")
        assert token and token["gapcontinue"] == "Zebra"
    service.sync_enumerate(session_factory, source, "en", max_pages=5, resume_from=token)
    with session_factory() as s:
        assert len(s.scalars(select(Document)).all()) == 3


def test_empty_page_with_continuation_does_not_stop_enumeration(session_factory, source, fake_wiki):
    with session_factory() as s:  # create the source row first so licence lookups use no queue slot
        service.ensure_source(s, source)
        s.commit()
    empty_with_token = httpx.Response(
        200,
        json={
            "query": {"pages": []},
            "continue": {"gapcontinue": "Zebra", "continue": "gapcontinue||"},
        },
    )
    fake_wiki.fail_next = [empty_with_token]  # first page request returns an empty page + token
    service.sync_enumerate(session_factory, source, "en", max_pages=3)
    with session_factory() as s:
        assert {d.title for d in s.scalars(select(Document))} == {"Zebra"}


def test_incremental_refresh_updates_changed_pages_and_soft_deletes_missing(
    session_factory, source, wiki_pages
):
    service.sync_titles(session_factory, source, "en", ["Mathematics", "Physics"])
    wiki_pages["en"]["Mathematics"] = {"revid": 99, "extract": "Mathematics, revised."}
    del wiki_pages["en"]["Physics"]
    assert service.sync_incremental(session_factory, source, "en") == "completed"
    with session_factory() as s:
        docs = {d.title: d for d in s.scalars(select(Document))}
        assert docs["Mathematics"].summary == "Mathematics, revised."
        assert docs["Physics"].is_active is False  # soft-deleted, row kept


def test_legacy_import_is_idempotent_and_records_bad_rows(session_factory, source, tmp_path):
    article = {
        "title": "Medicina",
        "summary": "La medicina es una ciencia.",
        "content": "x" * 400,
        "url": "https://es.wikipedia.org/wiki/Medicina",
        "language": "es",
        "categories": [],
    }
    bad = {
        "title": "Sin resumen",
        "summary": "",
        "content": "x" * 400,
        "url": "u",
        "language": "es",
    }
    broken = {"title": "Falta contenido"}
    short = {"title": "Corto", "summary": "s", "content": "x", "url": "u", "language": "es"}
    (tmp_path / "wikipedia_es.json").write_text(
        json.dumps([article, bad, broken, short]), encoding="utf-8"
    )
    first = service.import_legacy(session_factory, source, tmp_path)
    second = service.import_legacy(session_factory, source, tmp_path)
    assert first == second == "partial"  # missing en/fr files + 2 bad rows are reported, not hidden
    with session_factory() as s:
        docs = s.scalars(select(Document)).all()
        assert [(d.title, d.source_revision) for d in docs] == [
            ("Medicina", None)
        ]  # no invented revision
        jobs = s.scalars(select(IngestionJob).order_by(IngestionJob.created_at)).all()
        assert jobs[0].created_count == 1 and jobs[1].unchanged_count == 1
        codes = {f["code"] for f in jobs[0].failed_items}
        assert {"empty_summary", "malformed_record", "file_missing"} <= codes
