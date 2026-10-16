"""Ingestion service: idempotent upserts, revision-aware sync and job bookkeeping."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.db.models import ContentSource, Document, IngestionJob
from app.ingestion.sources import NormalizedRecord, PageStub, SourceError
from app.ingestion.text import bound_text, clean_text, content_sha256
from app.ingestion.wikipedia import ALLOWED_HOSTS, WikipediaSource

MAX_FAILED_ITEMS = 50
MAX_EMPTY_PAGES = 5
LEGACY_MIN_CONTENT = 300  # same bounds the legacy preprocessing used
LEGACY_MAX_CONTENT = 50_000


def _now() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------- source row
def ensure_source(session: Session, source: WikipediaSource) -> ContentSource:
    """Return the source row, creating it from *live* licence metadata if absent."""
    row = session.scalar(select(ContentSource).where(ContentSource.name == source.name))
    if row is not None:
        return row
    rights = source.fetch_rights_info()
    names = {r["name"] for r in rights.values()}
    if len(names) != 1:
        raise SourceError("rights_mismatch", "Language editions report different licences")
    licence_name = names.pop()
    licence_url = rights["en"]["url"]  # stored exactly as reported; other deeds differ by language
    row = ContentSource(
        name=source.name,
        base_url="https://www.wikipedia.org/",
        licence_name=licence_name,
        licence_url=licence_url,
        attribution_text=(
            f"Source: Wikipedia. Text is available under {licence_name}; "
            "see the linked article and its history for authorship."
        ),
        licence_verified_at=_now(),
    )
    session.add(row)
    session.flush()
    return row


# ----------------------------------------------------------------------- upsert
def upsert_document(session: Session, source_id: int, rec: NormalizedRecord) -> str:
    """Insert or update one document. Returns 'created', 'updated' or 'unchanged'.

    'unchanged' means nothing was written: same content hash, same revision, still active.
    """
    doc = session.scalar(
        select(Document).where(
            Document.source_id == source_id,
            Document.external_id == rec.external_id,
            Document.language == rec.language,
        )
    )
    if doc is None:
        session.add(
            Document(
                source_id=source_id,
                external_id=rec.external_id,
                language=rec.language,
                title=rec.title,
                summary=rec.summary,
                source_url=rec.source_url,
                source_revision=rec.source_revision,
                content_sha256=rec.content_sha256,
                source_updated_at=rec.source_updated_at,
                fetched_at=_now(),
            )
        )
        session.flush()
        return "created"
    if (
        doc.content_sha256 == rec.content_sha256
        and doc.source_revision == rec.source_revision
        and doc.is_active
    ):
        return "unchanged"
    doc.title, doc.summary = rec.title, rec.summary
    doc.source_url = rec.source_url
    doc.source_revision = rec.source_revision
    doc.content_sha256 = rec.content_sha256
    doc.source_updated_at = rec.source_updated_at
    doc.fetched_at = _now()
    doc.is_active = True
    session.flush()
    return "updated"


# -------------------------------------------------------------------- job state
class JobRun:
    """Owns one IngestionJob row and its counters/failure list."""

    def __init__(
        self, factory: sessionmaker[Session], source_id: int, mode: str, language: str | None
    ):
        self.factory = factory
        with factory() as s:
            job = IngestionJob(
                source_id=source_id,
                mode=mode,
                language=language,
                status="queued",
                fetched_count=0,
                created_count=0,
                updated_count=0,
                unchanged_count=0,
                failed_count=0,
                failed_items=[],
            )
            s.add(job)
            s.commit()
            self.job_id = job.id
        self.source_id = source_id
        self.counts = {"fetched": 0, "created": 0, "updated": 0, "unchanged": 0, "failed": 0}
        self.failed_items: list[dict[str, str]] = []
        self.continuation_state: dict | None = None
        self._update(status="running", started_at=_now())

    def fail_item(self, item: str, code: str) -> None:
        self.counts["failed"] += 1
        if len(self.failed_items) < MAX_FAILED_ITEMS:
            self.failed_items.append({"item": item[:200], "code": code})

    def _update(self, **fields: object) -> None:
        with self.factory() as s:
            job = s.get(IngestionJob, self.job_id)
            for key, value in fields.items():
                setattr(job, key, value)
            s.commit()

    def checkpoint(self) -> None:
        self._update(
            fetched_count=self.counts["fetched"],
            created_count=self.counts["created"],
            updated_count=self.counts["updated"],
            unchanged_count=self.counts["unchanged"],
            failed_count=self.counts["failed"],
            failed_items=self.failed_items,
            continuation_state=self.continuation_state,
        )

    def finish(self, error: SourceError | None = None) -> str:
        c = self.counts
        wrote_or_skipped = c["created"] + c["updated"] + c["unchanged"]
        if error is not None:
            status = "failed"
        elif c["failed"] == 0:
            status = "completed"
        elif wrote_or_skipped > 0:
            status = "partial"
        else:
            status = "failed"
        self.checkpoint()
        self._update(
            status=status,
            finished_at=_now(),
            error_code=error.code if error else None,
            error_message=error.public_message if error else None,
        )
        return status


# ------------------------------------------------------------------- sync flows
def _process_stubs(
    run: JobRun,
    source: WikipediaSource,
    language: str,
    stubs: Iterable[PageStub],
    missing: Iterable[str] = (),
    *,
    soft_delete_missing: bool = False,
) -> None:
    """Fetch only pages whose revision differs from the database, then upsert them."""
    stubs = list(stubs)
    batch = source.settings.titles_per_request
    with run.factory() as session:
        existing = {
            d.external_id: (d.source_revision, d.is_active)
            for d in session.scalars(
                select(Document).where(
                    Document.source_id == run.source_id,
                    Document.language == language,
                    Document.external_id.in_([s.title.replace(" ", "_") for s in stubs]),
                )
            )
        }
        to_fetch: list[str] = []
        for stub in stubs:
            known = existing.get(stub.title.replace(" ", "_"))
            if known and known[0] is not None and known[0] == stub.revision and known[1]:
                run.counts["unchanged"] += 1  # revision unchanged: skip the content fetch
            else:
                to_fetch.append(stub.title)
        for i in range(0, len(to_fetch), batch):
            records, not_found = source.fetch_records(language, to_fetch[i : i + batch])
            for title in not_found:
                run.fail_item(title, "page_not_found")
            for raw in records:
                run.counts["fetched"] += 1
                try:
                    rec = source.normalise_record(raw)
                except SourceError as exc:
                    run.fail_item(raw.title, exc.code)
                    continue
                run.counts[upsert_document(session, run.source_id, rec)] += 1
            session.commit()  # keep counters and stored rows in step per batch
        for title in missing:
            if soft_delete_missing:
                doc = session.scalar(
                    select(Document).where(
                        Document.source_id == run.source_id,
                        Document.language == language,
                        Document.external_id == title.replace(" ", "_"),
                    )
                )
                if doc is not None and doc.is_active:
                    doc.is_active = False
                    run.counts["updated"] += 1
                    continue
            run.fail_item(title, "page_not_found")
        session.commit()
    run.checkpoint()


def _bounded(max_pages: int | None) -> int:
    cap = get_settings().max_pages_per_job
    return cap if max_pages is None else max(1, min(max_pages, cap))


def sync_titles(
    factory: sessionmaker[Session], source: WikipediaSource, language: str, titles: list[str]
) -> str:
    """Seed: fetch specific titles (bounded). Missing titles are recorded as failures."""
    _check_language(language)
    titles = titles[: _bounded(len(titles))]
    with factory() as s:
        source_id = ensure_source(s, source).id
        s.commit()
    run = JobRun(factory, source_id, "seed", language)
    try:
        step = source.settings.titles_per_request * 2
        for i in range(0, len(titles), step):
            found = source.fetch_revisions(language, titles[i : i + step])
            stubs = [x for x in found if isinstance(x, PageStub)]
            missing = [x for x in found if isinstance(x, str)]
            _process_stubs(run, source, language, stubs, missing)
    except SourceError as exc:
        return run.finish(exc)
    return run.finish()


def sync_enumerate(
    factory: sessionmaker[Session],
    source: WikipediaSource,
    language: str,
    max_pages: int | None = None,
    resume_from: dict | None = None,
) -> str:
    """Seed: walk the wiki's page list with API continuation, at most ``max_pages`` pages."""
    _check_language(language)
    budget = _bounded(max_pages)
    with factory() as s:
        source_id = ensure_source(s, source).id
        s.commit()
    run = JobRun(factory, source_id, "seed", language)
    token = resume_from
    seen = 0
    empty_streak = 0
    try:
        while seen < budget:
            stubs, token = source.fetch_updates(language, token, min(50, budget - seen))
            _process_stubs(run, source, language, stubs)
            seen += len(stubs)
            run.continuation_state = token
            if token is None:
                break
            # The API may return an empty page that still carries a continuation token;
            # keep going, but never loop forever on a source that only returns empties.
            empty_streak = empty_streak + 1 if not stubs else 0
            if empty_streak >= MAX_EMPTY_PAGES:
                break
    except SourceError as exc:
        return run.finish(exc)
    return run.finish()


def sync_incremental(
    factory: sessionmaker[Session],
    source: WikipediaSource,
    language: str,
    max_pages: int | None = None,
) -> str:
    """Incremental: re-check the oldest-fetched stored documents against live revisions."""
    _check_language(language)
    budget = _bounded(max_pages)
    with factory() as s:
        source_id = ensure_source(s, source).id
        titles = [
            t.replace("_", " ")
            for t in s.scalars(
                select(Document.external_id)
                .where(
                    Document.source_id == source_id,
                    Document.language == language,
                    Document.is_active.is_(True),
                )
                .order_by(Document.fetched_at.asc())
                .limit(budget)
            )
        ]
        s.commit()
    run = JobRun(factory, source_id, "incremental", language)
    try:
        step = source.settings.titles_per_request * 2
        for i in range(0, len(titles), step):
            found = source.fetch_revisions(language, titles[i : i + step])
            stubs = [x for x in found if isinstance(x, PageStub)]
            missing = [x for x in found if isinstance(x, str)]
            _process_stubs(run, source, language, stubs, missing, soft_delete_missing=True)
    except SourceError as exc:
        return run.finish(exc)
    return run.finish()


def last_continuation(factory: sessionmaker[Session], language: str) -> dict | None:
    with factory() as s:
        job = s.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.language == language,
                IngestionJob.mode == "seed",
                IngestionJob.continuation_state.is_not(None),
            )
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        return job.continuation_state if job else None


# ---------------------------------------------------------------- legacy import
def import_legacy(
    factory: sessionmaker[Session],
    source: WikipediaSource,
    raw_dir: Path,
    limit: int | None = None,
) -> str:
    """Import the original raw Wikipedia JSON through the normalized document model.

    The legacy files carry no revision ids, so ``source_revision`` stays NULL for these rows;
    a later sync fills it in. Running the import twice leaves every row 'unchanged'.
    """
    with factory() as s:
        source_id = ensure_source(s, source).id
        s.commit()
    run = JobRun(factory, source_id, "seed", None)
    settings = source.settings
    try:
        for language in SUPPORTED_LANGUAGES:
            path = Path(raw_dir) / f"wikipedia_{language}.json"
            if not path.exists():
                run.fail_item(path.name, "file_missing")
                continue
            articles = json.loads(path.read_text(encoding="utf-8"))
            if limit is not None:
                articles = articles[:limit]
            with run.factory() as session:
                for article in articles:
                    try:
                        rec = _legacy_record(source, language, article, settings.summary_max_chars)
                    except (KeyError, TypeError, SourceError) as exc:
                        code = exc.code if isinstance(exc, SourceError) else "malformed_record"
                        run.fail_item(
                            str(article.get("title", "?")) if isinstance(article, dict) else "?",
                            code,
                        )
                        continue
                    if rec is None:
                        continue  # outside the legacy content-length bounds
                    run.counts["fetched"] += 1
                    run.counts[upsert_document(session, source_id, rec)] += 1
                session.commit()
            run.checkpoint()
    except SourceError as exc:
        return run.finish(exc)
    return run.finish()


def _legacy_record(
    source: WikipediaSource, language: str, article: dict, max_summary: int
) -> NormalizedRecord | None:
    content_len = len(clean_text(article["content"]))
    if content_len < LEGACY_MIN_CONTENT or content_len > LEGACY_MAX_CONTENT:
        return None
    title = clean_text(article["title"])
    summary = bound_text(clean_text(article["summary"]), max_summary)
    if not title:
        raise SourceError("empty_title", "Record has no title")
    if not summary:
        raise SourceError("empty_summary", "Record has no summary")
    return NormalizedRecord(
        external_id=title.replace(" ", "_"),
        language=language,
        title=title,
        summary=summary,
        source_url=source.canonical_url(language, title, article.get("url")),
        source_revision=None,
        source_updated_at=None,
        content_sha256=content_sha256(title, summary),
    )


def _check_language(language: str) -> None:
    if language not in ALLOWED_HOSTS:
        raise SourceError("unsupported_language", "Language is not supported")
