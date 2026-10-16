"""Safe re-indexing: export -> encode -> validate -> atomically activate.

The active index is never modified. A new immutable version is built next to it, checked
thoroughly, and only then made active in one step (database transaction, then one in-process
pointer swap). If *anything* fails the old version keeps serving and the new files are removed.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.api.logging_config import log_event
from app.api.service import register_index_version
from app.api.suggest import build_suggester
from app.db.models import ContentSource, Document, IndexJob
from app.evaluation.benchmark import load_benchmark, validate_benchmark
from app.evaluation.corpus import Corpus, export_corpus, load_corpus
from app.evaluation.runner import evaluate
from app.evidence import BenchmarkValidationError, result_problems
from app.ingestion.text import dense_text
from app.jobs.manager import transition
from app.search.artifacts import ArtifactError, build_index
from app.search.bundle import BundleRetriever, SearchBundle
from app.search.encoder import SentenceTransformerEncoder

SMOKE_TOP_K = 3
# Minimum cosine similarity of a document to its own indexed text. Dense vectors reproduce
# themselves almost exactly (observed 1.000); the lexical query differs from the indexed text
# only by punctuation (observed 0.984-1.000), so its threshold is a little lower.
SMOKE_MIN_SCORE = {"dense": 0.99, "lexical": 0.95}


class ValidationFailed(Exception):
    """A new index failed a check; ``code`` is safe to store and show."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message[:480]


class GuardedEncoder:
    """Wraps the shared encoder for the build: each batch takes an inference slot (so live
    searches interleave instead of starving), reports progress and honours shutdown."""

    def __init__(self, inner, guard, stop, cancelled_exc, on_batch):
        self._inner, self._guard, self._stop = inner, guard, stop
        self._cancelled, self._on_batch = cancelled_exc, on_batch
        self.name, self.revision, self.dimension = inner.name, inner.revision, inner.dimension

    def encode(self, texts):
        if self._stop.is_set():
            raise self._cancelled
        with self._guard.slot():
            vectors = self._inner.encode(texts)
        self._on_batch(len(texts))
        return vectors


# ---------------------------------------------------------------------- helpers
def _artifact_root(settings) -> Path:
    if settings.artifact_root:
        return Path(settings.artifact_root)
    if settings.active_index_path:
        return Path(settings.active_index_path).parent
    raise ValidationFailed("ARTIFACT_ROOT_NOT_CONFIGURED", "No artifact directory is configured.")


def _update(factory: sessionmaker, job_id: uuid.UUID, **fields: Any) -> None:
    with factory() as session:
        job = session.get(IndexJob, job_id)
        for key, value in fields.items():
            setattr(job, key, value)
        session.commit()


def _finish(factory, job_id, status: str, **fields: Any) -> None:
    with factory() as session:
        transition(session, session.get(IndexJob, job_id), status, **fields)
        session.commit()


def _active_keys(factory) -> dict[str, tuple[str, str]]:
    """document key -> (title, summary) for every active document of an enabled source."""
    with factory() as session:
        rows = session.execute(
            select(Document.language, Document.external_id, Document.title, Document.summary)
            .join(ContentSource, ContentSource.id == Document.source_id)
            .where(Document.is_active.is_(True), ContentSource.enabled.is_(True))
        )
        return {f"{lang}:{ext}": (title, summary) for lang, ext, title, summary in rows}


# -------------------------------------------------------------------- validation
def validate_new_index(
    state, path: Path, encoder, corpus: Corpus, previous: SearchBundle | None
) -> tuple[SearchBundle, dict | None]:
    """Run every check; raise ``ValidationFailed``/``ArtifactError`` on the first problem."""
    settings = state.settings
    bundle = SearchBundle.load(path, encoder=encoder)  # manifest, checksums, counts, dimensions

    db_docs = _active_keys(state.session_factory)
    if bundle.manifest["corpus"]["documents"] != len(db_docs):
        raise ValidationFailed(
            "CORPUS_COUNT_MISMATCH",
            f"index has {bundle.manifest['corpus']['documents']} documents, "
            f"database has {len(db_docs)} active documents",
        )
    if bundle.document_keys() != set(db_docs):
        raise ValidationFailed(
            "DOCUMENT_MAPPING_MISMATCH", "indexed documents do not match the database"
        )

    keys = sorted(db_docs)
    step = max(1, len(keys) // settings.reindex_smoke_documents)
    sample = keys[::step][: settings.reindex_smoke_documents]
    failures = 0
    for key in sample:
        language = key.split(":", 1)[0]
        title, summary = db_docs[key]
        # Integrity check: a document searched with the exact text that was indexed must find
        # itself with near-perfect similarity. A wrong id-to-document mapping, a vector in the
        # wrong partition or a damaged index fails this even if every file hash is valid. (A
        # title-only query is not used: short queries may legitimately rank a document lower.)
        query = dense_text(title, summary)
        for mode in ("dense", "lexical"):
            hits = bundle.search(query, mode, [language], k=SMOKE_TOP_K).hits
            found = next((h for h in hits if h.key == key), None)
            if found is None or found.score < SMOKE_MIN_SCORE[mode]:
                failures += 1
    if failures:
        raise ValidationFailed(
            "SMOKE_SEARCH_FAILED", f"{failures} smoke search(es) did not find their own document"
        )

    if not settings.benchmark_dir:
        return bundle, None  # benchmark gate not configured
    benchmark = load_benchmark(settings.benchmark_dir)
    validate_benchmark(benchmark, bundle.document_keys())  # raises BenchmarkValidationError
    results = {
        mode: evaluate(BundleRetriever(bundle, mode), benchmark, corpus)
        for mode in ("dense", "lexical", "hybrid")
    }
    for mode, result in results.items():
        problems = result_problems(result)
        if problems or result["failure_count"]:
            raise ValidationFailed(
                "BENCHMARK_RESULT_INVALID", f"{mode}: {'; '.join(problems) or 'failed queries'}"
            )
    previous_mrr = (
        ((previous.manifest.get("evaluation") or {}).get("metrics") or {}).get("dense") or {}
    ).get("mrr")
    new_mrr = results["dense"]["mrr"]
    if previous_mrr is not None and new_mrr < previous_mrr - settings.reindex_max_mrr_drop:
        raise ValidationFailed(
            "BENCHMARK_REGRESSION",
            f"dense MRR@10 fell from {previous_mrr:.4f} to {new_mrr:.4f} "
            f"(allowed drop {settings.reindex_max_mrr_drop})",
        )
    evaluation = {
        "benchmark_version": results["dense"]["benchmark"]["version"],
        "queries_sha256": results["dense"]["benchmark"]["queries_sha256"],
        "qrels_sha256": results["dense"]["benchmark"]["qrels_sha256"],
        "metrics": {
            m: {k: r[k] for k in ("mrr", "recall@10", "ndcg@10", "total_queries")}
            for m, r in results.items()
        },
    }
    return bundle, evaluation


def _finalize_manifest(path: Path, evaluation: dict | None, previous: SearchBundle | None) -> None:
    """Record the benchmark result and carry the default search mode over from the previous
    index (the mode is not re-selected here, and the manifest says so)."""
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"] = evaluation
    carried = (
        ((previous.manifest.get("default_mode") or {}).get("default_mode")) if previous else None
    )
    if carried:
        manifest["default_mode"] = {
            "default_mode": carried,
            "reason": f"carried over from {previous.version}; not re-selected by the re-index job",
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------- the job
def run_index_job(state, job_id: uuid.UUID, params: dict, stop, cancelled_exc) -> None:
    factory, settings = state.session_factory, state.settings
    created_dir: Path | None = None
    jobdir: Path | None = None
    try:
        with factory() as session:
            transition(session, session.get(IndexJob, job_id), "running", step="exporting")
            session.commit()
        root = _artifact_root(settings)
        root.mkdir(parents=True, exist_ok=True)
        jobdir = root / ".jobs" / str(job_id)
        with factory() as session:
            export_corpus(session, jobdir / "corpus.jsonl")
        corpus = load_corpus(jobdir / "corpus.jsonl")
        if not corpus.docs:
            raise ValidationFailed("EMPTY_CORPUS", "There are no active documents to index.")
        _update(
            factory,
            job_id,
            corpus_sha256=corpus.sha256,
            documents_total=len(corpus.docs),
            step="encoding",
        )

        previous, _ = state.snapshot()
        base_encoder = (
            previous.encoder
            if previous is not None and previous._encoder is not None  # noqa: SLF001
            else SentenceTransformerEncoder(settings.encoder_name, settings.encoder_revision)
        )
        done = [0]

        def progress(n: int) -> None:
            done[0] += n
            _update(factory, job_id, documents_encoded=done[0])

        existing = {p.name for p in root.iterdir() if p.is_dir()}
        guarded = GuardedEncoder(base_encoder, state.guard, stop, cancelled_exc, progress)
        # staged next to the live version, then atomically renamed into place
        path = build_index(corpus, guarded, root, batch_size=settings.index_batch_size)
        created_dir = path if path.name not in existing else None
        _update(factory, job_id, staged_version=path.name)

        if previous is not None and path.name == previous.version:
            _finish(factory, job_id, "completed", step="no_changes")
            log_event("reindex_no_changes", job_id=str(job_id), index_version=path.name)
            return

        if stop.is_set():
            raise cancelled_exc
        _update(factory, job_id, step="validating")
        bundle, evaluation = validate_new_index(state, path, base_encoder, corpus, previous)

        if stop.is_set():
            raise cancelled_exc
        _update(factory, job_id, step="activating")
        _finalize_manifest(path, evaluation, previous)
        bundle = SearchBundle.load(
            path, encoder=base_encoder
        )  # still valid after the manifest update
        suggester = build_suggester(
            state, bundle
        )  # everything that can fail happens before the swap
        with factory() as session:
            row = register_index_version(
                session, bundle
            )  # retire old + activate new, one transaction
            job = session.get(IndexJob, job_id)
            job.index_version_id = row.id
            session.commit()
            version_id = row.id
        state.activate(bundle, version_id, suggester)  # the single atomic in-process swap
        created_dir = None  # now the live version: never delete it
        _finish(factory, job_id, "completed", step="activated")
        log_event("reindex_activated", job_id=str(job_id), index_version=bundle.version)
    except (ValidationFailed, ArtifactError, BenchmarkValidationError) as exc:
        code = (
            exc.code
            if isinstance(exc, ValidationFailed)
            else (
                "BENCHMARK_INVALID"
                if isinstance(exc, BenchmarkValidationError)
                else "ARTIFACTS_INVALID"
            )
        )
        message = exc.message if isinstance(exc, ValidationFailed) else str(exc)[:480]
        _cleanup(created_dir, state)
        _finish(factory, job_id, "failed", error_code=code, error_message=message)
        log_event("reindex_failed", job_id=str(job_id), error_code=code)
    except BaseException:
        _cleanup(created_dir, state)
        raise  # the worker marks the job failed (SHUTDOWN or INTERNAL_ERROR)
    finally:
        if jobdir is not None:
            shutil.rmtree(jobdir, ignore_errors=True)


def _cleanup(created_dir: Path | None, state) -> None:
    """Remove files this job created, but never the version that is currently served."""
    if created_dir is None or not created_dir.exists():
        return
    bundle, _ = state.snapshot()
    if bundle is not None and Path(bundle.path).resolve() == created_dir.resolve():
        return
    shutil.rmtree(created_dir, ignore_errors=True)
