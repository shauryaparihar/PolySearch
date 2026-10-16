import threading
import time

import httpx
import pytest
from api_support import SEARCH, build_bundle
from app.api.service import register_index_version
from app.config import Settings
from app.db.models import IndexJob, IndexVersion, IngestionJob, SearchRun
from app.evaluation.corpus import export_corpus, load_corpus
from app.ingestion.wikipedia import WikipediaSource
from app.search.artifacts import build_index
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder
from fastapi.testclient import TestClient
from job_support import HEAD, KEY, add_document, get_job, start_index_job, version_dirs, wait_for
from sqlalchemy import select


def _bundle_b(session_factory, encoder, tmp_path):
    """A second bundle built from the current database (one more document than the fixture's)."""
    with session_factory() as s:
        export_corpus(s, tmp_path / "b" / "corpus.jsonl")
    corpus = load_corpus(tmp_path / "b" / "corpus.jsonl")
    return SearchBundle.load(
        build_index(corpus, encoder, tmp_path / "b" / "index"), encoder=encoder
    )


# ---------------------------------------------------------- atomic pointer swap
def test_searches_during_an_activation_always_see_one_consistent_index(
    make_client, session_factory, api_bundle, encoder, tmp_path
):
    client = make_client()
    state = client.app.state.polysearch
    add_document(session_factory, "Zebra Rayada", "es")
    new_bundle = _bundle_b(session_factory, encoder, tmp_path)
    with session_factory() as s:
        new_id = register_index_version(s, new_bundle).id
        s.commit()
    observed, errors, stop = [], [], threading.Event()

    def searcher(n):
        try:
            with TestClient(client.app) as c:
                i = 0
                while not stop.is_set():
                    r = c.post("/api/v1/search", json={**SEARCH, "query": f"volcano {n} {i}"})
                    assert r.status_code == 200, r.text
                    body = r.json()
                    observed.append(
                        (
                            body["search_id"],
                            body["index_version"],
                            [x["source_url"] for x in body["results"]],
                        )
                    )
                    i += 1
        except Exception as exc:  # collected and asserted below
            errors.append(repr(exc))

    threads = [threading.Thread(target=searcher, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    time.sleep(0.3)
    state.activate(new_bundle, new_id, None)  # the swap, while searches are running
    time.sleep(0.3)
    stop.set()
    [t.join() for t in threads]
    assert errors == [] and len(observed) > 20
    versions = {v for _, v, _ in observed}
    assert versions <= {api_bundle.version, new_bundle.version} and new_bundle.version in versions
    with session_factory() as s:
        for search_id, version, urls in observed:
            run = s.get(SearchRun, search_id)
            assert s.get(IndexVersion, run.index_version_id).version == version  # never mixed
            if version == api_bundle.version:
                assert not any(
                    "Zebra" in u for u in urls
                )  # the old index cannot return the new doc


# ------------------------------------------------------- real worker thread
def test_the_worker_thread_runs_a_job_and_stops_cleanly(make_client, session_factory, api_bundle):
    client = make_client(admin_api_key=KEY, synchronous_jobs=False)
    worker = client.app.state.polysearch.worker
    add_document(session_factory, "Zebra Rayada", "es")
    job_id = start_index_job(client)
    job = wait_for(client, "index", job_id)
    assert job["status"] == "completed" and job["step"] == "activated"
    assert worker._thread is not None and worker._thread.is_alive()
    worker.stop(5)
    assert worker._thread is None  # joined


def test_a_job_interrupted_by_shutdown_is_failed_and_leaves_no_partial_files(
    make_client, session_factory, tmp_path
):
    class SlowEncoder(FakeEncoder):
        def encode(self, texts):
            time.sleep(0.15)
            return super().encode(texts)

    slow = build_bundle(tmp_path, SlowEncoder())
    client = make_client(
        bundle=slow, admin_api_key=KEY, synchronous_jobs=False, index_batch_size=4,
        job_shutdown_timeout_seconds=10, artifact_root=slow.path.parent,
    )  # fmt: skip
    worker = client.app.state.polysearch.worker
    add_document(session_factory, "Zebra Rayada", "es")
    job_id = start_index_job(client)
    deadline = time.time() + 20
    while get_job(client, "index", job_id)["documents_encoded"] == 0:  # wait until it is mid-build
        assert time.time() < deadline
        time.sleep(0.02)
    worker.stop(10)  # what the lifespan does on shutdown
    with session_factory() as s:
        job = s.get(IndexJob, job_id)
        assert (job.status, job.error_code) == ("failed", "SHUTDOWN") and job.finished_at
        assert (
            s.scalar(select(IndexVersion.version).where(IndexVersion.status == "active"))
            == slow.version
        )
    root = slow.path.parent
    assert version_dirs(root) == [slow.version]  # the half-built version was removed
    assert not list((root / ".jobs").iterdir()) and not list(root.glob(".staging-*"))
    assert client.post("/api/v1/search", json=SEARCH).status_code == 200  # old index still serves


def test_a_queued_job_that_never_started_is_failed_at_shutdown(session_factory, make_client):
    client = make_client(admin_api_key=KEY, synchronous_jobs=False)
    worker = client.app.state.polysearch.worker
    worker.stop(2)  # worker gone; a submission can no longer run
    with session_factory() as s:
        job = IndexJob(status="queued")
        s.add(job)
        s.commit()
        job_id = job.id
    worker._queue.put_nowait(("index", job_id, {}))
    worker._fail_unstarted()
    with session_factory() as s:
        assert s.get(IndexJob, job_id).error_code == "SHUTDOWN"


def test_a_crashing_job_fails_safely_and_the_worker_keeps_working(
    make_client, session_factory, monkeypatch
):
    client = make_client(admin_api_key=KEY)
    from app.jobs import reindex

    def explode(*args, **kwargs):
        raise RuntimeError("secret detail: postgresql://user:password@host/db")

    monkeypatch.setattr(reindex, "export_corpus", explode)
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "failed" and job["error_code"] == "INTERNAL_ERROR"
    assert "password" not in str(job)
    monkeypatch.undo()
    assert (
        get_job(client, "index", start_index_job(client))["status"] == "completed"
    )  # worker survived


# --------------------------------------------------------------- ingestion jobs
@pytest.fixture
def wiki_client(make_client, fake_wiki):
    def source_factory():
        return WikipediaSource(
            Settings(_env_file=None, http_max_retries=0),
            transport=fake_wiki.transport(),
            sleep=lambda _s: None,
            jitter=lambda: 0.0,
        )

    return make_client(admin_api_key=KEY, source_factory=source_factory)


def _ingest(client, **body):
    return client.post("/api/v1/admin/ingestion-jobs", json=body, headers=HEAD)


def test_an_ingestion_job_syncs_titles_and_reports_counters(wiki_client, session_factory):
    r = _ingest(
        wiki_client, language="en", mode="seed", titles=["Mathematics", "Physics", "Missing page"]
    )
    assert r.status_code == 202 and r.headers["location"].endswith(r.json()["job_id"])
    job = get_job(wiki_client, "ingestion", r.json()["job_id"])
    assert job["status"] == "partial" and job["kind"] == "ingestion" and job["language"] == "en"
    assert job["counters"]["created"] == 2 and job["counters"]["failed"] == 1
    assert job["failed_items"] == [{"item": "Missing page", "code": "page_not_found"}]
    with session_factory() as s:
        assert (
            s.scalar(select(IngestionJob).where(IngestionJob.id == job["job_id"])).status
            == "partial"
        )


def test_incremental_and_enumeration_jobs_run_through_the_same_endpoint(wiki_client):
    seed = get_job(
        wiki_client,
        "ingestion",
        _ingest(wiki_client, language="en", mode="seed", max_pages=5).json()["job_id"],
    )
    assert (
        seed["status"] == "completed" and seed["counters"]["created"] == 3
    )  # the fake wiki's pages
    inc = get_job(
        wiki_client,
        "ingestion",
        _ingest(wiki_client, language="en", mode="incremental").json()["job_id"],
    )
    assert inc["status"] == "completed" and inc["counters"]["unchanged"] >= 3


@pytest.mark.parametrize(
    "body",
    [
        {"language": "de", "mode": "seed"},
        {"language": "en", "mode": "weekly"},
        {"language": "en", "mode": "incremental", "titles": ["Mathematics"]},
        {"language": "en", "mode": "incremental", "resume": True},
        {"language": "en", "mode": "seed", "titles": ["A|B"]},
        {"language": "en", "mode": "seed", "titles": []},
        {"language": "en", "mode": "seed", "titles": ["x"] * 51},
        {"language": "en", "mode": "seed", "titles": ["x" * 300]},
        {"language": "en", "mode": "seed", "max_pages": 0},
        {"language": "en", "mode": "seed", "max_pages": 5000},
        {"language": "en", "mode": "seed", "url": "https://evil.example"},
    ],
)
def test_ingestion_requests_are_validated_and_bounded(wiki_client, body):
    r = _ingest(wiki_client, **body)
    assert r.status_code == 422, body
    with wiki_client.app.state.polysearch.session_factory() as s:
        assert s.scalar(select(IngestionJob)) is None  # nothing was queued


def test_a_source_failure_is_recorded_as_a_failed_job_with_a_safe_code(make_client):
    def broken():
        def boom(request):
            raise httpx.ConnectError("no route", request=request)

        return WikipediaSource(
            Settings(_env_file=None, http_max_retries=0),
            transport=httpx.MockTransport(boom),
            sleep=lambda _s: None,
        )

    client = make_client(admin_api_key=KEY, source_factory=broken)
    job = get_job(
        client,
        "ingestion",
        _ingest(client, language="en", mode="seed", titles=["Mathematics"]).json()["job_id"],
    )
    assert job["status"] == "failed" and job["error_code"] == "network_error"
    assert "no route" not in str(job)


def test_ingestion_needs_an_existing_content_source(session_factory, api_bundle):
    from api_support import test_settings
    from app.main import create_app

    app = create_app(
        test_settings(admin_api_key=KEY),
        bundle=api_bundle,
        session_factory=session_factory,
        synchronous_jobs=True,
    )
    with TestClient(app) as c:
        r = _ingest(c, language="en", mode="seed")
        assert r.status_code == 409 and r.json()["error"]["code"] == "SOURCE_NOT_INITIALISED"


def test_unknown_or_malformed_job_ids_are_not_found(wiki_client):
    for kind in ("index", "ingestion"):
        for job_id in ("not-a-uuid", "00000000-0000-0000-0000-000000000000"):
            r = wiki_client.get(f"/api/v1/admin/{kind}-jobs/{job_id}", headers=HEAD)
            assert r.status_code == 404 and r.json()["error"]["code"] == "JOB_NOT_FOUND"
    # an ingestion job id is not an index job id
    jid = _ingest(wiki_client, language="en", mode="incremental").json()["job_id"]
    assert wiki_client.get(f"/api/v1/admin/index-jobs/{jid}", headers=HEAD).status_code == 404


def test_empty_corpus_exports_load_as_empty(tmp_path, session_factory):
    from app.evaluation.corpus import export_corpus, load_corpus

    with session_factory() as s:
        manifest = export_corpus(s, tmp_path / "c.jsonl")
    assert manifest["documents"] == 0 and (tmp_path / "c.jsonl").read_bytes() == b""
    assert load_corpus(tmp_path / "c.jsonl").docs == []
