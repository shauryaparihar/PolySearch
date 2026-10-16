import pytest
from api_support import SEARCH
from app.config import Settings
from app.db.models import ContentSource, Document, IndexJob, IndexVersion, IngestionJob
from app.jobs import reindex
from app.jobs.manager import RESTART_CODE, recover_stale_jobs, transition
from app.jobs.states import InvalidTransition, check_transition
from app.search.bundle import SearchBundle
from job_support import HEAD, KEY, add_document, get_job, start_index_job, version_dirs
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def admin(make_client):
    return make_client(admin_api_key=KEY)


def _active_version(factory) -> str:
    with factory() as s:
        return s.scalar(select(IndexVersion.version).where(IndexVersion.status == "active"))


def _status_counts(factory):
    with factory() as s:
        rows = s.execute(select(IndexVersion.status, func.count()).group_by(IndexVersion.status))
        return {status: count for status, count in rows}


# ------------------------------------------------------------------ transitions
def test_valid_transitions():
    for old, new in (("queued", "running"), ("running", "completed"), ("running", "partial"),
                     ("running", "failed"), ("queued", "failed")):  # fmt: skip
        check_transition(old, new)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("queued", "completed"),
        ("queued", "partial"),
        ("running", "queued"),
        ("completed", "running"),
        ("failed", "running"),
        ("partial", "completed"),
        ("completed", "failed"),
    ],
)
def test_invalid_transitions_are_refused(old, new):
    with pytest.raises(InvalidTransition):
        check_transition(old, new)


def test_a_finished_job_cannot_be_changed_again(session_factory):
    with session_factory() as s:
        job = IndexJob(status="queued")
        s.add(job)
        s.flush()
        transition(s, job, "running")
        transition(s, job, "completed")
        assert job.started_at and job.finished_at
        with pytest.raises(InvalidTransition):
            transition(s, job, "running")


# --------------------------------------------------------------- admin protection
def test_job_endpoints_need_the_admin_key(make_client):
    admin = make_client(admin_api_key=KEY)
    calls = [
        ("post", "/api/v1/admin/index-jobs", {}),
        ("get", "/api/v1/admin/index-jobs/00000000-0000-0000-0000-000000000000", None),
        ("post", "/api/v1/admin/ingestion-jobs", {"language": "en", "mode": "incremental"}),
        ("get", "/api/v1/admin/ingestion-jobs/00000000-0000-0000-0000-000000000000", None),
    ]
    for method, url, body in calls:
        kwargs = {"json": body} if body is not None else {}
        assert getattr(admin, method)(url, **kwargs).status_code == 401  # no key
        assert (
            getattr(admin, method)(url, headers={"x-admin-key": "wrong"}, **kwargs).status_code
            == 401
        )
    assert admin.get(calls[1][1], headers=HEAD).status_code == 404  # key ok, no such job
    assert not admin.app.state.polysearch.worker._queue.qsize()  # nothing was started


def test_job_endpoints_do_not_exist_without_a_configured_key(client):
    assert client.post("/api/v1/admin/index-jobs", json={}, headers=HEAD).status_code == 404


# ------------------------------------------------------------- the happy path
def test_reindex_builds_validates_and_atomically_activates_a_new_version(
    admin, session_factory, api_bundle
):
    old_version = api_bundle.version
    add_document(session_factory, "Zebra Rayada", "es")
    before = admin.post("/api/v1/search", json=SEARCH).json()
    assert before["index_version"] == old_version and before["cache_hit"] is False
    job_id = start_index_job(admin)
    job = get_job(admin, "index", job_id)
    assert job["status"] == "completed" and job["step"] == "activated"  # synchronous test worker
    assert job["documents_total"] == job["documents_encoded"] == 25
    new_version = job["index_version"]
    assert new_version != old_version and job["staged_version"] == new_version
    assert "/" not in str(job)  # a version name, never a filesystem path

    after = admin.post("/api/v1/search", json=SEARCH).json()
    assert (
        after["index_version"] == new_version and after["cache_hit"] is False
    )  # old cache unreachable
    assert admin.get("/api/v1/stats").json()["index_version"] == new_version
    assert admin.get("/readyz").status_code == 200
    titles = [
        s["text"]
        for s in admin.get("/api/v1/suggestions", params={"prefix": "zebra"}).json()["suggestions"]
    ]
    assert titles == ["Zebra Rayada"]  # the autocomplete trie was rebuilt for the new index
    assert _status_counts(session_factory) == {"active": 1, "retired": 1}
    assert _active_version(session_factory) == new_version
    assert set(version_dirs(api_bundle.path.parent)) == {
        old_version,
        new_version,
    }  # old kept for rollback


def test_an_unchanged_corpus_completes_without_activating_anything(
    admin, session_factory, api_bundle
):
    job = get_job(admin, "index", start_index_job(admin))
    assert (job["status"], job["step"]) == ("completed", "no_changes")
    assert _active_version(session_factory) == api_bundle.version
    assert version_dirs(api_bundle.path.parent) == [api_bundle.version]


def test_a_previously_retired_version_can_be_activated_again(admin, session_factory, api_bundle):
    original = api_bundle.version
    key = add_document(session_factory, "Zebra Rayada", "es")
    get_job(admin, "index", start_index_job(admin))
    with session_factory() as s:  # remove the extra document again
        from app.db.models import Document

        s.execute(
            Document.__table__.update()
            .where(Document.external_id == key.split(":", 1)[1])
            .values(is_active=False)
        )
        s.commit()
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "completed" and job["index_version"] == original
    assert _active_version(session_factory) == original
    assert original in version_dirs(api_bundle.path.parent)  # the original files were never deleted


# --------------------------------------------------------- failure keeps the old
def _assert_old_still_serves(admin, session_factory, api_bundle):
    root = api_bundle.path.parent
    assert _active_version(session_factory) == api_bundle.version
    assert admin.app.state.polysearch.bundle is api_bundle
    assert admin.get("/readyz").status_code == 200
    r = admin.post("/api/v1/search", json=SEARCH)
    assert r.status_code == 200 and r.json()["index_version"] == api_bundle.version
    assert version_dirs(root) == [api_bundle.version]  # the new, rejected files were removed
    assert not (root / ".jobs").exists() or not list(
        (root / ".jobs").iterdir()
    )  # scratch space too


def test_failed_validation_leaves_the_active_version_untouched(
    admin, session_factory, api_bundle, monkeypatch
):
    add_document(session_factory, "Zebra Rayada", "es")

    def refuse(*args, **kwargs):
        raise reindex.ValidationFailed("SMOKE_SEARCH_FAILED", "2 smoke searches failed")

    monkeypatch.setattr(reindex, "validate_new_index", refuse)
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "failed" and job["error_code"] == "SMOKE_SEARCH_FAILED"
    _assert_old_still_serves(admin, session_factory, api_bundle)
    assert _status_counts(session_factory) == {"active": 1}


def test_a_corrupted_staged_artifact_is_rejected_before_activation(
    admin, session_factory, api_bundle, monkeypatch
):
    add_document(session_factory, "Zebra Rayada", "es")
    real = reindex.build_index

    def build_then_corrupt(*args, **kwargs):
        path = real(*args, **kwargs)
        data = bytearray((path / "faiss_es.index").read_bytes())
        data[100] ^= 0xFF
        (path / "faiss_es.index").write_bytes(bytes(data))
        return path

    monkeypatch.setattr(reindex, "build_index", build_then_corrupt)
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "failed" and job["error_code"] == "ARTIFACTS_INVALID"
    assert "checksum mismatch" in job["error_message"]
    _assert_old_still_serves(admin, session_factory, api_bundle)


def test_the_database_must_match_the_new_index(admin, session_factory, api_bundle, monkeypatch):
    add_document(session_factory, "Zebra Rayada", "es")
    real_export = reindex.export_corpus

    def export_then_drift(session, path):
        result = real_export(session, path)
        add_document(session_factory, "Cebra Tardia", "es")  # database changes after the export
        return result

    monkeypatch.setattr(reindex, "export_corpus", export_then_drift)
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "failed" and job["error_code"] == "CORPUS_COUNT_MISMATCH"
    assert _active_version(session_factory) == api_bundle.version


def test_an_empty_corpus_is_never_indexed(admin, session_factory, api_bundle):
    with session_factory() as s:
        s.execute(text("update documents set is_active = false"))
        s.commit()
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "failed" and job["error_code"] == "EMPTY_CORPUS"
    assert _active_version(session_factory) == api_bundle.version


def test_no_artifact_directory_configured_fails_cleanly(session_factory, api_bundle):
    from api_support import seed_documents
    from app.main import create_app
    from fastapi.testclient import TestClient

    seed_documents(session_factory)
    settings = Settings(
        _env_file=None, admin_api_key=KEY, result_pool_size=20, readiness_cache_seconds=0
    )
    app = create_app(
        settings, bundle=api_bundle, session_factory=session_factory, synchronous_jobs=True
    )
    with TestClient(app) as c:
        job = get_job(c, "index", start_index_job(c))
    assert job["status"] == "failed" and job["error_code"] == "ARTIFACT_ROOT_NOT_CONFIGURED"


# --------------------------------------------------------------- benchmark gate
def _tiny_benchmark(directory, positives):
    from bench_helpers import qrel, query, write_benchmark

    queries = [query("c1-en-title", "c1", "en", "Volcano")]
    qrels = [qrel("c1-en-title", key, key.split(":")[0]) for key in positives]
    write_benchmark(directory, queries, qrels, ["c1"])
    return directory


def test_benchmark_results_are_recorded_in_the_new_manifest(
    make_client, session_factory, api_bundle, tmp_path
):
    bench = _tiny_benchmark(tmp_path / "bench", ["es:Volcán", "fr:Volcan"])
    client = make_client(admin_api_key=KEY, benchmark_dir=bench)
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "completed"
    manifest = client.app.state.polysearch.bundle.manifest
    assert set(manifest["evaluation"]["metrics"]) == {"dense", "lexical", "hybrid"}
    assert manifest["evaluation"]["metrics"]["dense"]["total_queries"] == 1


def test_a_benchmark_that_no_longer_matches_the_corpus_blocks_activation(
    make_client, session_factory, api_bundle, tmp_path
):
    bench = _tiny_benchmark(tmp_path / "bench", ["es:Volcán", "fr:Volcan"])
    client = make_client(admin_api_key=KEY, benchmark_dir=bench)
    add_document(session_factory, "Zebra Rayada", "es")

    with session_factory() as s:  # the benchmark's relevant document disappears
        s.execute(
            Document.__table__.update()
            .where(Document.external_id == "Volcán")
            .values(is_active=False)
        )
        s.commit()
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "failed" and job["error_code"] == "BENCHMARK_INVALID"
    assert _active_version(session_factory) == api_bundle.version


def test_a_quality_regression_blocks_activation(make_client, session_factory, api_bundle, tmp_path):
    bench = _tiny_benchmark(tmp_path / "bench", ["es:Volcán", "fr:Volcan"])
    client = make_client(admin_api_key=KEY, benchmark_dir=bench, reindex_max_mrr_drop=0.0)
    api_bundle.manifest["evaluation"] = {
        "metrics": {"dense": {"mrr": 1.01}}
    }  # impossibly good before
    add_document(session_factory, "Zebra Rayada", "es")
    job = get_job(client, "index", start_index_job(client))
    assert job["status"] == "failed" and job["error_code"] == "BENCHMARK_REGRESSION"
    assert _active_version(session_factory) == api_bundle.version


# ----------------------------------------------------------- conflicts and bounds
def test_a_second_job_is_rejected_while_one_is_active(admin, session_factory):
    with session_factory() as s:
        s.add(IndexJob(status="running"))
        s.commit()
    r = admin.post("/api/v1/admin/index-jobs", json={}, headers=HEAD)
    assert r.status_code == 409 and r.json()["error"]["code"] == "JOB_CONFLICT"
    r = admin.post(
        "/api/v1/admin/ingestion-jobs", json={"language": "en", "mode": "incremental"}, headers=HEAD
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "JOB_CONFLICT"  # shared worker


def test_an_active_ingestion_job_blocks_a_reindex(admin, session_factory):
    with session_factory() as s:
        source_id = s.scalar(
            select(__import__("app.db.models", fromlist=["ContentSource"]).ContentSource.id)
        )
        s.add(IngestionJob(source_id=source_id, mode="seed", status="queued"))
        s.commit()
    assert admin.post("/api/v1/admin/index-jobs", json={}, headers=HEAD).status_code == 409


def test_the_database_itself_allows_only_one_active_job(session_factory):
    with session_factory() as s:
        s.add(IndexJob(status="queued"))
        s.commit()
        s.add(IndexJob(status="running"))
        with pytest.raises(IntegrityError):
            s.commit()
        s.rollback()
        s.add(IndexJob(status="failed"))  # finished jobs are unlimited
        s.commit()


# ------------------------------------------------------------------ stale jobs
def test_stale_jobs_are_failed_instead_of_running_forever(session_factory):
    with session_factory() as s:
        s.add(IndexJob(status="running"))
        s.commit()
    assert recover_stale_jobs(session_factory) == 1
    with session_factory() as s:
        job = s.scalar(select(IndexJob))
        assert (job.status, job.error_code) == ("failed", RESTART_CODE) and job.finished_at
    assert recover_stale_jobs(session_factory) == 0  # nothing left to recover


def test_startup_recovers_stale_jobs_so_new_work_is_possible(session_factory):
    from api_support import seed_documents, test_settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    seed_documents(session_factory)
    with session_factory() as s:
        s.add(IndexJob(status="running"))
        s.add(
            IngestionJob(source_id=s.scalar(select(ContentSource.id)), mode="seed", status="queued")
        )
        s.commit()
    app = create_app(
        test_settings(admin_api_key=KEY), session_factory=session_factory, synchronous_jobs=True
    )
    with TestClient(app) as c:  # the lifespan runs recovery before accepting work
        with session_factory() as s:
            statuses = {j.status for j in s.scalars(select(IndexJob))}
            statuses |= {j.status for j in s.scalars(select(IngestionJob))}
        assert statuses == {"failed"}
        assert c.post("/api/v1/admin/index-jobs", json={}, headers=HEAD).status_code == 202


# ------------------------------------------------------------------ smoke checks
def test_smoke_searches_use_the_exact_indexed_text_not_just_the_title(
    admin, session_factory, monkeypatch
):
    from app.ingestion.text import dense_text

    add_document(session_factory, "Zebra Rayada", "es")
    seen = []
    real = SearchBundle.search

    def spy(self, query, mode, targets, k=10, pool=50):
        seen.append(query)
        return real(self, query, mode, targets, k=k, pool=pool)

    monkeypatch.setattr(SearchBundle, "search", spy)
    assert get_job(admin, "index", start_index_job(admin))["status"] == "completed"
    with session_factory() as s:
        texts = {dense_text(d.title, d.summary) for d in s.scalars(select(Document))}
    smoke_queries = [q for q in seen if q in texts]
    assert len(smoke_queries) >= 10  # 10 documents x 2 modes use indexed text
    assert not any(q in {"Volcán", "Glaciar", "Jazz"} for q in seen)  # never title-only


def test_a_wrong_id_to_document_mapping_is_caught_even_when_every_file_hash_is_valid(
    admin, session_factory, api_bundle, monkeypatch
):
    import hashlib
    import json

    add_document(session_factory, "Zebra Rayada", "es")
    real = reindex.build_index

    def build_with_swapped_mapping(*args, **kwargs):
        path = real(*args, **kwargs)
        mapping = json.loads((path / "mapping_es.json").read_text(encoding="utf-8"))
        mapping[0], mapping[1] = mapping[1], mapping[0]  # still valid keys, same count
        (path / "mapping_es.json").write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["files"]["mapping_es.json"] = hashlib.sha256(
            (path / "mapping_es.json").read_bytes()
        ).hexdigest()
        (path / "manifest.json").write_text(json.dumps(manifest))
        return path

    monkeypatch.setattr(reindex, "build_index", build_with_swapped_mapping)
    job = get_job(admin, "index", start_index_job(admin))
    assert job["status"] == "failed" and job["error_code"] == "SMOKE_SEARCH_FAILED"
    assert _active_version(session_factory) == api_bundle.version
    assert version_dirs(api_bundle.path.parent) == [api_bundle.version]
