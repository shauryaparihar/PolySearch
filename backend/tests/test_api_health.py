import shutil

from app.db.models import Document, IndexVersion
from app.main import create_app
from sqlalchemy import delete, update


def _ready(client):
    r = client.get("/readyz")
    return r.status_code, r.json()


def test_healthz_is_a_cheap_liveness_probe(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ready_when_everything_matches(client):
    status, body = _ready(client)
    assert status == 200 and body["ready"] is True
    assert set(body["checks"]) == {
        "database", "active_index", "artifacts", "encoder", "document_mappings",
    }  # fmt: skip
    assert all(body["checks"].values())


def test_not_ready_without_a_database(api_bundle):
    from api_support import test_settings
    from app.db.base import make_engine, make_session_factory
    from fastapi.testclient import TestClient

    dead = make_session_factory(make_engine("postgresql+psycopg://x:y@localhost:5999/none"))
    with TestClient(create_app(test_settings(), bundle=api_bundle, session_factory=dead)) as c:
        status, body = _ready(c)
        assert status == 503 and body["checks"]["database"] is False
        assert c.get("/healthz").status_code == 200  # still alive, just not ready


def test_not_ready_when_an_artifact_changes_after_startup(client, api_bundle):
    victim = api_bundle.path / "mapping_es.json"
    victim.write_text(victim.read_text() + " ")
    status, body = _ready(client)
    assert status == 503 and body["checks"]["artifacts"] is False


def test_not_ready_when_a_file_goes_missing(client, api_bundle):
    shutil.move(api_bundle.path / "faiss_fr.index", api_bundle.path / "gone")
    assert _ready(client)[1]["checks"]["artifacts"] is False


def test_not_ready_without_an_active_index_row(client, session_factory):
    with session_factory() as s:
        s.execute(update(IndexVersion).values(status="retired"))
        s.commit()
    assert _ready(client)[1]["checks"]["active_index"] is False


def test_not_ready_when_indexed_documents_no_longer_resolve(client, session_factory):
    with session_factory() as s:
        s.execute(update(Document).where(Document.external_id == "Jazz").values(is_active=False))
        s.commit()
    status, body = _ready(client)
    assert status == 503 and body["checks"]["document_mappings"] is False


def test_not_ready_without_an_encoder(make_client, api_bundle):
    from app.search.bundle import SearchBundle

    lexical_only = SearchBundle.load(api_bundle.path, load_encoder=False)
    client = make_client(bundle=lexical_only)
    assert _ready(client)[1]["checks"]["encoder"] is False


def test_no_index_means_search_is_unavailable_but_the_process_is_alive(session_factory):
    from api_support import test_settings
    from fastapi.testclient import TestClient

    with TestClient(create_app(test_settings(), session_factory=session_factory)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        r = c.post(
            "/api/v1/search",
            json={"query": "volcano", "query_language": "en", "target_languages": ["es"]},
        )
        assert r.status_code == 503 and r.json()["error"]["code"] == "INDEX_UNAVAILABLE"


def test_search_survives_missing_database_rows_by_dropping_them(client, session_factory):
    with session_factory() as s:
        s.execute(update(Document).where(Document.external_id == "Volcán").values(is_active=False))
        s.commit()
    body = client.post(
        "/api/v1/search",
        json={"query": "Volcano", "query_language": "en", "target_languages": ["es", "fr"]},
    ).json()
    assert all(not r["source_url"].endswith("/Volcán") for r in body["results"])
    assert body["total_results"] == 15  # indexed but no longer active: dropped, not a crash


def test_readiness_result_is_cached_for_the_configured_time(make_client, session_factory):
    client = make_client(readiness_cache_seconds=60)
    assert _ready(client)[0] == 200
    with session_factory() as s:
        s.execute(delete(IndexVersion))
        s.commit()
    assert _ready(client)[0] == 200  # still the cached answer
