from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from app.config import Settings
from app.db.base import make_engine, make_session_factory
from app.ingestion.wikipedia import WikipediaSource
from fake_mediawiki import FakeMediaWiki
from sqlalchemy import text
from sqlalchemy.engine import Engine

TEST_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://polysearch:polysearch_dev@localhost:5433/polysearch_test",
)
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Engine for the dedicated test database. Refuses any non-test database name."""
    eng = make_engine(TEST_URL)
    assert eng.url.database and eng.url.database.endswith("_test"), "tests must use a *_test DB"
    try:
        with eng.connect() as conn:
            conn.execute(text("select 1"))
    except Exception:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip("PostgreSQL test database unavailable (set REQUIRE_DB=1 to fail instead)")
    return eng


def _alembic_config() -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", TEST_URL.replace("%", "%%"))
    return cfg


def reset_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))


@pytest.fixture(scope="session")
def migrated_engine(engine: Engine) -> Engine:
    reset_schema(engine)
    command.upgrade(_alembic_config(), "head")
    return engine


@pytest.fixture
def alembic_config() -> Config:
    return _alembic_config()


@pytest.fixture
def session_factory(migrated_engine: Engine):
    with migrated_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE click_events, relevance_feedback, search_results, search_runs, "
                "index_jobs, index_versions, documents, ingestion_jobs, content_sources "
                "RESTART IDENTITY CASCADE"
            )
        )
    return make_session_factory(migrated_engine)


@pytest.fixture
def wiki_pages() -> dict:
    return {
        "en": {
            "Mathematics": {"revid": 11, "extract": "Mathematics is the study of numbers."},
            "Physics": {"revid": 12, "extract": "Physics is the study of matter."},
            "Zebra": {"revid": 13, "extract": "A zebra is an African equine."},
        },
        "es": {"Medicina": {"revid": 21, "extract": "La medicina es una ciencia."}},
        "fr": {},
    }


@pytest.fixture
def fake_wiki(wiki_pages) -> FakeMediaWiki:
    return FakeMediaWiki(wiki_pages)


@pytest.fixture
def source(fake_wiki) -> WikipediaSource:
    return WikipediaSource(
        Settings(http_max_retries=2),
        transport=fake_wiki.transport(),
        sleep=lambda _s: None,
        jitter=lambda: 0.0,
    )


@pytest.fixture
def encoder():
    from fake_encoder import FakeEncoder

    return FakeEncoder()


@pytest.fixture
def api_bundle(tmp_path, encoder):
    from api_support import build_bundle

    return build_bundle(tmp_path, encoder)


@pytest.fixture
def make_client(session_factory, api_bundle):
    """Factory: ``make_client(**settings)`` -> TestClient on an app wired to bundle + test DB."""
    from api_support import seed_documents, test_settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    seed_documents(session_factory)
    opened = []

    def factory(bundle=api_bundle, synchronous_jobs=True, source_factory=None, **settings):
        settings.setdefault("artifact_root", api_bundle.path.parent)
        app = create_app(
            test_settings(**settings),
            bundle=bundle,
            session_factory=session_factory,
            synchronous_jobs=synchronous_jobs,
            source_factory=source_factory,
        )
        client = TestClient(app)
        client.__enter__()  # runs lifespan: registers the active index version
        opened.append(client)
        return client

    yield factory
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
