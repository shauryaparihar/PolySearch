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
            text("TRUNCATE documents, ingestion_jobs, content_sources RESTART IDENTITY CASCADE")
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
