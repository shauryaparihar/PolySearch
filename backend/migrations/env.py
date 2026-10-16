import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

from alembic import context  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import models  # noqa: E402,F401  (registers tables on Base.metadata)
from app.db.base import Base  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Tests set sqlalchemy.url explicitly; otherwise use application settings.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
