from alembic import command
from conftest import reset_schema
from sqlalchemy import inspect, text


def test_migration_succeeds_on_empty_database_and_is_reversible(engine, alembic_config):
    reset_schema(engine)
    command.upgrade(alembic_config, "head")
    tables = set(inspect(engine).get_table_names())
    assert {"content_sources", "documents", "ingestion_jobs"} <= tables
    command.downgrade(alembic_config, "base")
    assert "documents" not in set(inspect(engine).get_table_names())
    command.upgrade(alembic_config, "head")  # leave schema usable for other tests
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from documents")).scalar() == 0
