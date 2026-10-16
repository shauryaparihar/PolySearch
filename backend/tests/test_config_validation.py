"""Configuration is validated at startup, and errors never echo secret values."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import Settings
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
GOOD_SECRET = "s" * 40
PRODUCTION = dict(
    environment="production",
    cursor_secret=GOOD_SECRET,
    admin_api_key="a" * 40,
    database_url="postgresql+psycopg://app:Zq9-long-random@db.internal:5432/polysearch",
    cors_allowed_origins=["https://polysearch.example"],
    artifact_root="/app/artifacts/index",
)


def build(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_a_complete_production_configuration_is_accepted():
    assert build(**PRODUCTION).environment == "production"


def test_development_defaults_still_work_without_any_secrets():
    s = build()
    assert s.environment == "development" and s.cursor_secret is None


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"cursor_secret": None}, "CURSOR_SECRET"),
        ({"cursor_secret": "short"}, "CURSOR_SECRET"),
        ({"admin_api_key": "short"}, "ADMIN_API_KEY"),
        ({"database_url": "postgresql+psycopg://polysearch:polysearch_dev@db/x"}, "DATABASE_URL"),
        ({"cors_allowed_origins": ["http://localhost:5173"]}, "CORS_ALLOWED_ORIGINS"),
        ({"artifact_root": None}, "ARTIFACT_ROOT"),
    ],
)
def test_production_rejects_weak_or_missing_settings(override, fragment):
    with pytest.raises(ValidationError, match=fragment):
        build(**{**PRODUCTION, **override})


def test_an_unset_admin_key_keeps_the_admin_api_off_in_production():
    values = {k: v for k, v in PRODUCTION.items() if k != "admin_api_key"}
    assert build(**values).admin_api_key is None


@pytest.mark.parametrize("name", ["cursor_secret", "admin_api_key", "database_url"])
def test_placeholders_are_refused_in_every_environment(name):
    with pytest.raises(ValidationError, match="placeholder"):
        build(**{name: "CHANGE_ME_TO_A_LONG_RANDOM_VALUE"})


def test_error_messages_never_contain_secret_values():
    secret = "super-secret-but-short"
    with pytest.raises(ValidationError) as caught:
        build(**{**PRODUCTION, "cursor_secret": secret, "database_url": "postgresql://u:pw9@h/d"})
    text = str(caught.value)
    assert secret not in text and "pw9" not in text and "input_value" not in text


def test_wildcard_origins_are_still_refused():
    with pytest.raises(ValidationError):
        build(cors_allowed_origins=["*"])


def test_the_example_env_file_documents_real_settings_and_cannot_be_used_unedited():
    """Every key in .env.example is a real setting; copying it unchanged fails fast."""
    pairs = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key.strip().lower()] = value
    not_settings = {"test_database_url"}  # read by the test suite
    not_settings |= {"postgres_password", "postgres_host_port", "api_host_port",
                     "frontend_host_port"}  # read by docker-compose.yml  # fmt: skip
    unknown = set(pairs) - set(Settings.model_fields) - not_settings
    assert not unknown, f".env.example lists settings that do not exist: {unknown}"
    assert any("CHANGE_ME" in v for v in pairs.values())
    with pytest.raises(ValidationError, match="placeholder"):
        build(**{k: v for k, v in pairs.items() if k in ("cursor_secret", "admin_api_key")})


def test_empty_environment_values_mean_not_set(monkeypatch):
    for name in ("ACTIVE_INDEX_PATH", "ARTIFACT_ROOT", "BENCHMARK_DIR", "CURSOR_SECRET",
                 "ADMIN_API_KEY"):  # fmt: skip
        monkeypatch.setenv(name, "")
    s = Settings(_env_file=None)
    assert s.active_index_path is None and s.artifact_root is None and s.benchmark_dir is None
    assert s.cursor_secret is None and s.admin_api_key is None  # admin API stays switched off
