"""Application settings, loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_LANGUAGES = ("en", "es", "fr")
PLACEHOLDER = "CHANGE_ME"  # the marker used by .env.example
MIN_SECRET_CHARS = 32
LOCAL_DEV_DB_PASSWORD = "polysearch_dev"  # the throwaway password in docker-compose.yml


class Settings(BaseSettings):
    # hide_input_in_errors: a validation error must never echo secret values into logs.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    # "production" turns on the strict checks in ``_validate_deployment`` below.
    environment: Literal["development", "production"] = "development"


    # Throwaway local-development credentials matching docker-compose.yml.
    database_url: str = (
        "postgresql+psycopg://polysearch:polysearch_dev@localhost:5433/polysearch_dev"
    )
    wikipedia_user_agent: str = (
        "PolySearch/0.1 (portfolio project; contact shauryaparihar196@gmail.com)"
    )
    # Network behaviour for the MediaWiki connector.
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 20.0
    http_max_retries: int = Field(default=3, ge=0, le=5)
    http_backoff_base: float = Field(default=0.5, gt=0)
    http_backoff_max: float = Field(default=8.0, gt=0)
    # Bounds so a job can never run away.
    max_pages_per_job: int = Field(default=200, ge=1, le=1000)
    titles_per_request: int = Field(default=20, ge=1, le=20)
    summary_max_chars: int = Field(default=2000, ge=200, le=5000)

    # ---- search API (Phase 4)
    active_index_path: Path | None = None  # artifact directory served by the API
    cors_allowed_origins: list[str] = ["http://localhost:5173"]  # exact origins, never "*"
    max_request_body_bytes: int = Field(default=8192, ge=256, le=1_000_000)
    search_rate_limit_per_minute: int = Field(default=30, ge=1, le=10_000)
    trusted_proxy_ips: list[str] = []  # forwarding headers are honoured only from these peers
    cache_max_entries: int = Field(default=512, ge=1, le=100_000)
    cache_ttl_seconds: int = Field(default=600, ge=1, le=86_400)
    inference_max_concurrency: int = Field(default=2, ge=1, le=16)
    inference_wait_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    result_pool_size: int = Field(default=100, ge=10, le=200)
    snippet_max_chars: int = Field(default=300, ge=50, le=1000)
    cursor_secret: str | None = None  # set in deployments so cursors survive restarts
    search_retention_days: int = Field(default=30, ge=1, le=3650)
    readiness_cache_seconds: float = Field(default=30.0, ge=0, le=600)
    # Raw query text is stored for analytics only when this is explicitly enabled.
    analytics_store_query_text: bool = False

    # ---- product signals (Phase 5)
    admin_api_key: str | None = None  # unset = admin endpoints do not exist (404)
    suggest_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    write_rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)  # clicks, feedback
    admin_failed_attempts_per_minute: int = Field(default=10, ge=1, le=1000)  # wrong keys only
    admin_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)  # authenticated calls
    popular_query_min_count: int = Field(default=5, ge=2, le=1000)
    max_suggestions: int = Field(default=10, ge=1, le=20)

    # ---- background sync and re-indexing (Phase 6)
    encoder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    encoder_revision: str = (
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"  # used only if nothing is loaded
    )
    artifact_root: Path | None = (
        None  # where index versions live; default: parent of the active index
    )
    benchmark_dir: Path | None = None  # benchmark gate for new indexes; None = gate skipped
    reindex_max_mrr_drop: float = Field(default=0.05, ge=0, le=1)
    reindex_smoke_documents: int = Field(default=10, ge=1, le=100)
    job_shutdown_timeout_seconds: float = Field(default=30.0, ge=0, le=600)
    index_batch_size: int = Field(default=64, ge=1, le=512)

    @field_validator(
        "active_index_path", "artifact_root", "benchmark_dir", "cursor_secret", "admin_api_key",
        mode="before",
    )
    @classmethod
    def _empty_means_unset(cls, value):
        """Compose/Docker pass unset variables as empty strings; treat those as "not set"."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("cors_allowed_origins")
    @classmethod
    def _no_wildcard_origins(cls, origins: list[str]) -> list[str]:
        if any(
            o.strip() in ("*", "") or not o.startswith(("http://", "https://")) for o in origins
        ):
            raise ValueError("CORS origins must be exact http(s) origins, never '*'")
        return origins

    @model_validator(mode="after")
    def _validate_deployment(self) -> Settings:
        """Fail at startup, with messages that name the setting but never print its value."""
        problems: list[str] = []
        secrets = {
            "CURSOR_SECRET": self.cursor_secret,
            "ADMIN_API_KEY": self.admin_api_key,
            "DATABASE_URL": self.database_url,
        }
        for name, value in secrets.items():  # an untouched .env.example must not start up
            if value and PLACEHOLDER in value:
                problems.append(f"{name} still contains the {PLACEHOLDER} placeholder")
        if self.environment == "production":
            for name, value in (("CURSOR_SECRET", self.cursor_secret),
                                ("ADMIN_API_KEY", self.admin_api_key)):  # fmt: skip
                if name == "ADMIN_API_KEY" and value is None:
                    continue  # unset keeps the admin API switched off, which is safe
                if not value or len(value) < MIN_SECRET_CHARS:
                    problems.append(f"{name} must be set to at least {MIN_SECRET_CHARS} characters")
            if LOCAL_DEV_DB_PASSWORD in self.database_url:
                problems.append("DATABASE_URL uses the local-development password")
            if any("localhost" in o or "127.0.0.1" in o for o in self.cors_allowed_origins):
                problems.append("CORS_ALLOWED_ORIGINS must name the deployed frontend")
            if not (self.active_index_path or self.artifact_root):
                problems.append("ACTIVE_INDEX_PATH or ARTIFACT_ROOT must point at index artifacts")
        if problems:
            raise ValueError("invalid configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
