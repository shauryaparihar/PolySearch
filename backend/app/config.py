"""Application settings, loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_LANGUAGES = ("en", "es", "fr")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    @field_validator("cors_allowed_origins")
    @classmethod
    def _no_wildcard_origins(cls, origins: list[str]) -> list[str]:
        if any(
            o.strip() in ("*", "") or not o.startswith(("http://", "https://")) for o in origins
        ):
            raise ValueError("CORS origins must be exact http(s) origins, never '*'")
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
