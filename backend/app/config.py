"""Application settings, loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
