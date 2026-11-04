"""Application settings, loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # ---- optional citation-backed answers (RAG). Disabled by default; search never needs these.
    rag_enabled: bool = False
    llm_base_url: str | None = None  # OpenAI-compatible endpoint root, e.g. https://host/v1
    llm_api_key: SecretStr | None = None  # never printed: SecretStr hides it in repr and logs
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    llm_max_output_tokens: int = Field(default=600, ge=50, le=4000)
    # Ask the provider for a JSON-only reply (response_format=json_object). Turn off for an
    # OpenAI-compatible server that rejects it; the prompt still demands JSON and every reply is
    # validated either way.
    llm_json_mode: bool = True
    # Prices (per million tokens) are used only for cost *estimates*; unset = no estimate.
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0)
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0)
    answer_max_question_chars: int = Field(default=300, ge=20, le=500)
    rag_candidate_documents: int = Field(default=10, ge=1, le=50)
    rag_final_passages: int = Field(default=6, ge=1, le=12)
    rag_max_passages_per_document: int = Field(default=2, ge=1, le=5)
    rag_max_passage_tokens: int = Field(default=200, ge=20, le=500)
    rag_max_context_tokens: int = Field(default=1500, ge=100, le=6000)
    # Below this cosine similarity between the question and the best passage the system abstains
    # without calling the model. It is a cost-saving pre-filter, not a safety gate: on a
    # hand-written 160-question set the score alone separated answerable from unanswerable only
    # moderately (AUC about 0.85), so the model's own abstention and the citation validator
    # remain the real safeguards. Tune with backend/scripts/calibrate_evidence_threshold.py.
    rag_min_evidence_score: float = Field(default=0.55, ge=-1.0, le=1.0)
    rag_max_repair_attempts: int = Field(default=1, ge=0, le=1)  # at most one repair call
    answer_rate_limit_per_minute: int = Field(default=5, ge=1, le=1000)
    generation_max_concurrency: int = Field(default=1, ge=1, le=4)
    rag_daily_token_budget: int = Field(default=100_000, ge=0, le=100_000_000)  # 0 = no calls
    rag_cache_ttl_seconds: int = Field(default=600, ge=1, le=86_400)
    # Local debugging only: store the raw question text with each run. Never enable in tests.
    rag_store_question_text: bool = False

    @field_validator(
        "active_index_path", "artifact_root", "benchmark_dir", "cursor_secret", "admin_api_key",
        "llm_base_url", "llm_api_key", "llm_model", "llm_input_price_per_mtok",
        "llm_output_price_per_mtok",
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

    @field_validator("llm_base_url")
    @classmethod
    def _llm_url_is_http(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must start with http:// or https://")
        return value.rstrip("/") if value else value

    @property
    def rag_configured(self) -> bool:
        """True only when answers are switched on AND a provider is fully configured."""
        return bool(
            self.rag_enabled and self.llm_base_url and self.llm_api_key and self.llm_model
        )

    @model_validator(mode="after")
    def _validate_deployment(self) -> Settings:
        """Fail at startup, with messages that name the setting but never print its value."""
        problems: list[str] = []
        secrets = {
            "CURSOR_SECRET": self.cursor_secret,
            "ADMIN_API_KEY": self.admin_api_key,
            "DATABASE_URL": self.database_url,
            "LLM_API_KEY": self.llm_api_key.get_secret_value() if self.llm_api_key else None,
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
            if self.llm_base_url and not self.llm_base_url.startswith("https://"):
                problems.append("LLM_BASE_URL must use https:// in production")
            if not (self.active_index_path or self.artifact_root):
                problems.append("ACTIVE_INDEX_PATH or ARTIFACT_ROOT must point at index artifacts")
        if problems:
            raise ValueError("invalid configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
