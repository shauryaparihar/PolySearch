"""Structured JSON logging with an allow-list of fields.

``log_event`` refuses any field that is not on the allow-list, so raw queries, document text,
admin keys, database URLs and headers cannot be logged by accident.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "polysearch"
ALLOWED_FIELDS = {
    "request_id", "search_id", "job_id", "route", "method", "status", "duration_ms",
    "encode_ms", "lexical_ms", "faiss_ms", "fusion_ms", "db_ms", "total_ms",
    "cache_hit", "result_count", "languages", "sources", "mode", "index_version",
    "error_code", "query_sha256", "dropped_documents", "limit", "retry_after",
}  # fmt: skip


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_polysearch", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._polysearch = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    unknown = set(fields) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"log fields not allowed: {sorted(unknown)}")
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"fields": fields})
