"""Helpers shared by the answer-mode tests (the fixtures live in conftest.py)."""

from __future__ import annotations

import json

from app.db.models import GenerationRun
from sqlalchemy import func, select

PROVIDER = dict(
    rag_enabled=True,
    llm_base_url="https://provider.invalid/v1",
    llm_api_key="sk-test-secret-value",
    llm_model="m",
    rag_min_evidence_score=-1.0,  # fake-encoder scores mean nothing; thresholds are tested apart
)
QUESTION = "What is a volcano?"
ASK = {"question": QUESTION, "languages": ["en", "es", "fr"]}


def answer_json(*claims: tuple[str, list[str]], answer: str = "A cited answer.") -> str:
    return json.dumps(
        {
            "answer": answer,
            "claims": [{"text": t, "citation_ids": ids} for t, ids in claims],
            "abstained": False,
            "abstention_reason": None,
        }
    )


def flatten(records) -> str:
    return " ".join(
        r.getMessage() + json.dumps(getattr(r, "fields", {}), default=str) for r in records
    )


def ask(client, **body):
    return client.post("/api/v1/answers", json={**ASK, **body})


def runs(factory) -> list[GenerationRun]:
    with factory() as session:
        return list(session.scalars(select(GenerationRun).order_by(GenerationRun.created_at)))


def count(factory, model) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model))
