"""Shared builders for API tests: a real bundle (fake encoder) plus matching database rows."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.db.models import ContentSource
from app.evaluation.corpus import Corpus, CorpusDoc
from app.ingestion import service
from app.ingestion.sources import NormalizedRecord
from app.ingestion.text import content_sha256
from app.search.artifacts import build_index
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder
from sqlalchemy.orm import sessionmaker

CONCEPTS = {
    "en": ["Volcano", "Glacier", "Jazz", "Cancer", "Opera", "Coral", "Tsunami", "Vitamin"],
    "es": ["Volcán", "Glaciar", "Jazz", "Cáncer", "Ópera", "Coral", "Tsunami", "Vitamina"],
    "fr": ["Volcan", "Glacier", "Jazz", "Cancer", "Opéra", "Corail", "Tsunami", "Vitamine"],
}


def summary_of(title: str, language: str) -> str:
    return f"{title} es un tema de la enciclopedia ({language}). Texto de ejemplo para {title}."


def docs() -> list[CorpusDoc]:
    out = []
    for language, titles in CONCEPTS.items():
        for title in titles:
            out.append(
                CorpusDoc(f"{language}:{title.replace(' ', '_')}", language, title,
                          summary_of(title, language))
            )  # fmt: skip
    out.sort(key=lambda d: d.key)  # same order as the corpus export, so versions compare equal
    return out


def build_bundle(tmp_path: Path, encoder: FakeEncoder | None = None) -> SearchBundle:
    encoder = encoder or FakeEncoder()
    path = build_index(Corpus(docs(), "d" * 64), encoder, tmp_path / "index")
    return SearchBundle.load(path, encoder=encoder)


def seed_documents(factory: sessionmaker) -> int:
    """Insert the content source and one active document per corpus doc; returns the source id."""
    with factory() as session:
        source = ContentSource(
            name="wikipedia",
            base_url="https://www.wikipedia.org/",
            licence_name="Creative Commons Attribution-Share Alike 4.0",
            licence_url="https://creativecommons.org/licenses/by-sa/4.0/deed.en",
            attribution_text="Source: Wikipedia. Text is available under CC BY-SA 4.0.",
        )
        session.add(source)
        session.flush()
        for d in docs():
            service.upsert_document(
                session,
                source.id,
                NormalizedRecord(
                    external_id=d.key.split(":", 1)[1],
                    language=d.language,
                    title=d.title,
                    summary=d.summary,
                    source_url=f"https://{d.language}.wikipedia.org/wiki/{d.key.split(':', 1)[1]}",
                    source_revision="42",
                    source_updated_at=None,
                    content_sha256=content_sha256(d.title, d.summary),
                ),
            )
        session.commit()
        return source.id


def test_settings(**overrides) -> Settings:
    base = dict(
        cursor_secret="test-secret",
        result_pool_size=20,
        search_rate_limit_per_minute=1000,
        readiness_cache_seconds=0,
        cors_allowed_origins=["http://localhost:5173"],
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


SEARCH = {"query": "Volcano", "query_language": "en", "target_languages": ["es", "fr"]}
