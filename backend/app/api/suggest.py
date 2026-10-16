"""Build the autocomplete index for the active search bundle."""

from __future__ import annotations

import threading

from sqlalchemy import func, select

from app.api.state import AppState
from app.config import SUPPORTED_LANGUAGES
from app.db.models import ContentSource, Document, SearchRun
from app.search.trie import PrefixTrie, Suggester, looks_sensitive

_build_lock = threading.Lock()


def popular_queries(session, min_count: int) -> list[tuple[str, int]]:
    """Normalised query text searched at least ``min_count`` times, minus anything that looks
    sensitive. Empty unless query text was explicitly stored (analytics_store_query_text)."""
    rows = session.execute(
        select(SearchRun.analytics_query_text, func.count().label("n"))
        .where(
            SearchRun.analytics_query_text.is_not(None),
            SearchRun.status == "ok",
            SearchRun.result_pool_size > 0,
        )
        .group_by(SearchRun.analytics_query_text)
        .having(func.count() >= min_count)
        .order_by(func.count().desc())
        .limit(500)
    ).all()
    return [(text, n) for text, n in rows if text and not looks_sensitive(text)]


def build_suggester(state: AppState) -> Suggester:
    bundle = state.bundle
    indexed = bundle.document_keys()
    tries = {lang: PrefixTrie() for lang in SUPPORTED_LANGUAGES}
    combined = PrefixTrie()
    with state.session_factory() as session:
        rows = session.execute(
            select(Document.language, Document.external_id, Document.title)
            .join(ContentSource, ContentSource.id == Document.source_id)
            .where(Document.is_active.is_(True), ContentSource.enabled.is_(True))
        )
        for language, external_id, title in rows:
            if f"{language}:{external_id}" in indexed:  # only titles that can be searched
                tries[language].add(title, languages={language})
                combined.add(title, languages={language})
        if state.settings.analytics_store_query_text:
            for text, count in popular_queries(session, state.settings.popular_query_min_count):
                combined.add(text, weight=count)
    return Suggester(
        index_version=bundle.version,
        all=combined.build(),
        by_language={lang: trie.build() for lang, trie in tries.items()},
    )


def get_suggester(state: AppState) -> Suggester | None:
    """The suggester for the *current* bundle, rebuilt if the bundle changed."""
    if state.bundle is None:
        return None
    current = state.suggester
    if current is not None and current.index_version == state.bundle.version:
        return current
    with _build_lock:
        current = state.suggester
        if current is None or current.index_version != state.bundle.version:
            state.suggester = build_suggester(state)
        return state.suggester
