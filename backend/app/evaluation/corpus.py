"""The canonical searchable corpus: a deterministic JSONL export of active documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document


def doc_key(language: str, external_id: str) -> str:
    """Stable, database-independent document identity, e.g. ``es:Torre_Eiffel``."""
    return f"{language}:{external_id}"


@dataclass(frozen=True)
class CorpusDoc:
    key: str
    language: str
    title: str
    summary: str


@dataclass(frozen=True)
class Corpus:
    docs: list[CorpusDoc]
    sha256: str

    @property
    def keys(self) -> set[str]:
        return {d.key for d in self.docs}

    @property
    def language_of(self) -> dict[str, str]:
        return {d.key: d.language for d in self.docs}

    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.docs:
            counts[d.language] = counts.get(d.language, 0) + 1
        return dict(sorted(counts.items()))


def _line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def export_corpus(session: Session, path: Path) -> dict:
    """Write active documents to ``path`` sorted by key; return a manifest for the export."""
    rows = []
    for d in session.scalars(select(Document).where(Document.is_active.is_(True))):
        rows.append(
            {
                "key": doc_key(d.language, d.external_id),
                "language": d.language,
                "title": d.title,
                "summary": d.summary,
                "source_url": d.source_url,
                "source_revision": d.source_revision,
                "content_sha256": d.content_sha256,
            }
        )
    rows.sort(key=lambda r: r["key"])
    text = "".join(_line(r) + "\n" for r in rows)  # an empty corpus is an empty file
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["language"]] = counts.get(r["language"], 0) + 1
    return {
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "documents": len(rows),
        "per_language": dict(sorted(counts.items())),
        "with_source_revision": sum(1 for r in rows if r["source_revision"]),
        "without_source_revision": sum(1 for r in rows if not r["source_revision"]),
    }


def load_corpus(path: Path) -> Corpus:
    data = Path(path).read_bytes()
    docs = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(CorpusDoc(row["key"], row["language"], row["title"], row["summary"]))
    keys = [d.key for d in docs]
    if len(keys) != len(set(keys)):
        raise ValueError("corpus contains duplicate document keys")
    return Corpus(docs, hashlib.sha256(data).hexdigest())
