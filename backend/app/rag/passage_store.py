"""Versioned passage artifacts: build, validate, load, and synchronise with PostgreSQL.

Layout of one passage artifact (``<ARTIFACT_ROOT>/passages/<index_version>/``):

    manifest.json    schema/chunker/encoder identity, parent index, counts, SHA-256 of both files
    passages.jsonl   one JSON object per passage, in a fixed order (document key, ordinal)
    vectors.npy      float32 unit vectors, row i belongs to line i of passages.jsonl

Like the search index it is pickle-free, checksummed and moved into place atomically. It is
tied to exactly one search index version and is optional: if it is missing or invalid the
search engine is unaffected and only answer mode reports itself unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.config import SUPPORTED_LANGUAGES
from app.db.models import Document, DocumentPassage, IndexVersion
from app.evaluation.corpus import Corpus, doc_key
from app.ingestion.text import dense_text
from app.rag.passages import (
    CHUNKER_VERSION,
    DEFAULT_CONFIG,
    ChunkConfig,
    Passage,
    build_passages,
    content_sha256,
)
from app.rag.types import PASSAGE_SCHEMA_VERSION
from app.search.encoder import Encoder, normalise

PASSAGES_DIR = "passages"
MANIFEST = "manifest.json"
DATA_FILES = ("passages.jsonl", "vectors.npy")
BATCH_SIZE = 64


class PassageArtifactError(Exception):
    """A passage artifact is missing, corrupt, inconsistent or built for another index."""


def passage_dir(artifact_root: Path, index_version: str) -> Path:
    return Path(artifact_root) / PASSAGES_DIR / index_version


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _line(passage: Passage) -> str:
    return json.dumps(asdict(passage), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------- build
def _embedding_key(title: str, text: str) -> str:
    return hashlib.sha256(dense_text(title, text).encode("utf-8")).hexdigest()


def _reusable_vectors(previous: PassageStore | None, encoder: Encoder) -> dict[str, np.ndarray]:
    """Vectors of an earlier artifact keyed by the exact text that was embedded.

    Only usable when the same encoder (name, revision, dimension) produced them; the key is a
    hash of the embedded text itself, so a changed document, title or chunking never matches.
    """
    if previous is None:
        return {}
    meta = previous.manifest.get("encoder", {})
    if (meta.get("name"), meta.get("revision"), meta.get("dimension")) != (
        encoder.name,
        encoder.revision,
        encoder.dimension,
    ):
        return {}
    return {
        _embedding_key(p.title, p.text): previous.vectors[row]
        for row, p in enumerate(previous.passages)
    }


def build_passage_artifact(
    corpus: Corpus,
    encoder: Encoder,
    index_version: str,
    parent_corpus_sha256: str,
    artifact_root: Path,
    config: ChunkConfig = DEFAULT_CONFIG,
    batch_size: int = BATCH_SIZE,
    reuse: PassageStore | None = None,
    stats: dict | None = None,
) -> Path:
    """Chunk and embed every document, then publish the artifact. Returns its directory.

    ``reuse`` (the previously served artifact) lets a re-index skip embedding passages whose
    text is unchanged; ``stats``, if given, receives ``{"reused": n, "encoded": m}``."""
    passages: list[Passage] = []
    for doc in corpus.docs:
        passages += build_passages(
            index_version, doc.key, doc.language, doc.title, doc.summary, config
        )
    if not passages:
        raise PassageArtifactError("the corpus produced no passages")
    titles = {d.key: d.title for d in corpus.docs}

    final = passage_dir(artifact_root, index_version)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        texts = [dense_text(titles[p.doc_key], p.text) for p in passages]
        known = _reusable_vectors(reuse, encoder)
        keys = [_embedding_key(titles[p.doc_key], p.text) for p in passages]
        missing = [i for i, key in enumerate(keys) if key not in known]
        vectors = np.zeros((len(passages), encoder.dimension), dtype=np.float32)
        for i, key in enumerate(keys):
            if key in known:
                vectors[i] = known[key]
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            vectors[chunk] = encoder.encode([texts[i] for i in chunk])
        if stats is not None:
            stats.update(reused=len(passages) - len(missing), encoded=len(missing))
        vectors = normalise(vectors)
        if vectors.shape != (len(passages), encoder.dimension):
            raise PassageArtifactError("encoder returned vectors of an unexpected shape")
        (staging / "passages.jsonl").write_text(
            "".join(_line(p) + "\n" for p in passages), encoding="utf-8"
        )
        np.save(staging / "vectors.npy", vectors, allow_pickle=False)

        files = {name: _sha256(staging / name) for name in DATA_FILES}
        per_language = Counter(p.language for p in passages)
        version = (
            "psg-"
            + hashlib.sha256(
                "\n".join(f"{n}:{h}" for n, h in sorted(files.items())).encode()
            ).hexdigest()[:12]
        )
        manifest = {
            "schema_version": PASSAGE_SCHEMA_VERSION,
            "version": version,
            "chunker": {"version": CHUNKER_VERSION, **asdict(config)},
            "encoder": {
                "name": encoder.name,
                "revision": encoder.revision,
                "dimension": encoder.dimension,
                "text": "title. passage text",
            },
            "parent": {"index_version": index_version, "corpus_sha256": parent_corpus_sha256},
            "documents": len(corpus.docs),
            "passages": len(passages),
            "per_language": {lang: per_language.get(lang, 0) for lang in SUPPORTED_LANGUAGES},
            "files": files,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (staging / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if final.exists():  # same index version rebuilt: replace the old artifact atomically
            old = final.parent / f".old-{uuid.uuid4().hex[:8]}"
            os.rename(final, old)
            os.rename(staging, final)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(staging, final)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ------------------------------------------------------------------------------ load
class PassageStore:
    """A validated, immutable passage artifact held in memory."""

    def __init__(self, path: Path, manifest: dict, passages: list[Passage], vectors: np.ndarray):
        self.path, self.manifest = path, manifest
        self.passages, self.vectors = passages, vectors
        self._rows_by_doc: dict[str, list[int]] = {}
        for row, passage in enumerate(passages):
            self._rows_by_doc.setdefault(passage.doc_key, []).append(row)

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def index_version(self) -> str:
        return self.manifest["parent"]["index_version"]

    @property
    def count(self) -> int:
        return len(self.passages)

    def rows_for(self, doc_keys: Sequence[str]) -> list[int]:
        rows: list[int] = []
        for key in doc_keys:
            rows += self._rows_by_doc.get(key, [])
        return rows

    @classmethod
    def load(
        cls,
        path: Path,
        expect_index_version: str | None = None,
        expect_encoder: tuple[str, str] | None = None,
    ) -> PassageStore:
        """Load and fully validate; raise ``PassageArtifactError`` on any inconsistency."""
        path = Path(path)
        try:
            manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise PassageArtifactError("manifest unreadable") from None
        try:
            if manifest["schema_version"] != PASSAGE_SCHEMA_VERSION:
                raise PassageArtifactError("unsupported passage artifact schema")
            if expect_index_version and manifest["parent"]["index_version"] != expect_index_version:
                raise PassageArtifactError("passages were built for a different index version")
            enc = manifest["encoder"]
            if expect_encoder and (enc["name"], enc["revision"]) != tuple(expect_encoder):
                raise PassageArtifactError("passages were built with a different encoder")
            for name, expected in manifest["files"].items():
                file = path / name
                if not file.is_file():
                    raise PassageArtifactError(f"missing file {name}")
                if _sha256(file) != expected:
                    raise PassageArtifactError(f"checksum mismatch for {name}")
            passages = _read_passages(path / "passages.jsonl")
            vectors = np.load(path / "vectors.npy", allow_pickle=False)
            _validate(manifest, passages, vectors)
        except PassageArtifactError:
            raise
        except (KeyError, TypeError, ValueError, OSError):
            raise PassageArtifactError("passage artifact is malformed") from None
        return cls(path, manifest, passages, vectors)


def _read_passages(path: Path) -> list[Passage]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        out.append(Passage(**json.loads(line)))
    return out


def _validate(manifest: dict, passages: list[Passage], vectors: np.ndarray) -> None:
    dim = manifest["encoder"]["dimension"]
    if not len(passages) == manifest["passages"] == vectors.shape[0]:
        raise PassageArtifactError("passage rows and vector rows differ")
    if vectors.ndim != 2 or vectors.shape[1] != dim or vectors.dtype != np.float32:
        raise PassageArtifactError("vectors have an unexpected shape or type")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3
    ):
        raise PassageArtifactError("vectors are not finite unit vectors")
    if len({p.passage_id for p in passages}) != len(passages):
        raise PassageArtifactError("duplicate passage ids")
    counts = Counter(p.language for p in passages)
    if any(counts.get(lang, 0) != n for lang, n in manifest["per_language"].items()):
        raise PassageArtifactError("per-language counts differ from the manifest")
    for p in passages:
        if content_sha256(p.text) != p.content_sha256 or p.char_end - p.char_start != len(p.text):
            raise PassageArtifactError("a passage does not match its recorded hash or offsets")


# ----------------------------------------------------------------- PostgreSQL sync
def sync_passages(session: Session, store: PassageStore, index_version: IndexVersion) -> dict:
    """Replace the stored passages of ``index_version`` with this artifact's (idempotent).

    Rows whose document is no longer active are skipped and counted, never inserted. The caller
    owns the transaction.
    """
    if store.index_version != index_version.version:
        raise PassageArtifactError("passage artifact belongs to a different index version")
    ids = {
        doc_key(lang, ext): doc_id
        for doc_id, lang, ext in session.execute(
            select(Document.id, Document.language, Document.external_id).where(
                Document.is_active.is_(True)
            )
        )
    }
    session.execute(
        delete(DocumentPassage).where(DocumentPassage.index_version_id == index_version.id)
    )
    rows, skipped = [], 0
    for p in store.passages:
        document_id = ids.get(p.doc_key)
        if document_id is None:
            skipped += 1
            continue
        rows.append(
            dict(
                id=uuid.UUID(p.passage_id),
                index_version_id=index_version.id,
                document_id=document_id,
                ordinal=p.ordinal,
                text=p.text,
                language=p.language,
                char_start=p.char_start,
                char_end=p.char_end,
                estimated_tokens=p.estimated_tokens,
                content_sha256=p.content_sha256,
            )  # fmt: skip
        )
    for start in range(0, len(rows), 1000):
        session.execute(insert(DocumentPassage), rows[start : start + 1000])
    return {"inserted": len(rows), "skipped_inactive_documents": skipped}
