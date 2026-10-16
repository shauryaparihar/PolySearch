"""Build versioned search artifacts: per-language FAISS partitions, a lexical index and a manifest.

Layout of one immutable version directory (``artifacts/index/<version>/``):

    manifest.json             encoder + corpus identity, counts, SHA-256 of every file below
    faiss_<lang>.index        exact inner-product index (IndexIDMap2 over IndexFlatIP)
    mapping_<lang>.json       position -> document key (the FAISS id is the position)
    lexical_*.{json,npy,npz}  character n-gram TF-IDF index (see lexical_index.py)

All formats are pickle-free. The version name is derived from the file hashes, and the
directory is moved into place atomically only after every file was written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np

from app.config import SUPPORTED_LANGUAGES
from app.evaluation.corpus import Corpus
from app.ingestion.text import dense_text, lexical_text
from app.search import lexical_index
from app.search.encoder import Encoder, normalise

FORMAT_VERSION = 1
BATCH_SIZE = 64


class ArtifactError(Exception):
    """An artifact set is missing, corrupt, inconsistent or built from other inputs."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(
    corpus: Corpus, encoder: Encoder, out_root: Path, batch_size: int = BATCH_SIZE
) -> Path:
    """Encode the corpus and write a complete, validated artifact set. Returns its directory."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    staging = out_root / f".staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        docs = corpus.docs
        vectors = []
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            vectors.append(encoder.encode([dense_text(d.title, d.summary) for d in batch]))
        embeddings = normalise(np.vstack(vectors))
        if embeddings.shape != (len(docs), encoder.dimension):
            raise ArtifactError("encoder returned vectors of an unexpected shape")

        partitions = {}
        for language in SUPPORTED_LANGUAGES:
            rows = [i for i, d in enumerate(docs) if d.language == language]
            keys = [docs[i].key for i in rows]
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(encoder.dimension))
            index.add_with_ids(embeddings[rows], np.arange(len(rows), dtype=np.int64))
            faiss.write_index(index, str(staging / f"faiss_{language}.index"))
            (staging / f"mapping_{language}.json").write_text(
                json.dumps(keys, ensure_ascii=False), encoding="utf-8"
            )
            partitions[language] = {
                "documents": len(keys),
                "index_file": f"faiss_{language}.index",
                "mapping_file": f"mapping_{language}.json",
            }

        lexical = lexical_index.LexicalIndex.build(
            [lexical_text(d.title, d.summary) for d in docs],
            [d.key for d in docs],
            [d.language for d in docs],
        )
        lexical.save(staging)

        files = sorted(p.name for p in staging.iterdir() if p.is_file())
        hashes = {name: file_sha256(staging / name) for name in files}
        version = (
            "idx-"
            + hashlib.sha256(
                "\n".join(f"{n}:{h}" for n, h in sorted(hashes.items())).encode()
            ).hexdigest()[:12]
        )
        manifest = {
            "format": FORMAT_VERSION,
            "version": version,
            "encoder": {
                "name": encoder.name,
                "revision": encoder.revision,
                "dimension": encoder.dimension,
                "normalised": True,
                "text": "title. summary (case and punctuation preserved)",
            },
            "corpus": {
                "sha256": corpus.sha256,
                "documents": len(docs),
                "per_language": corpus.language_counts(),
            },
            "partitions": partitions,
            "lexical": {
                "config": lexical_index.CONFIG,
                "vocabulary_size": len(lexical.vocab),
                "documents": len(docs),
            },
            "files": hashes,
            "evaluation": None,
            "default_mode": None,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final = out_root / version
        if final.exists():
            shutil.rmtree(staging)  # identical content already built
            return final
        os.rename(staging, final)  # atomic within one filesystem
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
