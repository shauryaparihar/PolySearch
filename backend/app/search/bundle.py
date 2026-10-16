"""The immutable SearchBundle: validated artifacts plus dense, lexical and hybrid search."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from app.config import SUPPORTED_LANGUAGES
from app.search import lexical_index
from app.search.artifacts import FORMAT_VERSION, ArtifactError, file_sha256
from app.search.encoder import Encoder, SentenceTransformerEncoder
from app.search.fusion import RRF_K, rrf_fuse

MODES = ("dense", "lexical", "hybrid")
DEFAULT_POOL = 50
MAX_POOL = 200
MAX_K = 100


@dataclass(frozen=True)
class Hit:
    key: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class SearchBundle:
    def __init__(
        self,
        path: Path,
        manifest: dict,
        indexes: dict[str, faiss.Index],
        mappings: dict[str, list[str]],
        lexical: lexical_index.LexicalIndex,
        encoder: Encoder | None,
    ):
        self.path = path
        self.manifest = manifest
        self._indexes = indexes
        self._mappings = mappings
        self._lexical = lexical
        self._encoder = encoder

    @property
    def version(self) -> str:
        return self.manifest["version"]

    # --------------------------------------------------------------------- load
    @classmethod
    def load(
        cls, path: Path, encoder: Encoder | None = None, load_encoder: bool = True
    ) -> SearchBundle:
        """Load and fully validate an artifact set; raise ``ArtifactError`` on any mismatch.

        The encoder is created and warmed up *before* any FAISS operation. On macOS, running
        the transformer for the first time after FAISS has initialised its OpenMP runtime
        crashes the process, so this order is required, not cosmetic. ``load_encoder=False``
        gives a lexical-only bundle (dense and hybrid searches then raise).
        """
        path = Path(path)
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"manifest unreadable: {type(exc).__name__}") from None
        if manifest.get("format") != FORMAT_VERSION:
            raise ArtifactError(f"unsupported artifact format {manifest.get('format')!r}")

        if encoder is None and load_encoder:
            enc = manifest["encoder"]
            encoder = SentenceTransformerEncoder(enc["name"], enc["revision"])
        if encoder is not None:
            encoder.encode(["warm-up"])  # initialise the transformer's threads before FAISS

        for name, expected in manifest.get("files", {}).items():
            file = path / name
            if not file.is_file():
                raise ArtifactError(f"missing artifact file {name}")
            if file_sha256(file) != expected:
                raise ArtifactError(f"checksum mismatch for {name}")

        dimension = manifest["encoder"]["dimension"]
        indexes, mappings = {}, {}
        for language in SUPPORTED_LANGUAGES:
            part = manifest["partitions"].get(language)
            if part is None:
                raise ArtifactError(f"manifest has no partition for {language}")
            keys = json.loads((path / part["mapping_file"]).read_text(encoding="utf-8"))
            index = faiss.read_index(str(path / part["index_file"]))
            if index.d != dimension:
                raise ArtifactError(f"{language} index dimension {index.d} != manifest {dimension}")
            if not (index.ntotal == len(keys) == part["documents"]):
                raise ArtifactError(f"{language} document count mismatch")
            if len(set(keys)) != len(keys) or any(not k.startswith(f"{language}:") for k in keys):
                raise ArtifactError(f"{language} mapping contains duplicate or foreign keys")
            indexes[language], mappings[language] = index, keys

        lexical = lexical_index.LexicalIndex.load(path)
        total = sum(len(v) for v in mappings.values())
        if manifest["lexical"]["config"] != lexical_index.CONFIG:
            raise ArtifactError("lexical configuration differs from this code version")
        if not (
            lexical.matrix.shape[0] == len(lexical.keys) == total == manifest["corpus"]["documents"]
        ):
            raise ArtifactError("lexical index and dense partitions cover different documents")
        if lexical.matrix.shape[1] != len(lexical.vocab) or len(lexical.idf) != len(lexical.vocab):
            raise ArtifactError("lexical vocabulary size mismatch")
        if set(lexical.keys.tolist()) != {k for ks in mappings.values() for k in ks}:
            raise ArtifactError("lexical index and dense partitions use different document keys")

        if encoder is not None:
            enc = manifest["encoder"]
            if encoder.dimension != dimension:
                raise ArtifactError("supplied encoder dimension differs from the artifacts")
            if (encoder.name, encoder.revision) != (enc["name"], enc["revision"]):
                raise ArtifactError("supplied encoder is not the one the artifacts were built with")
        return cls(path, manifest, indexes, mappings, lexical, encoder)

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            raise ArtifactError("bundle was loaded without an encoder; dense search is unavailable")
        return self._encoder

    def document_keys(self) -> set[str]:
        return {k for ks in self._mappings.values() for k in ks}

    # ------------------------------------------------------------------- search
    def _dense(
        self, vector: np.ndarray, languages: Sequence[str], pool: int
    ) -> list[tuple[str, float]]:
        merged: list[tuple[str, float]] = []
        for language in languages:
            index = self._indexes[language]
            scores, ids = index.search(vector[None, :], min(pool, index.ntotal))
            merged += [
                (self._mappings[language][int(i)], float(s))
                for i, s in zip(ids[0], scores[0], strict=True)
                if i >= 0
            ]
        merged.sort(key=lambda item: (-item[1], item[0]))
        return merged[:pool]

    def search(
        self,
        query: str,
        mode: str,
        target_languages: Sequence[str],
        k: int = 10,
        pool: int = DEFAULT_POOL,
    ) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        languages = sorted(set(target_languages))
        if not languages or any(lang not in SUPPORTED_LANGUAGES for lang in languages):
            raise ValueError(
                "target_languages must be a non-empty subset of the supported languages"
            )
        if not 1 <= k <= MAX_K or not 1 <= pool <= MAX_POOL:
            raise ValueError("k or pool out of range")
        pool = max(pool, k)

        from app.ingestion.text import lexical_text

        t: dict[str, float] = {}
        total_start = time.perf_counter()
        dense_hits: list[tuple[str, float]] = []
        lexical_hits: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid"):
            s = time.perf_counter()
            vector = self.encoder.encode([query])[0]
            t["encode_ms"] = (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            dense_hits = self._dense(vector, languages, pool)
            t["faiss_ms"] = (time.perf_counter() - s) * 1000
        if mode in ("lexical", "hybrid"):
            s = time.perf_counter()
            lexical_hits = self._lexical.search(lexical_text(query, ""), languages, pool)
            t["lexical_ms"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        dense_info = {key: (r, sc) for r, (key, sc) in enumerate(dense_hits, start=1)}
        lexical_info = {key: (r, sc) for r, (key, sc) in enumerate(lexical_hits, start=1)}
        if mode == "dense":
            ordered = [(key, sc, {"dense": r}) for key, (r, sc) in dense_info.items()]
        elif mode == "lexical":
            ordered = [(key, sc, {"lexical": r}) for key, (r, sc) in lexical_info.items()]
        else:
            ordered = rrf_fuse(
                {"dense": [k_ for k_, _ in dense_hits], "lexical": [k_ for k_, _ in lexical_hits]},
                RRF_K,
            )
        hits = [
            Hit(
                key=key,
                score=score,
                dense_rank=ranks.get("dense"),
                lexical_rank=ranks.get("lexical"),
                dense_score=dense_info[key][1] if key in dense_info else None,
                lexical_score=lexical_info[key][1] if key in lexical_info else None,
            )
            for key, score, ranks in ordered[:k]
        ]
        t["fusion_ms"] = (time.perf_counter() - s) * 1000
        t["total_ms"] = (time.perf_counter() - total_start) * 1000
        return SearchResult(hits, t)


class BundleRetriever:
    """Adapter exposing one bundle mode through the evaluation ``Retriever`` interface."""

    NAMES = {"dense": "dense-minilm", "lexical": "lexical-tfidf-char", "hybrid": "hybrid-rrf"}

    def __init__(self, bundle: SearchBundle, mode: str):
        self.bundle, self.mode = bundle, mode
        self.name = self.NAMES[mode]
        enc = bundle.manifest["encoder"]
        if mode == "lexical":
            self.config = dict(lexical_index.CONFIG)
        elif mode == "dense":
            self.config = {"encoder": enc["name"], "revision": enc["revision"], "index": "exact IP"}
        else:
            self.config = {
                "fusion": "reciprocal rank fusion",
                "rrf_k": RRF_K,
                "candidate_pool": DEFAULT_POOL,
                "encoder": enc["name"],
                "revision": enc["revision"],
            }

    def search(self, query: str, target_languages: list[str], k: int) -> list[tuple[str, float]]:
        result = self.bundle.search(query, self.mode, target_languages, k=k)
        return [(h.key, h.score) for h in result.hits]
