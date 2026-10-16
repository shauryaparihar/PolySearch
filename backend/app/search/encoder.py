"""Text encoders. The real encoder is imported lazily so tests never download a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Encoder(Protocol):
    name: str
    revision: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return unit-length float32 vectors, shape (len(texts), dimension)."""


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise to float32 and refuse non-finite or zero vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embeddings contain a zero vector")
    return (vectors / norms).astype(np.float32)


def resolve_revision(name: str) -> str:
    """Latest commit sha of a Hugging Face model repository (used once, then pinned)."""
    from huggingface_hub import HfApi

    sha = HfApi().model_info(name).sha
    if not sha:
        raise RuntimeError(f"could not resolve a revision for {name}")
    return sha


class SentenceTransformerEncoder:
    def __init__(self, name: str = DEFAULT_ENCODER, revision: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.revision = revision or resolve_revision(name)
        self._model = SentenceTransformer(name, revision=self.revision, device="cpu")
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return normalise(vectors)
