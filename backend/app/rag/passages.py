"""Deterministic passage chunking for the optional answer mode.

The stored documents are Wikipedia *summaries* (one block of text, at most 2,000 characters, no
paragraph breaks), so a "passage" here is a window of whole sentences from that summary. The
same text and settings always produce the same passages, offsets and ids.

Rules, in order:
1. split the text into sentences (a small, documented heuristic, not a language model);
2. a sentence longer than the limit is cut at word boundaries into pieces;
3. pack consecutive sentences greedily while the window stays within ``max_chars``;
4. the next window repeats the previous window's last sentence (overlap) when that sentence is
   short enough and the repeat will still be followed by new text.

Every passage is an exact slice of the source text, so ``text == source[char_start:char_end]``.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

CHUNKER_VERSION = "sentence-window-v1"
CHARS_PER_TOKEN = 3  # deliberately low: it over-estimates tokens so budgets err on the safe side
# Fixed namespace so passage ids are reproducible on any machine.
_ID_NAMESPACE = uuid.UUID("5b1f2c0e-8a47-4d0b-9f55-6a1d3e7c2b90")

# A sentence ends at . ! ? … (optionally followed by closing quotes/brackets), then whitespace,
# then something that looks like the start of a sentence.
_BOUNDARY = re.compile(r"(?P<end>[.!?…]+[\"”»)'’\]]*)(?P<gap>\s+)(?=[\"“«(\[¿¡]?[A-ZÀ-ÖØ-Þ0-9])")
_WORD_BEFORE = re.compile(r"([^\W\d_]+)$")
# Abbreviations whose full stop is not a sentence end (lower-case, no dot).
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms st jr sr vs no mt prof sra mme mlle etc fig inc ltd co ca cf av dept".split()
)


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 600  # about 200 estimated tokens
    overlap_max_chars: int = 250  # only a short last sentence is repeated

    @classmethod
    def from_max_tokens(cls, max_tokens: int) -> ChunkConfig:
        return cls(max_chars=max_tokens * CHARS_PER_TOKEN)


DEFAULT_CONFIG = ChunkConfig()


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_key: str
    language: str
    title: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    estimated_tokens: int
    content_sha256: str


def estimate_tokens(text: str) -> int:
    """Conservative deterministic estimate (no tokenizer needed): ceil(characters / 3)."""
    return -(-len(text) // CHARS_PER_TOKEN) if text else 0


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def passage_id(index_version: str, doc_key: str, ordinal: int, sha256: str) -> str:
    """Stable id from the index version, the document, the position and the content hash."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{index_version}|{doc_key}|{ordinal}|{sha256}"))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences, whitespace-trimmed, in order."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end_punct = match.group("end")
        before = _WORD_BEFORE.search(text[: match.start()])
        word = before.group(1) if before else ""
        is_abbreviation = end_punct == "." and (
            word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isupper())
        )
        if is_abbreviation:
            continue
        spans.append((start, match.start("gap")))
        start = match.end("gap")
    spans.append((start, len(text)))
    return [(a, b) for a, b in _trim(text, spans) if b > a]


def _trim(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        out.append((a, b))
    return out


def _cut_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut one over-long span into pieces of at most ``max_chars``, at word boundaries."""
    pieces = []
    while end - start > max_chars:
        limit = start + max_chars
        cut = text.rfind(" ", start + 1, limit + 1)
        if cut <= start:  # no space at all (e.g. a very long token): cut at the limit
            cut = limit
        pieces.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if end > start:
        pieces.append((start, end))
    return pieces


def chunk_spans(text: str, config: ChunkConfig = DEFAULT_CONFIG) -> list[tuple[int, int]]:
    """The ``(start, end)`` character span of every passage of ``text``."""
    units: list[tuple[int, int, bool]] = []  # (start, end, is_piece_of_a_long_sentence)
    for a, b in sentence_spans(text):
        if b - a <= config.max_chars:
            units.append((a, b, False))
        else:
            units.extend((x, y, True) for x, y in _cut_long(text, a, b, config.max_chars))
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        start = units[i][0]
        j = i
        while j + 1 < len(units) and units[j + 1][1] - start <= config.max_chars:
            j += 1
        chunks.append((start, units[j][1]))
        if j + 1 >= len(units):
            break
        last_start, last_end, is_piece = units[j]
        next_end = units[j + 1][1]
        can_overlap = (
            j > i
            and not is_piece
            and last_end - last_start <= config.overlap_max_chars
            and next_end - last_start <= config.max_chars  # the repeat is followed by new text
        )
        i = j if can_overlap else j + 1
    return chunks


def build_passages(
    index_version: str,
    doc_key: str,
    language: str,
    title: str,
    text: str,
    config: ChunkConfig = DEFAULT_CONFIG,
) -> list[Passage]:
    passages = []
    for ordinal, (start, end) in enumerate(chunk_spans(text, config)):
        piece = text[start:end]
        sha = content_sha256(piece)
        passages.append(
            Passage(
                passage_id=passage_id(index_version, doc_key, ordinal, sha),
                doc_key=doc_key,
                language=language,
                title=title,
                ordinal=ordinal,
                text=piece,
                char_start=start,
                char_end=end,
                estimated_tokens=estimate_tokens(piece),
                content_sha256=sha,
            )
        )
    return passages
