"""Text preparation. Dense and lexical retrieval deliberately get different text.

* ``dense_text``: natural text. Case, punctuation and function words are kept because
  transformer encoders were trained on natural language. Only markup, invisible
  characters and whitespace are cleaned.
* ``lexical_text``: normalised for exact-term matching: NFKC, case-folded and accent-folded
  (so "Educación" matches "educacion"). No stop-word removal and no stemming: an English
  stop-word list would damage Spanish and French, and we do not claim stemming we have
  not built and evaluated.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)[^>]*>")
# Tags that separate words/lines; removing them must leave a space. Inline tags (b, i, a...) are
# removed without one so "<b>word</b>." stays "word.".
_BLOCK_TAGS = {
    "br",
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "tr",
    "td",
    "th",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
}
_WS_RE = re.compile(r"\s+")
# Zero-width / formatting characters that appear in Wikipedia extracts but carry no
# meaning in English, Spanish or French text.
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")


def clean_text(text: str) -> str:
    """Remove markup/invisible characters, normalise Unicode (NFC) and whitespace."""
    text = html.unescape(text)
    text = _TAG_RE.sub(lambda m: " " if m.group(1).lower() in _BLOCK_TAGS else "", text)
    text = _INVISIBLE_RE.sub("", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text).strip()


def bound_text(text: str, max_chars: int) -> str:
    """Truncate to ``max_chars`` at a word boundary (no ellipsis added)."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars // 2 else cut).rstrip()


def dense_text(title: str, summary: str) -> str:
    return f"{title}. {summary}" if summary else title


def lexical_text(title: str, summary: str) -> str:
    text = unicodedata.normalize("NFKC", f"{title} {summary}").casefold()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", text).strip()


def content_sha256(title: str, summary: str) -> str:
    """Deterministic fingerprint of the cleaned title + summary (revision-independent)."""
    canonical = f"{clean_text(title)}\x1f{clean_text(summary)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
