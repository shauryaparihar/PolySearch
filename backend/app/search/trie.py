"""Unicode-aware prefix trie for autocomplete.

* Matching is on a *folded* key: NFKC, case-folded, accents removed, whitespace collapsed, so
  "volc", "VOLC" and "Volcán" all lead to "Volcán". The original display string is preserved.
* Each node pre-stores its best ``max_per_node`` entries, so a lookup is O(prefix length) with
  no scan of the subtree. Memory is O(total title characters), tiny for a few thousand titles.
* Built once when an index bundle becomes active; the finished trie is immutable, so searches
  can read it without locks while a replacement is built.
* Duplicates (same folded key, e.g. "Jazz" in three languages) collapse to one suggestion that
  records every language it exists in.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")
MAX_PER_NODE = 10


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return _WS.sub(" ", text).strip()


@dataclass
class Entry:
    display: str
    weight: float = 0.0
    languages: set[str] = field(default_factory=set)

    def sort_key(self) -> tuple:
        # Higher popularity first, then shorter (closer to the prefix), then alphabetical.
        return (-self.weight, len(self.display), self.display.casefold(), self.display)


class _Node:
    __slots__ = ("children", "top")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.top: list[Entry] = []


class PrefixTrie:
    def __init__(self, max_per_node: int = MAX_PER_NODE):
        self.max_per_node = max_per_node
        self._entries: dict[str, Entry] = {}
        self._root: _Node | None = None

    def add(self, display: str, weight: float = 0.0, languages: tuple[str, ...] | set[str] = ()):
        key = fold(display)
        if not key:
            return
        existing = self._entries.get(key)
        if existing is None:
            self._entries[key] = Entry(display.strip(), weight, set(languages))
        else:
            existing.weight = max(existing.weight, weight)
            existing.languages |= set(languages)
        self._root = None  # needs (re)building

    def build(self) -> PrefixTrie:
        root = _Node()
        # Best entries first: each node's list then fills with its top results automatically.
        for key, entry in sorted(self._entries.items(), key=lambda kv: kv[1].sort_key()):
            node = root
            for char in key:
                node = node.children.setdefault(char, _Node())
                if len(node.top) < self.max_per_node:
                    node.top.append(entry)
        self._root = root
        return self

    def __len__(self) -> int:
        return len(self._entries)

    def suggest(self, prefix: str, limit: int = 8) -> list[Entry]:
        if self._root is None:
            self.build()
        folded = fold(prefix)
        if not folded:
            return []  # an empty prefix is not an autocomplete request
        node = self._root
        for char in folded:
            node = node.children.get(char)
            if node is None:
                return []
        return node.top[:limit]


@dataclass(frozen=True)
class Suggester:
    """One trie per language plus a combined one, built together for one index version."""

    index_version: str
    all: PrefixTrie
    by_language: dict[str, PrefixTrie]

    def suggest(self, prefix: str, language: str | None, limit: int) -> list[Entry]:
        trie = self.by_language.get(language) if language else self.all
        return trie.suggest(prefix, limit) if trie else []


# --------------------------------------------------------------------- privacy
_EMAIL_OR_URL = re.compile(r"@|https?://|www\.", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"\d{6,}")
MAX_POPULAR_LENGTH = 60


def looks_sensitive(text: str) -> bool:
    """Conservative filter for query text that must never be suggested to other users:
    anything resembling an email address, URL, long number (phone, card, id) or a very long
    free-form string."""
    return bool(
        _EMAIL_OR_URL.search(text) or _LONG_DIGITS.search(text) or len(text) > MAX_POPULAR_LENGTH
    )
