"""Deterministic metrics for answer-mode evaluation. Pure functions: no model, no network.

Retrieval quality and generation quality are measured separately. Everything here is a
transparent rule (substring and set arithmetic), so a number can be reproduced by hand.
Lexical "support" is a *proxy* for groundedness, not an entailment judgement.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.evaluation.rag_benchmark import GoldItem, fold

_TOKEN = re.compile(r"[^\W_]{4,}", re.UNICODE)
SUPPORT_THRESHOLD = 0.6  # share of a claim's content words that must appear in cited text


@dataclass(frozen=True)
class RetrievedPassage:
    doc_key: str
    text: str


def matches_gold_quote(passage: RetrievedPassage, doc_key: str, quote: str) -> bool:
    return passage.doc_key == doc_key and fold(quote) in fold(passage.text)


def item_satisfied(item: GoldItem, passages: Iterable[RetrievedPassage]) -> bool:
    passages = list(passages)
    return any(matches_gold_quote(p, q.doc_key, q.quote) for q in item.any_of for p in passages)


def passage_recall_at_k(
    gold: Sequence[GoldItem], ranked: Sequence[RetrievedPassage], k: int
) -> float | None:
    """Share of gold evidence items found among the first ``k`` passages (None: no gold)."""
    if not gold:
        return None
    top = ranked[:k]
    return sum(item_satisfied(item, top) for item in gold) / len(gold)


def passage_reciprocal_rank(
    gold: Sequence[GoldItem], ranked: Sequence[RetrievedPassage]
) -> float | None:
    """1 / rank of the first passage that is gold evidence (0 when none is retrieved)."""
    if not gold:
        return None
    for rank, passage in enumerate(ranked, start=1):
        if any(matches_gold_quote(passage, q.doc_key, q.quote) for g in gold for q in g.any_of):
            return 1.0 / rank
    return 0.0


def document_recall_at_k(
    gold: Sequence[GoldItem], ranked_doc_keys: Sequence[str], k: int
) -> float | None:
    """Document-level recall for search-only retrieval (no passages involved)."""
    if not gold:
        return None
    top = set(ranked_doc_keys[:k])
    return sum(any(q.doc_key in top for q in item.any_of) for item in gold) / len(gold)


# ------------------------------------------------------------------------ answer content
def key_fact_hits(answer_text: str, key_facts: Sequence[Sequence[str]]) -> tuple[int, int]:
    """(facts found, facts required); a fact is found when any alternative occurs (folded)."""
    text = fold(answer_text)
    found = sum(any(fold(alt) in text for alt in fact) for fact in key_facts)
    return found, len(key_facts)


def contains_forbidden(answer_text: str, forbidden: Iterable[str]) -> bool:
    text = fold(answer_text)
    return any(fold(f) in text for f in forbidden)


def citation_precision(cited: Sequence[RetrievedPassage], gold: Sequence[GoldItem]) -> float | None:
    """Share of cited passages that are gold evidence (None when nothing was cited)."""
    if not cited:
        return None
    ok = sum(
        any(matches_gold_quote(p, q.doc_key, q.quote) for g in gold for q in g.any_of)
        for p in cited
    )
    return ok / len(cited)


def citation_recall(cited: Sequence[RetrievedPassage], gold: Sequence[GoldItem]) -> float | None:
    """Share of gold evidence items that some cited passage covers."""
    if not gold:
        return None
    return sum(item_satisfied(item, cited) for item in gold) / len(gold)


def claim_support(claim_text: str, cited_texts: Iterable[str]) -> float:
    """Share of the claim's content words (4+ letters) that occur in the cited passages."""
    words = set(_TOKEN.findall(fold(claim_text)))
    if not words:
        return 1.0
    haystack = fold(" ".join(cited_texts))
    return sum(w in haystack for w in words) / len(words)


_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_NAME = re.compile(r"(?<=[\s(\"'])[^\W\d_]*[A-ZÁÉÍÓÚÑÀÂÇÈÊËÎÏÔÙÛÜ][^\W\d_]+")


def strict_claim_supported(claim_text: str, cited_texts: Iterable[str]) -> bool:
    """Every number and every capitalised word after the first in the claim must occur in the
    cited passages. Stricter than ``claim_support``: a swapped date or name fails. Still a text
    check, not entailment: a claim can reuse the right words and say something false."""
    haystack = fold(" ".join(cited_texts))
    needed = [fold(x) for x in _NUMBER.findall(claim_text)]
    needed += [fold(x) for x in _NAME.findall(claim_text)]
    return all(item in haystack for item in needed)


def strict_supported_claim_rate(claims: Sequence[tuple[str, Sequence[str]]]) -> float | None:
    if not claims:
        return None
    return sum(strict_claim_supported(t, texts) for t, texts in claims) / len(claims)


def supported_claim_rate(
    claims: Sequence[tuple[str, Sequence[str]]], threshold: float = SUPPORT_THRESHOLD
) -> float | None:
    """``claims`` = (claim text, texts of the passages it cites)."""
    if not claims:
        return None
    return sum(claim_support(t, texts) >= threshold for t, texts in claims) / len(claims)


# ------------------------------------------------------------------------- abstention
@dataclass(frozen=True)
class AbstentionMatrix:
    """Positive class = "the system should abstain" (the question is unanswerable)."""

    tp: int  # unanswerable and abstained
    fp: int  # answerable but abstained
    fn: int  # unanswerable but answered (hallucination risk)
    tn: int  # answerable and answered
    fallbacks: int  # excluded: the system did not complete (neither answered nor abstained)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None


def abstention_matrix(outcomes: Iterable[tuple[bool, str]]) -> AbstentionMatrix:
    """``outcomes`` = (case is answerable, status); status is answered, abstained or anything else
    (fallback, failed), which is counted separately."""
    tp = fp = fn = tn = fallbacks = 0
    for answerable, status in outcomes:
        if status == "abstained":
            fp, tp = (fp + 1, tp) if answerable else (fp, tp + 1)
        elif status == "answered":
            tn, fn = (tn + 1, fn) if answerable else (tn, fn + 1)
        else:
            fallbacks += 1
    return AbstentionMatrix(tp, fp, fn, tn, fallbacks)


# --------------------------------------------------------------------------- numbers
def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def rate(count: int, total: int) -> float | None:
    return count / total if total else None
