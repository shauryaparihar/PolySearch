"""Frozen answer-mode benchmark: case schema, loading, and checks against the real corpus.

Cases are tied to *documents and exact quotes*, not to passage ids, because passage ids change
with every index version. A retrieved passage counts as gold evidence when it comes from the
gold document and contains the gold quote. Validation proves, against the corpus, that every
gold quote exists, that "unanswerable" cases really have no answer text in the corpus, and that
cross-language cases only accept evidence in another language than the question.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BENCHMARK_VERSION = "rag-v1"
CATEGORIES = ("direct", "cross_language", "multi_evidence", "unanswerable", "adversarial")
Category = Literal["direct", "cross_language", "multi_evidence", "unanswerable", "adversarial"]
Language = Literal["en", "es", "fr"]
_ID = re.compile(r"^[a-z]+-[0-9]{2}$")


def fold(text: str) -> str:
    """Case- and accent-insensitive form used for every deterministic text comparison."""
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_key: str
    quote: str = Field(min_length=6)


class GoldItem(BaseModel):
    """One piece of required evidence. Any member of ``any_of`` satisfies it (for example the
    same fact stated in two languages)."""

    model_config = ConfigDict(extra="forbid")

    any_of: list[Quote] = Field(min_length=1)


class InjectedPassage(BaseModel):
    """Synthetic hostile evidence added at run time; it is never part of the index."""

    model_config = ConfigDict(extra="forbid")

    title: str
    text: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    tags: list[str] = []
    question: str = Field(min_length=8, max_length=300)
    question_language: Language
    answerable: bool
    reference_answer: str | None = None  # a concise rubric answer written by hand, not by a model
    key_facts: list[list[str]] = []  # every fact required; any alternative within a fact is enough
    gold_evidence: list[GoldItem] = []
    absent_strings: list[str] = []  # unanswerable: text that must not occur anywhere in the corpus
    injected_passages: list[InjectedPassage] = []
    forbidden_strings: list[str] = []  # adversarial: must never appear in the answer

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("id must look like direct-01")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if self.answerable:
            if not (self.reference_answer and self.key_facts and self.gold_evidence):
                raise ValueError("an answerable case needs a reference answer, key facts and gold")
            if any(not fact or any(not alt.strip() for alt in fact) for fact in self.key_facts):
                raise ValueError("every key fact needs at least one non-empty alternative")
        else:
            if self.gold_evidence or self.key_facts or not self.absent_strings:
                raise ValueError("an unanswerable case has no gold and lists absent_strings")
        if (self.category == "unanswerable") == self.answerable:
            raise ValueError("category 'unanswerable' and answerable=false must agree")
        if self.category == "adversarial" and not (
            self.injected_passages and self.forbidden_strings
        ):
            raise ValueError("an adversarial case needs injected passages and forbidden strings")
        if self.category == "multi_evidence" and len(self.gold_evidence) < 2:
            raise ValueError("a multi-evidence case needs at least two gold items")
        return self


class Benchmark(BaseModel):
    version: str
    cases: list[Case]
    sha256: str

    def by_category(self) -> dict[str, int]:
        counts = Counter(c.category for c in self.cases)
        return {c: counts.get(c, 0) for c in CATEGORIES}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(path: Path) -> Benchmark:
    path = Path(path)
    cases = [
        Case.model_validate(json.loads(line))
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return Benchmark(version=BENCHMARK_VERSION, cases=cases, sha256=file_sha256(path))


def validate_against_corpus(
    cases: Sequence[Case], summaries: dict[str, str], languages: dict[str, str]
) -> list[str]:
    """Return human-readable problems (empty when the benchmark is consistent with the corpus).

    ``summaries`` maps document key -> summary text, ``languages`` maps document key -> language.
    """
    problems: list[str] = []
    folded = {key: fold(text) for key, text in summaries.items()}
    everything = "\n".join(folded.values())
    for case in cases:
        for item in case.gold_evidence:
            for q in item.any_of:
                if q.doc_key not in folded:
                    problems.append(f"{case.id}: gold document {q.doc_key} is not in the corpus")
                elif fold(q.quote) not in folded[q.doc_key]:
                    problems.append(f"{case.id}: quote not found in {q.doc_key}: {q.quote!r}")
                elif case.category == "cross_language" and (
                    languages[q.doc_key] == case.question_language
                ):
                    problems.append(f"{case.id}: cross-language gold is in the question language")
        for absent in case.absent_strings:
            if fold(absent) in everything:
                problems.append(f"{case.id}: 'absent' text occurs in the corpus: {absent!r}")
        for injected in case.injected_passages:
            if fold(injected.text) in everything:
                problems.append(f"{case.id}: injected passage text occurs in the corpus")
        for forbidden in case.forbidden_strings:
            if fold(forbidden) in everything:
                problems.append(f"{case.id}: forbidden string occurs in the corpus: {forbidden!r}")
    return problems


def quotes_inside_passages(
    cases: Iterable[Case], passages_by_doc: dict[str, list[str]]
) -> list[str]:
    """Each gold quote must lie wholly inside at least one passage (not be split by chunking)."""
    problems = []
    for case in cases:
        for item in case.gold_evidence:
            for q in item.any_of:
                texts = passages_by_doc.get(q.doc_key, [])
                if not any(fold(q.quote) in fold(t) for t in texts):
                    problems.append(f"{case.id}: quote split across passages in {q.doc_key}")
    return problems
