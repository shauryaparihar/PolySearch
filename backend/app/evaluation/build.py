"""Build the cross-language benchmark from Wikipedia interlanguage links.

Concepts are a fixed hand-chosen seed list of English article titles. Each is resolved through
MediaWiki language links to its Spanish and French articles; only concepts present in all three
languages are kept. The result is a *constructed* benchmark, not real user judgements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import SUPPORTED_LANGUAGES
from app.db.models import Document
from app.evaluation.benchmark import QRELS_FIELDS, QUERY_FIELDS
from app.evaluation.corpus import doc_key
from app.evidence import BenchmarkValidationError
from app.ingestion import service
from app.ingestion.service import LEGACY_MAX_CONTENT, LEGACY_MIN_CONTENT
from app.ingestion.text import bound_text, lexical_text
from app.ingestion.wikipedia import WikipediaSource

SEED_CONCEPTS = [
    "Climate change",
    "Photosynthesis",
    "Vaccine",
    "Internet",
    "Renaissance",
    "Industrial Revolution",
    "World War II",
    "Democracy",
    "Black hole",
    "DNA",
    "Evolution",
    "Solar System",
    "Volcano",
    "Earthquake",
    "Tsunami",
    "Cancer",
    "Diabetes",
    "Antibiotic",
    "Printing press",
    "Telescope",
    "Electricity",
    "Nuclear power",
    "Solar energy",
    "Wind power",
    "Artificial intelligence",
    "Cryptography",
    "Olympic Games",
    "Association football",
    "Basketball",
    "Chess",
    "Opera",
    "Jazz",
    "Impressionism",
    "Cubism",
    "William Shakespeare",
    "Leonardo da Vinci",
    "Isaac Newton",
    "Albert Einstein",
    "Marie Curie",
    "Amazon River",
    "Mount Everest",
    "Sahara",
    "Great Barrier Reef",
    "Coral reef",
    "Rainforest",
    "Biodiversity",
    "Inflation",
    "Globalization",
    "Vitamin",
    "Glacier",
]
MIN_CONCEPTS = 20
LEAD_MIN_CHARS = 30
LEAD_MAX_CHARS = 200
_PARENS = re.compile(r"\([^()]*\)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def lead_sentence(summary: str) -> str | None:
    """First sentence with parenthetical asides removed (they hold foreign-language names and
    pronunciations that would leak the answer). None if too short to be a useful query."""
    text = summary
    while _PARENS.search(text):
        text = _PARENS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    first = _SENTENCE_END.split(text, maxsplit=1)[0]
    first = bound_text(first, LEAD_MAX_CHARS)
    return first if len(first) >= LEAD_MIN_CHARS else None


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_concepts(source: WikipediaSource, seeds: list[str]) -> tuple[list[dict], list[dict]]:
    """Return (complete concept groups, dropped seeds with reasons)."""
    dropped: list[dict] = []
    canonical_en: dict[str, str] = {}
    for chunk in _chunks(seeds, 40):
        for seed, title in source.resolve_titles("en", chunk).items():
            if title is None:
                dropped.append({"seed": seed, "reason": "english page missing"})
            else:
                canonical_en[seed] = title

    linked: dict[str, dict[str, str]] = {seed: {"en": t} for seed, t in canonical_en.items()}
    for target in ("es", "fr"):
        en_titles = sorted(set(canonical_en.values()))
        raw_links: dict[str, str | None] = {}
        for chunk in _chunks(en_titles, 40):
            raw_links.update(source.fetch_langlinks("en", chunk, target))
        wanted = sorted({t for t in raw_links.values() if t})
        resolved: dict[str, str | None] = {}
        for chunk in _chunks(wanted, 40):
            resolved.update(source.resolve_titles(target, chunk))
        for seed, en_title in canonical_en.items():
            link = raw_links.get(en_title)
            final = resolved.get(link) if link else None
            if final:
                linked[seed][target] = final

    groups = []
    for index, seed in enumerate(seeds):
        if seed not in canonical_en:
            continue
        titles = linked[seed]
        missing = [lang for lang in SUPPORTED_LANGUAGES if lang not in titles]
        if missing:
            dropped.append({"seed": seed, "reason": f"no linked page in {','.join(missing)}"})
            continue
        groups.append({"concept_id": f"c{index:03d}", "seed_title": seed, "titles": titles})
    return groups, dropped


def build_benchmark(
    source: WikipediaSource,
    factory: sessionmaker[Session],
    out_dir: Path,
    seeds: list[str] | None = None,
    min_concepts: int = MIN_CONCEPTS,
) -> dict:
    seeds = list(SEED_CONCEPTS if seeds is None else seeds)
    groups, dropped = resolve_concepts(source, seeds)

    # Make sure every linked page is in the corpus (through the normal ingestion path).
    for language in SUPPORTED_LANGUAGES:
        titles = sorted({g["titles"][language] for g in groups})
        for chunk in _chunks(titles, 100):
            service.sync_titles(factory, source, language, chunk)

    with factory() as session:
        docs = {
            doc_key(d.language, d.external_id): d
            for d in session.scalars(select(Document).where(Document.is_active.is_(True)))
        }

    concepts, used_docs = [], set()
    for g in groups:
        keys = {
            lang: doc_key(lang, g["titles"][lang].replace(" ", "_")) for lang in SUPPORTED_LANGUAGES
        }
        missing = [lang for lang, k in keys.items() if k not in docs]
        if missing:
            dropped.append(
                {
                    "seed": g["seed_title"],
                    "reason": f"not in corpus after sync: {','.join(missing)}",
                }
            )
            continue
        if used_docs & set(keys.values()):
            dropped.append(
                {"seed": g["seed_title"], "reason": "documents already used by another concept"}
            )
            continue
        used_docs |= set(keys.values())
        concepts.append(
            {
                "concept_id": g["concept_id"],
                "seed_title": g["seed_title"],
                "documents": {
                    lang: {
                        "key": keys[lang],
                        "title": docs[keys[lang]].title,
                        "source_url": docs[keys[lang]].source_url,
                        "source_revision": docs[keys[lang]].source_revision,
                    }
                    for lang in SUPPORTED_LANGUAGES
                },
            }
        )
    if len(concepts) < min_concepts:
        raise BenchmarkValidationError(
            f"only {len(concepts)} complete concepts resolved (need {min_concepts}); "
            "refusing to pad the benchmark"
        )

    queries, qrels = [], []
    for position, concept in enumerate(concepts):
        qlang = SUPPORTED_LANGUAGES[position % len(SUPPORTED_LANGUAGES)]  # rotate query language
        own = docs[concept["documents"][qlang]["key"]]
        texts = [("title", own.title)]
        lead = lead_sentence(own.summary)
        if lead:
            texts.append(("lead_sentence", lead))
        targets = {lang: d for lang, d in concept["documents"].items() if lang != qlang}
        target_titles = {lexical_text(d["title"], "") for d in targets.values()}
        informative = {t for t in target_titles if len(t) > 3}
        for qtype, text in texts:
            query_id = f"{concept['concept_id']}-{qlang}-{qtype}"
            queries.append(
                {
                    "query_id": query_id,
                    "concept_id": concept["concept_id"],
                    "query_language": qlang,
                    "query_type": qtype,
                    "query_text": text,
                    "text_equals_target_title": str(
                        lexical_text(text, "") in target_titles
                    ).lower(),
                    "text_contains_target_title": str(
                        any(t in lexical_text(text, "") for t in informative)
                    ).lower(),
                }
            )
            for lang, d in targets.items():
                qrels.append(
                    {
                        "query_id": query_id,
                        "doc_key": d["key"],
                        "doc_language": lang,
                        "relevance": 1,
                    }
                )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries.sort(key=lambda r: r["query_id"])
    qrels.sort(key=lambda r: (r["query_id"], r["doc_key"]))
    _write_csv(out_dir / "queries.csv", QUERY_FIELDS, queries)
    _write_csv(out_dir / "qrels.csv", QRELS_FIELDS, qrels)
    (out_dir / "concepts.json").write_text(
        json.dumps(concepts, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "benchmark_version": "v1",
        "kind": "constructed (Wikipedia interlanguage links); not real user relevance judgements",
        "seed_concepts": len(seeds),
        "concepts_kept": len(concepts),
        "dropped": sorted(dropped, key=lambda d: d["seed"]),
        "queries": len(queries),
        "qrels": len(qrels),
        "query_types": ["title", "lead_sentence (parenthetical asides removed)"],
        "query_language_rotation": list(SUPPORTED_LANGUAGES),
        "queries_whose_text_equals_a_target_title": sum(
            q["text_equals_target_title"] == "true" for q in queries
        ),
        "queries_whose_text_contains_a_target_title": sum(
            q["text_contains_target_title"] == "true" for q in queries
        ),
        "source": "MediaWiki API on en/es/fr.wikipedia.org (https only, allow-listed hosts)",
        "files_sha256": {
            n: _sha(out_dir / n) for n in ("queries.csv", "qrels.csv", "concepts.json")
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fold(language: str, title: str) -> str:
    """Punctuation-insensitive key: the legacy cleaner stripped characters such as en-dashes
    from titles ("1959-60 Scottish Cup" became "195960 Scottish Cup")."""
    from app.ingestion.text import clean_text

    return language + ":" + re.sub(r"\W", "", clean_text(title).casefold())


def corpus_manifest(export: dict, root: Path) -> dict:
    """Describe the canonical corpus and reconcile it with the legacy processed CSV files.

    ``root`` is the repository root. If the legacy files are absent (for example on a fresh
    clone) the reconciliation section records that instead of guessing.
    """
    import pandas as pd

    from app.ingestion.text import clean_text

    manifest: dict = {**export, "built_from": "active rows of the documents table"}
    raw_dir, processed_dir = root / "data" / "raw", root / "data" / "processed"
    raw: dict[str, list[dict]] = {}
    files = {}
    for language in SUPPORTED_LANGUAGES:
        path = raw_dir / f"wikipedia_{language}.json"
        if path.exists():
            raw[language] = json.loads(path.read_text(encoding="utf-8"))
            files[path.name] = {"sha256": _sha(path), "articles": len(raw[language])}
    manifest["legacy_raw_files"] = files or "not present"

    parts = [processed_dir / "all_train.csv", processed_dir / "all_test.csv"]
    corpus_path = root / "artifacts" / "corpus_v1" / "corpus.jsonl"
    if not all(p.exists() for p in parts) or not corpus_path.exists():
        manifest["legacy_reconciliation"] = "processed CSV files or corpus export not present"
        return manifest
    frame = pd.concat([pd.read_csv(p, usecols=["title", "language"]) for p in parts])
    legacy = [_fold(r.language, str(r.title)) for r in frame.itertuples()]
    legacy_unique = set(legacy)
    in_corpus = {
        _fold(row["language"], row["title"])
        for row in map(json.loads, corpus_path.read_text(encoding="utf-8").splitlines())
    }
    absent = sorted(legacy_unique - in_corpus)

    reasons = {k: "unknown" for k in absent}
    for language, articles in raw.items():
        for art in articles:
            key = _fold(language, art["title"])
            if key not in reasons:
                continue
            length = len(clean_text(art["content"]))
            if not clean_text(art["summary"]):
                reasons[key] = "empty_summary"
            elif length > LEGACY_MAX_CONTENT:
                reasons[key] = "over_content_length_bound"
            elif length < LEGACY_MIN_CONTENT:
                reasons[key] = "under_content_length_bound"
    by_reason: dict[str, int] = {}
    for reason in reasons.values():
        by_reason[reason] = by_reason.get(reason, 0) + 1
    manifest["legacy_reconciliation"] = {
        "processed_rows_train_plus_test": len(legacy),
        "unique_language_title_pairs_ignoring_punctuation": len(legacy_unique),
        "duplicate_rows_collapsed": len(legacy) - len(legacy_unique),
        "present_in_canonical_corpus": len(legacy_unique & in_corpus),
        "absent_from_canonical_corpus": len(absent),
        "absent_by_reason": dict(sorted(by_reason.items())),
        "absent_titles": absent,
        "added_beyond_legacy": len(in_corpus - legacy_unique),
        "matching": "titles compared ignoring case and punctuation (legacy cleaning removed it)",
    }
    return manifest
