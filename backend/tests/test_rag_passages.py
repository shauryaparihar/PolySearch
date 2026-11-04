"""Deterministic chunking: exact slices, bounded size, stable ids, and awkward text."""

from __future__ import annotations

import pytest
from app.rag.passages import (
    CHARS_PER_TOKEN,
    ChunkConfig,
    build_passages,
    chunk_spans,
    content_sha256,
    estimate_tokens,
    passage_id,
    sentence_spans,
)

SHORT = ChunkConfig(max_chars=100, overlap_max_chars=60)
TEXT = (
    "Dr. John Smith (born 1970) is an American actor. He starred in many films. "
    "In 2001 he won an award. He lives in Texas. His brother J. R. Smith is a painter."
)


def build(text, config=SHORT, version="idx-a", key="en:Doc"):
    return build_passages(version, key, "en", "Doc", text, config)


def test_every_passage_is_an_exact_slice_of_the_source():
    passages = build(TEXT)
    assert len(passages) >= 2
    for p in passages:
        assert TEXT[p.char_start : p.char_end] == p.text
        assert p.content_sha256 == content_sha256(p.text)


def test_no_passage_exceeds_the_limit_and_ordinals_are_sequential():
    passages = build(TEXT * 5)
    assert all(len(p.text) <= SHORT.max_chars for p in passages)
    assert [p.ordinal for p in passages] == list(range(len(passages)))


def test_abbreviations_and_initials_do_not_end_a_sentence():
    spans = sentence_spans(TEXT)
    sentences = [TEXT[a:b] for a, b in spans]
    assert sentences[0].startswith("Dr. John Smith") and sentences[0].endswith("actor.")
    assert any("J. R. Smith" in s for s in sentences)
    assert len(sentences) == 5


def test_decimals_and_urls_are_not_split():
    text = "Pi is about 3.14159 in value. See https://example.org/a.b for more. Done."
    assert [text[a:b] for a, b in sentence_spans(text)] == [
        "Pi is about 3.14159 in value.",
        "See https://example.org/a.b for more.",
        "Done.",
    ]


def test_every_sentence_appears_whole_in_some_passage():
    passages = build(TEXT)
    for a, b in sentence_spans(TEXT):
        assert any(p.char_start <= a and b <= p.char_end for p in passages), TEXT[a:b]


def test_a_short_last_sentence_is_repeated_as_overlap():
    passages = build(TEXT)
    assert passages[0].char_end > passages[1].char_start  # the windows overlap
    # the repeated part is a whole sentence
    overlap = TEXT[passages[1].char_start : passages[0].char_end]
    assert overlap.endswith(".") and overlap in TEXT


def test_a_long_last_sentence_is_not_repeated():
    long_sentence = "Esta es una frase muy larga " + "que sigue y sigue " * 5 + "hasta el final."
    text = f"Primera frase corta. {long_sentence} Otra frase final."
    config = ChunkConfig(max_chars=140, overlap_max_chars=40)
    passages = build(text, config)
    for earlier, later in zip(passages, passages[1:], strict=False):
        assert later.char_start >= earlier.char_end  # nothing is repeated


def test_an_oversized_sentence_is_cut_at_word_boundaries():
    sentence = ("palabra " * 80).strip() + "."
    passages = build(sentence, ChunkConfig(max_chars=100))
    assert len(passages) > 1
    assert all(len(p.text) <= 100 for p in passages)
    assert all(not p.text.startswith(" ") and not p.text.endswith(" ") for p in passages)
    assert "".join(p.text.replace(" ", "") for p in passages) == sentence.replace(" ", "")


def test_a_single_unbroken_token_longer_than_the_limit_is_still_bounded():
    passages = build("x" * 450, ChunkConfig(max_chars=100))
    assert [len(p.text) for p in passages] == [100, 100, 100, 100, 50]


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_empty_or_blank_text_gives_no_passages(text):
    assert build(text) == []


def test_unicode_text_keeps_offsets_and_survives():
    text = (
        "¿Qué es un volcán? Es una estructura geológica. «Le volcan» est actif. "
        "Café résumé naïve — emoji 🌋 aussi. Ñandú 東京 に 行く。 Combining: école."
    )
    passages = build(text, ChunkConfig(max_chars=60))
    assert passages
    for p in passages:
        assert text[p.char_start : p.char_end] == p.text
        assert p.text == p.text.strip()
    assert any("🌋" in p.text for p in passages)


def test_the_same_input_always_gives_the_same_passages_and_ids():
    first, second = build(TEXT), build(TEXT)
    assert first == second
    assert [p.passage_id for p in first] == [p.passage_id for p in second]


def test_ids_change_with_version_document_position_or_content():
    base = passage_id("idx-a", "en:Doc", 0, "a" * 64)
    assert base == passage_id("idx-a", "en:Doc", 0, "a" * 64)
    assert base != passage_id("idx-b", "en:Doc", 0, "a" * 64)
    assert base != passage_id("idx-a", "en:Other", 0, "a" * 64)
    assert base != passage_id("idx-a", "en:Doc", 1, "a" * 64)
    assert base != passage_id("idx-a", "en:Doc", 0, "b" * 64)
    assert len({p.passage_id for p in build(TEXT)}) == len(build(TEXT))


def test_editing_one_sentence_changes_only_the_affected_passage_ids():
    before = {p.ordinal: p.passage_id for p in build(TEXT)}
    after = {p.ordinal: p.passage_id for p in build(TEXT.replace("Texas", "Ohio"))}
    assert before[0] == after[0] and before[max(before)] != after[max(after)]


def test_token_estimates_are_conservative():
    assert estimate_tokens("") == 0 and estimate_tokens("abc") == 1 and estimate_tokens("abcd") == 2
    for p in build(TEXT):
        assert p.estimated_tokens >= len(p.text.split())  # never below the word count
        assert p.estimated_tokens * CHARS_PER_TOKEN >= len(p.text)


def test_the_token_limit_maps_to_a_character_limit():
    assert ChunkConfig.from_max_tokens(200).max_chars == 200 * CHARS_PER_TOKEN
    spans = chunk_spans(TEXT * 8, ChunkConfig.from_max_tokens(40))
    assert all(b - a <= 120 for a, b in spans)


def test_progress_is_always_made_on_adversarial_input():
    for text in ("a. " * 500, "A. B. C. D. " * 200, "!" * 300, ". " * 300, "Hi there. " * 300):
        spans = chunk_spans(text, ChunkConfig(max_chars=50, overlap_max_chars=200))
        assert spans and all(b > a for a, b in spans)
        assert [a for a, _ in spans] == sorted(set(a for a, _ in spans))  # strictly advancing
