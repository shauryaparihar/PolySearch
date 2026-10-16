from app.ingestion.text import bound_text, clean_text, content_sha256, dense_text, lexical_text


def test_content_hash_is_deterministic_and_ignores_cosmetic_differences():
    a = content_sha256("Torre  Eiffel", "La torre​ es de hierro.")
    b = content_sha256("Torre Eiffel", "La  torre es de hierro.")
    assert a == b
    assert len(a) == 64
    assert a != content_sha256("Torre Eiffel", "La torre es de acero.")


def test_clean_text_strips_markup_and_invisible_characters():
    assert clean_text("<b>Hello</b>&nbsp;&amp; ​world\x00") == "Hello & world"
    assert clean_text("a < b and c > d") == "a < b and c > d"  # not a tag


def test_dense_text_keeps_case_and_function_words():
    text = dense_text("The Rise of Rome", "It was the largest empire.")
    assert text == "The Rise of Rome. It was the largest empire."


def test_lexical_text_folds_case_and_accents_without_stopword_removal():
    assert (
        lexical_text("Educación", "La ÉDUCATION de la femme")
        == "educacion la education de la femme"
    )


def test_bound_text_cuts_at_word_boundary():
    assert bound_text("alpha beta gamma", 12) == "alpha beta"
    assert bound_text("short", 100) == "short"


def test_inline_tags_do_not_leave_stray_spaces_but_block_tags_separate_words():
    assert clean_text("<b>here</b>.") == "here."
    assert clean_text("one<br>two") == "one two"
