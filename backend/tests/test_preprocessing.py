import pytest

from src.preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor(tmp_path):
    return DataPreprocessor(input_dir=tmp_path / "raw", output_dir=tmp_path / "out")


def _article(**overrides):
    article = {
        "title": "Torre   Eiffel",
        "summary": "La Torre Eiffel es una torre de hierro.",
        "content": "La torre " * 100,
        "language": "es",
        "url": "https://es.wikipedia.org/wiki/Torre_Eiffel",
        "categories": ["Francia", "Monumentos"],
    }
    article.update(overrides)
    return article


def test_valid_article_is_cleaned_and_kept(preprocessor):
    df = preprocessor.process_articles([_article()])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["title"] == "Torre Eiffel"  # whitespace collapsed
    assert row["language"] == "es"
    assert row["doc_id"] == "es_0"


def test_too_short_article_is_rejected(preprocessor):
    df = preprocessor.process_articles([_article(content="demasiado corto")])
    assert len(df) == 0


def test_missing_required_field_fails_loudly(preprocessor):
    bad = _article()
    del bad["content"]
    with pytest.raises(KeyError):
        preprocessor.process_articles([bad])
