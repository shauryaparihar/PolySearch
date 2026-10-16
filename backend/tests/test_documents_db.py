import pytest
from app.db.models import ContentSource, Document
from sqlalchemy.exc import IntegrityError

SHA = "a" * 64


def _source(session):
    src = ContentSource(
        name="wikipedia",
        base_url="https://www.wikipedia.org/",
        licence_name="CC BY-SA 4.0",
        licence_url="https://example.org/licence",
        attribution_text="attr",
    )
    session.add(src)
    session.flush()
    return src


def _doc(source_id, **kw):
    base = dict(
        source_id=source_id,
        external_id="Mathematics",
        language="en",
        title="Mathematics",
        summary="s",
        source_url="https://en.wikipedia.org/wiki/Mathematics",
        content_sha256=SHA,
    )
    return Document(**{**base, **kw})


def test_same_source_external_id_and_language_is_unique(session_factory):
    with session_factory() as s:
        src = _source(s)
        s.add(_doc(src.id))
        s.flush()
        s.add(_doc(src.id))
        with pytest.raises(IntegrityError):
            s.flush()


def test_same_title_in_another_language_is_allowed(session_factory):
    with session_factory() as s:
        src = _source(s)
        s.add_all([_doc(src.id), _doc(src.id, language="fr")])
        s.flush()


def test_unsupported_language_is_rejected(session_factory):
    with session_factory() as s:
        src = _source(s)
        s.add(_doc(src.id, language="de"))
        with pytest.raises(IntegrityError):
            s.flush()


def test_malformed_content_hash_is_rejected(session_factory):
    with session_factory() as s:
        src = _source(s)
        s.add(_doc(src.id, content_sha256="short"))
        with pytest.raises(IntegrityError):
            s.flush()
