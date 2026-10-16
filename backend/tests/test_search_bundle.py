import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from app.ingestion.text import dense_text, lexical_text
from app.search import lexical_index
from app.search.artifacts import ArtifactError, build_index
from app.search.bundle import BundleRetriever, SearchBundle
from app.search.encoder import normalise
from fake_encoder import FakeEncoder
from search_fixtures import DOCS, corpus
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    return build_index(corpus(), FakeEncoder(), tmp_path_factory.mktemp("index"))


@pytest.fixture
def bundle(built) -> SearchBundle:
    return SearchBundle.load(built, encoder=FakeEncoder())


def _copy(built, tmp_path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(built, target)
    return target


def _edit_manifest(path: Path, fn):
    manifest = json.loads((path / "manifest.json").read_text())
    fn(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))


def _resign(path: Path, name: str):
    _edit_manifest(
        path,
        lambda m: m["files"].__setitem__(
            name, hashlib.sha256((path / name).read_bytes()).hexdigest()
        ),
    )


# ---------------------------------------------------------------- embeddings
def test_embeddings_are_unit_length_finite_float32():
    vectors = normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert vectors.dtype == np.float32 and np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    with pytest.raises(ValueError, match="NaN"):
        normalise(np.array([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="zero vector"):
        normalise(np.zeros((1, 3)))


def test_stored_vectors_are_normalised(bundle):
    index = bundle._indexes["es"]
    norms = np.linalg.norm(np.vstack([index.reconstruct(i) for i in range(index.ntotal)]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ------------------------------------------------------------------- mapping
def test_vector_ids_map_to_the_right_documents(bundle):
    for doc in DOCS:
        top = bundle.search(dense_text(doc.title, doc.summary), "dense", [doc.language], k=1)
        assert top.hits[0].key == doc.key  # a document's own text finds itself, not a neighbour


def test_building_twice_gives_the_same_version_and_mapping(built, tmp_path):
    again = build_index(corpus(), FakeEncoder(), tmp_path)
    assert again.name == built.name
    assert (again / "mapping_es.json").read_bytes() == (built / "mapping_es.json").read_bytes()
    # FAISS id i is position i in corpus order, so ids are stable for a given corpus
    assert json.loads((built / "mapping_es.json").read_text()) == [
        "es:Volcán",
        "es:Glaciar",
        "es:Jazz",
    ]


# ----------------------------------------------------------------- filtering
@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_results_are_limited_to_the_requested_language_partitions(bundle, mode):
    only_es = bundle.search("volcano lava", mode, ["es"], k=5)
    assert {h.key.split(":")[0] for h in only_es.hits} == {"es"}
    es_fr = bundle.search("volcano lava", mode, ["fr", "es"], k=9)
    assert {h.key.split(":")[0] for h in es_fr.hits} <= {"es", "fr"}


def test_cross_language_query_finds_the_linked_documents(bundle):
    hits = [h.key for h in bundle.search("Volcano", "hybrid", ["es", "fr"], k=2).hits]
    assert set(hits) == {"es:Volcán", "fr:Volcan"}


def test_hybrid_reports_the_component_ranks(bundle):
    hit = bundle.search("volcano lava", "hybrid", ["es", "fr"], k=3).hits[0]
    assert hit.dense_rank is not None and hit.lexical_rank is not None
    assert hit.score == pytest.approx(1 / (60 + hit.dense_rank) + 1 / (60 + hit.lexical_rank))


def test_invalid_search_parameters_are_rejected(bundle):
    for kwargs in (
        {"mode": "magic", "target_languages": ["es"]},
        {"mode": "dense", "target_languages": []},
        {"mode": "dense", "target_languages": ["de"]},
        {"mode": "dense", "target_languages": ["es"], "k": 0},
        {"mode": "dense", "target_languages": ["es"], "k": 101},
    ):
        with pytest.raises(ValueError):
            bundle.search("x", **kwargs)


def test_bundle_retriever_matches_the_evaluation_interface(bundle):
    retriever = BundleRetriever(bundle, "hybrid")
    assert retriever.name == "hybrid-rrf" and retriever.config["rrf_k"] == 60
    assert retriever.search("Volcano", ["es"], 1) == [
        (bundle.search("Volcano", "hybrid", ["es"], 1).hits[0].key, pytest.approx(1 / 61 + 1 / 61))
    ]


# ------------------------------------------------------------------- lexical
def test_lexical_index_equals_scikit_learn_tfidf_and_survives_save_load(tmp_path):
    texts = [lexical_text(d.title, d.summary) for d in DOCS]
    keys, langs = [d.key for d in DOCS], [d.language for d in DOCS]
    mine = lexical_index.LexicalIndex.build(texts, keys, langs)
    sk = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=False)
    matrix = sk.fit_transform(texts)
    query = lexical_text("volcano lava", "")
    expected = np.asarray((matrix @ sk.transform([query]).T).todense()).ravel()
    assert np.allclose(mine.score(query), expected, atol=1e-12)
    mine.save(tmp_path)
    loaded = lexical_index.LexicalIndex.load(tmp_path)
    assert np.allclose(loaded.score(query), expected, atol=1e-12)


# ------------------------------------------------------ tampering / mismatches
def test_untouched_artifacts_load(built):
    assert SearchBundle.load(built, load_encoder=False).version == built.name


def test_a_changed_file_hash_is_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    data = bytearray((path / "faiss_es.index").read_bytes())
    data[100] ^= 0xFF
    (path / "faiss_es.index").write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="checksum mismatch for faiss_es.index"):
        SearchBundle.load(path, load_encoder=False)


def test_a_wrong_dimension_is_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    _edit_manifest(path, lambda m: m["encoder"].__setitem__("dimension", 65))
    with pytest.raises(ArtifactError, match="dimension"):
        SearchBundle.load(path, load_encoder=False)


def test_a_wrong_document_count_is_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    _edit_manifest(path, lambda m: m["partitions"]["fr"].__setitem__("documents", 99))
    with pytest.raises(ArtifactError, match="fr document count"):
        SearchBundle.load(path, load_encoder=False)


def test_a_missing_file_is_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    (path / "mapping_en.json").unlink()
    with pytest.raises(ArtifactError, match="missing artifact file"):
        SearchBundle.load(path, load_encoder=False)


def test_a_re_signed_but_inconsistent_mapping_is_still_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    keys = json.loads((path / "mapping_es.json").read_text())
    keys[0] = "fr:Intruso"  # wrong language partition
    (path / "mapping_es.json").write_text(json.dumps(keys))
    _resign(path, "mapping_es.json")
    with pytest.raises(ArtifactError, match="duplicate or foreign keys"):
        SearchBundle.load(path, load_encoder=False)


def test_corrupt_or_unknown_manifests_are_refused(built, tmp_path):
    path = _copy(built, tmp_path)
    _edit_manifest(path, lambda m: m.__setitem__("format", 99))
    with pytest.raises(ArtifactError, match="unsupported artifact format"):
        SearchBundle.load(path, load_encoder=False)
    (path / "manifest.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="manifest unreadable"):
        SearchBundle.load(path, load_encoder=False)


def test_a_different_encoder_is_refused(built):
    class Other(FakeEncoder):
        revision = "rev-2"

    with pytest.raises(ArtifactError, match="not the one the artifacts were built with"):
        SearchBundle.load(built, encoder=Other())


def test_dense_search_without_an_encoder_is_refused_clearly(built):
    lexical_only = SearchBundle.load(built, load_encoder=False)
    assert lexical_only.search("volcano", "lexical", ["es"], 1).hits
    with pytest.raises(ArtifactError, match="without an encoder"):
        lexical_only.search("volcano", "dense", ["es"], 1)


def test_encoder_is_warmed_up_once_at_load(built):
    encoder = FakeEncoder()
    SearchBundle.load(built, encoder=encoder)
    assert encoder.calls == 1


# ----------------------------------------------------------- build behaviour
def test_a_failed_build_leaves_no_partial_artifacts(tmp_path):
    class Exploding(FakeEncoder):
        def encode(self, texts):
            raise RuntimeError("encoder died")

    with pytest.raises(RuntimeError):
        build_index(corpus(), Exploding(), tmp_path / "index")
    assert list((tmp_path / "index").iterdir()) == []  # neither a version nor a staging dir


# -------------------------------------------------------------- no pickle
def test_active_code_never_loads_pickles():
    backend = Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in backend.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import pickle" in text or "joblib" in text or "allow_pickle=True" in text:
            offenders.append(str(path.relative_to(backend)))
    assert offenders == []


def test_numpy_files_in_artifacts_load_without_pickle(built):
    np.load(built / "lexical_idf.npy", allow_pickle=False)  # raises if the file needs pickle
