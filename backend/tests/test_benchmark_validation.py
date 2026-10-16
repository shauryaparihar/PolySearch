import pytest
from app.evaluation.benchmark import load_benchmark, validate_benchmark
from app.evidence import BenchmarkValidationError
from bench_helpers import qrel, query, write_benchmark

CORPUS = {"en:Cancer", "es:Cáncer", "fr:Cancer", "es:Diabetes", "fr:Diabète"}


def _good(tmp_path):
    write_benchmark(
        tmp_path,
        [query()],
        [qrel("c1-en-title", "es:Cáncer", "es"), qrel("c1-en-title", "fr:Cancer", "fr")],
        ["c1"],
    )
    return load_benchmark(tmp_path)


def test_valid_benchmark_passes_and_exposes_hashes(tmp_path):
    bench = _good(tmp_path)
    validate_benchmark(bench, CORPUS)
    assert bench.qrel_count == 2
    assert len(bench.queries_sha256) == 64 and len(bench.qrels_sha256) == 64


def test_duplicate_query_ids_are_rejected(tmp_path):
    write_benchmark(
        tmp_path,
        [query(), query()],
        [qrel("c1-en-title", "es:Cáncer", "es")],
        ["c1"],
    )
    with pytest.raises(BenchmarkValidationError, match="duplicate query ids"):
        validate_benchmark(load_benchmark(tmp_path), CORPUS)


def test_duplicate_concept_ids_are_rejected(tmp_path):
    write_benchmark(tmp_path, [query()], [qrel("c1-en-title", "es:Cáncer", "es")], ["c1", "c1"])
    with pytest.raises(BenchmarkValidationError, match="duplicate concept ids"):
        validate_benchmark(load_benchmark(tmp_path), CORPUS)


def test_relevant_document_in_the_query_language_is_rejected(tmp_path):
    write_benchmark(tmp_path, [query()], [qrel("c1-en-title", "en:Cancer", "en")], ["c1"])
    with pytest.raises(BenchmarkValidationError, match="own language"):
        validate_benchmark(load_benchmark(tmp_path), CORPUS)


def test_qrels_must_point_at_documents_in_the_corpus(tmp_path):
    write_benchmark(
        tmp_path,
        [query()],
        [qrel("c1-en-title", "es:Cáncer", "es"), qrel("c1-en-title", "fr:Absent", "fr")],
        ["c1"],
    )
    with pytest.raises(BenchmarkValidationError, match="missing from the corpus"):
        validate_benchmark(load_benchmark(tmp_path), CORPUS)


def test_query_without_relevant_documents_is_rejected(tmp_path):
    write_benchmark(
        tmp_path,
        [query(), query("c1-en-lead", text="Cancer is a disease.", qtype="lead_sentence")],
        [qrel("c1-en-title", "es:Cáncer", "es")],
        ["c1"],
    )
    with pytest.raises(BenchmarkValidationError, match="no relevant documents"):
        validate_benchmark(load_benchmark(tmp_path), CORPUS)


def test_unknown_concept_and_language_mismatch_are_rejected(tmp_path):
    write_benchmark(
        tmp_path,
        [query(concept="c9")],
        [qrel("c1-en-title", "es:Cáncer", "fr")],
        ["c1"],
    )
    with pytest.raises(BenchmarkValidationError) as exc:
        validate_benchmark(load_benchmark(tmp_path), CORPUS)
    assert "unknown concept" in str(exc.value) and "does not match" in str(exc.value)
