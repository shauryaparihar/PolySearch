"""The threshold calibration tool: its arithmetic, and a retrieval-only run end to end."""

from __future__ import annotations

import importlib.util
import json
from functools import partial
from pathlib import Path

import pytest
from api_support import build_bundle
from app.evaluation.corpus import Corpus
from app.rag.passage_store import build_passage_artifact
from app.rag.passages import DEFAULT_CONFIG
from app.search.bundle import SearchBundle
from fake_encoder import FakeEncoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_evidence_threshold.py"


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("calibrate_tool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_is_one_for_perfect_separation_and_half_for_none(tool):
    assert tool.auc([0.8, 0.9], [0.1, 0.2]) == 1.0
    assert tool.auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert tool.auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_sweep_reports_both_error_rates_at_each_threshold(tool):
    rows = {r["threshold"]: r for r in tool.sweep([0.6, 0.8], [0.3, 0.7], lo=0.5, hi=0.9, step=0.1)}
    assert rows[0.5]["false_abstain_rate"] == 0.0 and rows[0.5]["false_answer_rate"] == 0.5
    assert (
        rows[0.7]["false_abstain_rate"] == 0.5
    )  # a score equal to the threshold is answered and rows[0.7]["false_answer_rate"] == 0.5
    assert rows[0.9]["false_abstain_rate"] == 1.0 and rows[0.9]["false_answer_rate"] == 0.0
    assert all(0 <= r["balanced_accuracy"] <= 1 for r in rows.values())


def test_a_run_needs_both_kinds_of_question(tool, tmp_path, monkeypatch, capsys):
    from api_support import docs

    bundle = build_bundle(tmp_path)
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"question": "volcano", "answerable": True}) + "\n")
    monkeypatch.setattr(
        tool.SearchBundle, "load", partial(SearchBundle.load, encoder=FakeEncoder())
    )
    build_passage_artifact(
        Corpus(docs(), "d" * 64), FakeEncoder(), bundle.version,
        bundle.manifest["corpus"]["sha256"], bundle.path.parent, DEFAULT_CONFIG,
    )  # fmt: skip
    assert tool.main(["--index-dir", str(bundle.path), "--questions", str(questions)]) == 2
    assert "at least one answerable" in capsys.readouterr().err


def test_a_full_run_writes_a_report_without_any_model(tool, tmp_path, monkeypatch, capsys):
    from api_support import docs

    bundle = build_bundle(tmp_path)
    build_passage_artifact(
        Corpus(docs(), "d" * 64), FakeEncoder(), bundle.version,
        bundle.manifest["corpus"]["sha256"], bundle.path.parent, DEFAULT_CONFIG,
    )  # fmt: skip
    monkeypatch.setattr(
        tool.SearchBundle, "load", partial(SearchBundle.load, encoder=FakeEncoder())
    )
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"question": "Volcano", "answerable": True},
                {"question": "Glacier", "answerable": True},
                {"question": "zzzz qqqq xxxx", "answerable": False},
                {"question": "kkkk wwww", "answerable": False},
            ]
        )
    )
    out = tmp_path / "report.json"
    assert tool.main(["--index-dir", str(bundle.path), "--questions", str(questions),
                      "--out", str(out)]) == 0  # fmt: skip
    report = json.loads(out.read_text())
    assert report["summary"]["questions"] == 4 and len(report["questions"]) == 4
    assert report["summary"]["answerable"]["n"] == 2 and 0 <= report["summary"]["auc"] <= 1
    assert "balanced_accuracy" in report["summary"]["best_balanced_accuracy"]
    assert capsys.readouterr().err == ""
