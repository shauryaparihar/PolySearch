"""Startup passage sync and pruning of orphaned passage artifacts."""

import importlib.util
from pathlib import Path

from app.rag import lifecycle


def _dirs(root: Path, indexes: list[str], passages: list[str]) -> None:
    for name in indexes:
        (root / name).mkdir(parents=True)
    for name in passages:
        (root / "passages" / name).mkdir(parents=True)
        (root / "passages" / name / "manifest.json").write_text("{}")


def test_prune_lists_only_orphans_and_deletes_nothing_by_default(tmp_path):
    _dirs(tmp_path, ["idx-live"], ["idx-live", "idx-gone", "idx-old"])
    assert lifecycle.prune_passage_artifacts(tmp_path, set()) == ["idx-gone", "idx-old"]
    assert (tmp_path / "passages/idx-gone").exists()  # dry run


def test_prune_delete_removes_orphans_and_keeps_live_and_named(tmp_path):
    _dirs(tmp_path, ["idx-live"], ["idx-live", "idx-gone", "idx-keepme"])
    removed = lifecycle.prune_passage_artifacts(tmp_path, {"idx-keepme"}, delete=True)
    assert removed == ["idx-gone"]
    assert (tmp_path / "passages/idx-live").exists()
    assert (tmp_path / "passages/idx-keepme").exists()
    assert not (tmp_path / "passages/idx-gone").exists()


def test_prune_without_a_passages_folder_is_a_no_op(tmp_path):
    assert lifecycle.prune_passage_artifacts(tmp_path, set(), delete=True) == []


def test_prune_cli_prints_and_deletes_only_with_flag(tmp_path, capsys):
    _dirs(tmp_path, [], ["idx-gone"])
    spec = importlib.util.spec_from_file_location(
        "prune_passages", Path(__file__).resolve().parents[1] / "scripts/prune_passages.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--artifact-root", str(tmp_path)]) == 0
    assert "would delete: passages/idx-gone" in capsys.readouterr().out
    assert (tmp_path / "passages/idx-gone").exists()
    cli.main(["--artifact-root", str(tmp_path), "--delete"])
    assert not (tmp_path / "passages/idx-gone").exists()


class _BrokenFactory:
    def __call__(self):
        raise RuntimeError("database down")


class _Store:
    index_version = "idx-x"


def test_startup_sync_never_raises_when_the_database_is_down():
    assert lifecycle.sync_if_missing(_BrokenFactory(), _Store()) == "failed"
