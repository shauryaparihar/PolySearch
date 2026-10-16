import os

import pytest
from app.search import openmp


def _fake_site(tmp_path, with_sklearn=True):
    """Three fake packages, each with its own *different* libomp copy (the real situation)."""
    dirs = {}
    for name, content in (("torch", b"torch-omp"), ("faiss", b"faiss-omp"), ("sklearn", b"sk-omp")):
        if name == "sklearn" and not with_sklearn:
            dirs[name] = None
            continue
        base = tmp_path / name
        lib = base / openmp.LOCATIONS[name]
        lib.parent.mkdir(parents=True)
        lib.write_bytes(content)
        dirs[name] = base
    return dirs


def test_separate_copies_are_detected_as_split(tmp_path):
    runtimes = openmp.find_runtimes(_fake_site(tmp_path))
    assert openmp.is_unified(runtimes) is False
    assert openmp.needs_cap("darwin", runtimes) is True
    assert openmp.needs_cap("linux", runtimes) is False  # Linux shares libgomp: never capped


def test_unify_points_other_packages_at_torch_and_keeps_backups(tmp_path):
    dirs = _fake_site(tmp_path)
    changed = openmp.unify(openmp.find_runtimes(dirs))
    assert len(changed) == 2
    runtimes = openmp.find_runtimes(dirs)
    assert openmp.is_unified(runtimes) and openmp.needs_cap("darwin", runtimes) is False
    faiss_lib = dirs["faiss"] / ".dylibs" / "libomp.dylib"
    assert faiss_lib.is_symlink() and not os.path.isabs(os.readlink(faiss_lib))  # relative link
    assert faiss_lib.read_bytes() == b"torch-omp"
    assert (dirs["faiss"] / ".dylibs" / "libomp.dylib.orig").read_bytes() == b"faiss-omp"


def test_unify_is_idempotent(tmp_path):
    dirs = _fake_site(tmp_path)
    openmp.unify(openmp.find_runtimes(dirs))
    assert openmp.unify(openmp.find_runtimes(dirs)) == []  # nothing left to change
    backup = dirs["sklearn"] / ".dylibs" / "libomp.dylib.orig"
    assert backup.read_bytes() == b"sk-omp"  # the original was not overwritten by a link


def test_revert_restores_the_original_bytes(tmp_path):
    dirs = _fake_site(tmp_path)
    openmp.unify(openmp.find_runtimes(dirs))
    assert sorted(openmp.revert(openmp.find_runtimes(dirs))) == ["faiss", "sklearn"]
    assert (dirs["faiss"] / ".dylibs" / "libomp.dylib").read_bytes() == b"faiss-omp"
    assert openmp.is_unified(openmp.find_runtimes(dirs)) is False


def test_reinstalling_a_package_is_detected_and_can_be_fixed_again(tmp_path):
    dirs = _fake_site(tmp_path)
    openmp.unify(openmp.find_runtimes(dirs))
    lib = dirs["faiss"] / ".dylibs" / "libomp.dylib"
    lib.unlink()  # what `pip install --force-reinstall faiss-cpu` effectively does
    lib.write_bytes(b"faiss-omp-new-build")
    assert openmp.is_unified(openmp.find_runtimes(dirs)) is False
    openmp.unify(openmp.find_runtimes(dirs))
    assert openmp.is_unified(openmp.find_runtimes(dirs)) is True


def test_packages_without_a_libomp_are_ignored(tmp_path):
    dirs = _fake_site(tmp_path, with_sklearn=False)
    runtimes = openmp.find_runtimes(dirs)
    assert runtimes["sklearn"].path is None
    openmp.unify(runtimes)
    assert openmp.is_unified(openmp.find_runtimes(dirs))


def test_unify_refuses_without_torchs_runtime(tmp_path):
    dirs = _fake_site(tmp_path)
    dirs["torch"] = None
    with pytest.raises(RuntimeError, match="torch"):
        openmp.unify(openmp.find_runtimes(dirs))


def test_policy_is_a_no_op_on_linux(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert openmp.apply_runtime_policy("linux") == "n/a"
    assert "OMP_NUM_THREADS" not in os.environ


def test_split_runtimes_on_macos_fall_back_to_one_thread(monkeypatch, tmp_path):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    split = openmp.find_runtimes(_fake_site(tmp_path))
    monkeypatch.setattr(openmp, "find_runtimes", lambda: split)
    assert openmp.apply_runtime_policy("darwin") == "capped"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_a_unified_runtime_on_macos_keeps_full_parallelism(monkeypatch, tmp_path):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    dirs = _fake_site(tmp_path)
    openmp.unify(openmp.find_runtimes(dirs))
    unified = openmp.find_runtimes(dirs)
    monkeypatch.setattr(openmp, "find_runtimes", lambda: unified)
    assert openmp.apply_runtime_policy("darwin") == "unified"
    assert "OMP_NUM_THREADS" not in os.environ


def test_an_explicit_thread_setting_is_never_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    split = openmp.find_runtimes(_fake_site(tmp_path))
    monkeypatch.setattr(openmp, "find_runtimes", lambda: split)
    openmp.apply_runtime_policy("darwin")
    assert os.environ["OMP_NUM_THREADS"] == "4"
