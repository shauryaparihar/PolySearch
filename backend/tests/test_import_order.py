import subprocess
import sys
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])


def test_torch_is_imported_before_faiss_by_the_search_package():
    """faiss imported before torch segfaults transformer inference on macOS (OpenMP clash)."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import app.search.artifacts\n"
        "mods = list(sys.modules)\n"
        "assert 'faiss' in mods\n"
        "assert 'torch' in mods, 'torch missing'\n"
        "assert mods.index('torch') < mods.index('faiss'), 'torch imported after faiss'\n"
    ) % BACKEND
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


def test_package_policy_matches_the_environment_and_explicit_settings_win():
    code = (
        "import sys, os; sys.path.insert(0, %r)\n"
        "import app.search\n"
        "print(app.search.OPENMP_POLICY, os.environ.get('OMP_NUM_THREADS'))\n"
    ) % BACKEND
    env = {k: v for k, v in __import__("os").environ.items() if k != "OMP_NUM_THREADS"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-500:]
    policy, threads = out.stdout.strip().splitlines()[-1].split()
    if sys.platform == "darwin":
        # capped only while the wheels still ship separate runtimes; unified -> full parallelism
        assert (policy, threads) in {("capped", "1"), ("unified", "None")}
    else:
        assert (policy, threads) == ("n/a", "None")  # Linux: untouched
    env["OMP_NUM_THREADS"] = "4"  # an explicit choice is respected
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip().splitlines()[-1].endswith(" 4")
