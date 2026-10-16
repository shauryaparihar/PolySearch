"""One OpenMP runtime for torch, faiss and scikit-learn on macOS.

Root cause of the macOS crash. Each of the three pip wheels ships its *own* copy of
``libomp.dylib`` (three different builds in this project's environment). Two OpenMP runtimes
in one process corrupt each other: once FAISS has used its runtime, the next transformer call
segfaults. Linux wheels share the system ``libgomp`` and are not affected.

The fix is to make the wheels load a single file. ``unify`` replaces the copies inside the
faiss and scikit-learn packages with a relative symlink to the copy inside torch (all three
report the same OpenMP ABI, compatibility version 5.0.0). ``revert`` restores the originals;
reinstalling a package also restores its original, which ``is_unified`` then detects.

This module is deliberately light: it imports no numeric libraries, so it can run before any
of them are loaded.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

LIB = "libomp.dylib"
BACKUP_SUFFIX = ".orig"
# package -> path of its libomp relative to the package directory
LOCATIONS = {
    "torch": Path("lib") / LIB,
    "faiss": Path(".dylibs") / LIB,
    "sklearn": Path(".dylibs") / LIB,
}
OWNER = "torch"  # every other package is pointed at torch's copy


@dataclass(frozen=True)
class Runtime:
    package: str
    path: Path | None  # the libomp file for this package, None if the package has none

    @property
    def real(self) -> Path | None:
        return Path(os.path.realpath(self.path)) if self.path and self.path.exists() else None


def package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)  # locates without importing the package
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def find_runtimes(dirs: dict[str, Path | None] | None = None) -> dict[str, Runtime]:
    """libomp location for every known package (``dirs`` lets tests supply fake packages)."""
    dirs = dirs if dirs is not None else {n: package_dir(n) for n in LOCATIONS}
    out = {}
    for name, rel in LOCATIONS.items():
        base = dirs.get(name)
        candidate = base / rel if base else None
        out[name] = Runtime(name, candidate if candidate and candidate.exists() else None)
    return out


def is_unified(runtimes: dict[str, Runtime]) -> bool:
    """True when every installed package resolves to the same libomp file."""
    present = [r.real for r in runtimes.values() if r.path is not None]
    if not present or any(p is None for p in present):
        return False
    return len(set(present)) == 1


def needs_cap(platform: str, runtimes: dict[str, Runtime]) -> bool:
    """macOS with more than one distinct runtime needs the single-thread fallback."""
    if platform != "darwin":
        return False
    distinct = {r.real for r in runtimes.values() if r.path is not None}
    return len(distinct) > 1


def unify(runtimes: dict[str, Runtime]) -> list[str]:
    """Point faiss/sklearn at torch's libomp. Returns a log of what changed (idempotent)."""
    owner = runtimes[OWNER]
    if owner.path is None:
        raise RuntimeError("torch's libomp.dylib was not found; nothing to unify onto")
    changed = []
    for name, rt in runtimes.items():
        if name == OWNER or rt.path is None:
            continue
        if rt.real == owner.real:
            continue
        backup = rt.path.with_name(rt.path.name + BACKUP_SUFFIX)
        if not backup.exists():
            os.replace(rt.path, backup)  # keep the original so revert is exact
        else:
            rt.path.unlink()
        rt.path.symlink_to(os.path.relpath(owner.path, rt.path.parent))
        changed.append(f"{name}: {rt.path.name} -> {owner.path}")
    return changed


def revert(runtimes: dict[str, Runtime]) -> list[str]:
    restored = []
    for name, rt in runtimes.items():
        if name == OWNER or rt.path is None:
            continue
        backup = rt.path.with_name(rt.path.name + BACKUP_SUFFIX)
        if backup.exists():
            if rt.path.is_symlink() or rt.path.exists():
                rt.path.unlink()
            os.replace(backup, rt.path)
            restored.append(name)
    return restored


def apply_runtime_policy(platform: str = sys.platform) -> str:
    """Run once, before torch/faiss are imported. Returns 'unified', 'capped' or 'n/a'.

    * Linux/other: nothing to do.
    * macOS with one shared runtime: full OpenMP parallelism.
    * macOS with split runtimes: limit OpenMP to one thread (the measured fallback; an explicit
      ``OMP_NUM_THREADS`` in the environment still wins).
    """
    if platform != "darwin":
        return "n/a"
    runtimes = find_runtimes()
    if not needs_cap(platform, runtimes):
        return "unified"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return "capped"
