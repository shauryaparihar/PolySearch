"""Make torch, faiss and scikit-learn share one OpenMP runtime (macOS only).

    python backend/scripts/macos_unify_openmp.py            # apply
    python backend/scripts/macos_unify_openmp.py --check    # exit 1 if runtimes are split
    python backend/scripts/macos_unify_openmp.py --revert   # restore the original files

Run it with the project's virtualenv after every ``pip install`` that touches faiss, torch or
scikit-learn. See app/search/openmp.py for why this is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search import openmp  # noqa: E402  (importing the package applies the runtime policy)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("not macOS: wheels share the system libgomp, nothing to do")
        return 0
    runtimes = openmp.find_runtimes()
    for name, rt in runtimes.items():
        print(f"  {name:8s} {rt.real if rt.real else 'no libomp found'}")
    if args.check:
        unified = openmp.is_unified(runtimes)
        print(
            "single shared runtime"
            if unified
            else "SPLIT runtimes (OMP_NUM_THREADS=1 fallback in use)"
        )
        return 0 if unified else 1
    if args.revert:
        print("restored:", openmp.revert(runtimes) or "nothing to restore")
        return 0
    changes = openmp.unify(runtimes)
    print("\n".join(changes) if changes else "already unified")
    ok = openmp.is_unified(openmp.find_runtimes())
    print("single shared runtime" if ok else "WARNING: runtimes are still split")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
