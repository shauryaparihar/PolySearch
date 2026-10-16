"""Search artifacts, retrieval modes and fusion.

macOS OpenMP guard (see ``app/search/openmp.py`` for the full explanation). The wheels of
torch, faiss and scikit-learn each bundle their own OpenMP runtime; two runtimes in one process
crash the transformer after FAISS has run. This file, which runs before any of them is imported:

* decides the runtime policy: with a single shared runtime (``scripts/macos_unify_openmp.py``)
  nothing is restricted; with split runtimes OpenMP is limited to one thread as a fallback;
* imports ``torch`` first, so its runtime is the one that initialises first.

Linux, where Docker deployment runs, shares the system ``libgomp`` and is unaffected.
"""

from app.search import openmp

OPENMP_POLICY = openmp.apply_runtime_policy()

try:
    import torch  # noqa: E402,F401  (must precede any faiss import)
except ImportError:  # lexical-only tooling can run without torch
    pass
