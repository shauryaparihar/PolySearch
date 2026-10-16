"""Search artifacts, retrieval modes and fusion.

macOS OpenMP guard. ``faiss`` and ``torch`` each bundle their own OpenMP runtime. On macOS,
once FAISS has run in a process, the next transformer call segfaults (measured here: 0 of 6
runs survived with default threads, 8 of 8 with ``OMP_NUM_THREADS=1``). Two things are
therefore fixed in this file, before either library is imported:

* ``torch`` is imported first;
* on macOS, OpenMP is limited to one thread (``setdefault``: an explicit environment value
  still wins, at the caller's risk). Linux, where Docker deployment runs, is unaffected.

The cost is single-threaded encoding on macOS; query encoding is one short text, so it is
small. The latency report records ``torch_threads`` so the number is never ambiguous.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch  # noqa: E402,F401  (must precede any faiss import)
except ImportError:  # lexical-only tooling can run without torch
    pass
