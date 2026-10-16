# macOS: one OpenMP runtime for torch, faiss and scikit-learn

## Symptom
On macOS the search process crashed with a segmentation fault (exit 139) the first time the
transformer ran after FAISS had been used. Measured with the unguarded code: 0 of 3 runs
survived.

## Root cause
The macOS wheels of `torch`, `faiss-cpu` and `scikit-learn` each bundle their own
`libomp.dylib`. In this environment they are three different builds (different SHA-256),
all reporting OpenMP ABI compatibility version 5.0.0. Two or more OpenMP runtimes in one
process corrupt each other. Linux wheels link the system `libgomp` and are not affected, so
CI and the Docker image never see this.

## Fix
`backend/scripts/macos_unify_openmp.py` replaces the copies inside `faiss` and `sklearn`
with a relative symlink to the copy inside `torch`, so every library loads the same file.
The originals are kept as `libomp.dylib.orig`.

```bash
.venv/bin/python backend/scripts/macos_unify_openmp.py            # apply (idempotent)
.venv/bin/python backend/scripts/macos_unify_openmp.py --check    # exit 0 = single runtime
.venv/bin/python backend/scripts/macos_unify_openmp.py --revert   # restore the originals
```

Measured after applying it: unguarded, full threads, 8 of 8 runs survived; through the real
package 6 of 6.

Run it again after any `pip install` that reinstalls faiss, torch or scikit-learn (a
reinstall restores the original file).

## Fallback if the runtimes are still split
`import app.search` checks the environment before anything is loaded. With split runtimes on
macOS it sets `OMP_NUM_THREADS=1` (unless you set it yourself), which avoids the crash by
disabling OpenMP parallelism (8 of 8 runs survived). With a single shared runtime nothing is
restricted. `app.search.OPENMP_POLICY` reports `unified`, `capped` or `n/a` (not macOS).

## Verification and tests
`backend/tests/test_openmp.py` covers detection, unify, idempotence, revert, a reinstall
undoing the fix, and the policy decision (using fake packages, so it needs neither macOS nor
the real libraries). `test_import_order.py` checks that `torch` is imported before `faiss`.
