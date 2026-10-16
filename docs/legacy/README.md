# Legacy material (isolated, not used by the search engine)

| Item | Status |
|---|---|
| `Cross_lingual_project_paper.pdf` | Original paper, unchanged. Contains unsupported figures (see `../evidence_audit.md`). To be revised after the project is complete. |
| `sentence-transformers_LaBSE_results.json` | Original all-zero result, unchanged. |
| `CHECKSUMS.sha256` | SHA-256 of the two files above. |
| `src/`, `web_app/`, `colab_encoding/` (repository root) | Original research scripts and Streamlit demo. They still contain the pickle-based LaBSE embedding loader. Nothing under `backend/` imports them (checked by a test that forbids `pickle`/`joblib` in `backend/app`). |
| `data/legacy/embeddings/*.pkl` (untracked) | The old LaBSE embeddings, moved out of `data/embeddings/` so no active path can find them. |
