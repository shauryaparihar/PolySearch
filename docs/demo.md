# Demo script and seed queries

Real output from the benchmark index (`dense` mode, top 5, cross-language). Misses are kept in
on purpose: the engine is good at concept matching and still returns noise after the first hits.

| Query (language) | Targets | Top results |
|---|---|---|
| `climate change` (en) | es, fr | 1 *Changement climatique* (fr) · 2 *Calentamiento global* (es) · 3 *Grenelle de l'environnement* (fr) |
| `cambio climático` (es) | en, fr | 1 *Changement climatique* (fr) · 2 *Climate change* (en) · 3 *Grenelle de l'environnement* (fr) |
| `agujero negro` (es) | en, fr | 1 *Black hole* (en) · 2 *Trou noir* (fr) · 3 *Trou de ver* (fr) · 5 *The Black Keys* (en) |
| `révolution industrielle` (fr) | en, es | 1 *Revolución Industrial* (es) · 2 *Industrial Revolution* (en) |
| `vaccine` (en) | es, fr | 1 *Vacuna* (es) · 2 *Vaccin* (fr) · 3 *OTs-38 Stechkin* (es, **noise**) |
| `how plants turn sunlight into energy` (en) | es, fr | 1 *Énergie solaire* · 2 *Energía solar* · 3 *Photosynthèse* · 4 *Fotosíntesis* (a descriptive query finds the topic, not always the best article first) |
| `Renaissance` (en, `hybrid`) | es, fr | 1 *Renaissance* (fr) |

## Five-minute walkthrough
1. Start the stack and open <http://localhost:8080> ([`containers.md`](containers.md)).
2. Type `agu`: autocomplete suggests article titles. Search `agujero negro` with targets
   English and French: the English *Black hole* and French *Trou noir* come back though no
   translation happened; point out the rank 5 miss.
3. Switch the mode to `hybrid` and search `Renaissance`: each result shows its meaning rank and
   keyword rank, so the fusion is visible.
4. Click "Load more" (paging reads a stored result pool, so pages never reshuffle), open a
   result (it opens the Wikipedia article and records the click), and mark one result relevant.
5. Show `GET /readyz` (every readiness check), then stop PostgreSQL: `/healthz` stays 200,
   `/readyz` turns 503 immediately and search answers a clean `503`; start it again and it
   recovers without a restart.
6. Admin only: `GET /api/v1/admin/analytics` with `X-Admin-Key` shows searches, cache hit
   rate, CTR and feedback; `POST /api/v1/admin/index-jobs` rebuilds and atomically swaps the index.
