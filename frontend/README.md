# PolySearch frontend

A deliberately thin React + TypeScript + Vite client. All search logic (ranking, filtering,
pagination, caching, rate limits) lives in the backend; this app only collects input, calls the
public API and renders what comes back as text.

```bash
cd frontend
cp .env.example .env.local        # VITE_API_URL=http://localhost:8000
npm install
npm run dev                       # http://localhost:5173 (the API allows this exact origin)
npm test                          # Vitest + React Testing Library
npm run build                     # type-check + production build
```

It never contains, sends or links to anything admin-related: no admin key, no `/admin/`
endpoints. Titles and snippets are rendered through React's text interpolation only.

## Docker

`frontend/Dockerfile` builds the static site and serves it with unprivileged nginx on port 8080.
`docker compose up -d --build` (repository root) builds it with `VITE_API_URL` pointing at the
local API; see `docs/containers.md`.
