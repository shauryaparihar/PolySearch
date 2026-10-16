#!/usr/bin/env bash
# Boots the Compose stack from empty volumes and checks what must be true WITHOUT an index or the
# model download: the images build, migrations apply, the API is alive but honestly "not ready",
# search fails safely, and the web UI is served. Used by CI and runnable locally:
#
#   scripts/stack_smoke.sh
#
# It uses its own Compose project and ports, so it can run next to a development stack.
set -euo pipefail
cd "$(dirname "$0")/.."

export POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-5435}"
export API_HOST_PORT="${API_HOST_PORT:-8001}"
export FRONTEND_HOST_PORT="${FRONTEND_HOST_PORT:-8081}"
PROJECT="${COMPOSE_PROJECT:-polysearch_smoke}"
dc() { docker compose -p "$PROJECT" "$@"; }

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- smoke check failed; recent API logs:"
    dc logs --no-color --tail 40 api || true
  fi
  dc down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
API="http://127.0.0.1:${API_HOST_PORT}"
WEB="http://127.0.0.1:${FRONTEND_HOST_PORT}"

dc config -q
dc build
dc up -d

echo "waiting for the API to answer /healthz ..."
for _ in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "$API/healthz" || true)" = 200 ] && break
  sleep 2
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "$API/healthz")" = 200 ] || fail "/healthz is not 200"

ready_status=$(curl -s -o /tmp/readyz.json -w '%{http_code}' "$API/readyz")
[ "$ready_status" = 503 ] || fail "/readyz should be 503 without an index, got $ready_status"
grep -q '"database":true' /tmp/readyz.json || fail "database check should pass"
grep -q '"schema":true' /tmp/readyz.json || fail "migrations should have been applied"
grep -q '"active_index":false' /tmp/readyz.json || fail "active_index should be false"

search_status=$(curl -s -o /tmp/search.json -w '%{http_code}' -X POST "$API/api/v1/search" \
  -H 'content-type: application/json' \
  -d '{"query":"physics","query_language":"en","target_languages":["es"]}')
[ "$search_status" = 503 ] || fail "search without an index should be 503, got $search_status"
grep -q INDEX_UNAVAILABLE /tmp/search.json || fail "expected the INDEX_UNAVAILABLE error code"
grep -qi traceback /tmp/search.json && fail "an error response leaked a traceback"

[ "$(curl -s -o /tmp/web.html -w '%{http_code}' "$WEB/")" = 200 ] || fail "web UI is not served"
grep -q 'id="root"' /tmp/web.html || fail "web UI HTML is not the PolySearch app"

echo "stack smoke check passed"
