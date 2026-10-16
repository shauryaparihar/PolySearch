import { vi } from "vitest";
import { createClient } from "../api/client";
import type { ResultItem, SearchResponse } from "../api/types";

export const BASE = "http://api.test";

export const attribution = {
  wikipedia: {
    name: "wikipedia",
    licence_name: "Creative Commons Attribution-Share Alike 4.0",
    licence_url: "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
    attribution_text: "Text is available under a Creative Commons licence.",
  },
};

export function result(rank: number, over: Partial<ResultItem> = {}): ResultItem {
  return {
    rank,
    document_id: `00000000-0000-0000-0000-${String(rank).padStart(12, "0")}`,
    title: `Título ${rank}`,
    snippet: `Resumen del documento ${rank}.`,
    language: "es",
    source: "wikipedia",
    source_url: `https://es.wikipedia.org/wiki/Titulo_${rank}`,
    source_revision: "42",
    score: 0.5,
    dense_rank: rank,
    lexical_rank: null,
    ...over,
  };
}

export function searchResponse(over: Partial<SearchResponse> = {}): SearchResponse {
  return {
    search_id: "11111111-1111-1111-1111-111111111111",
    results: [result(1), result(2), result(3)],
    next_cursor: null,
    total_results: 3,
    attribution,
    index_version: "idx-test",
    mode: "dense",
    cache_hit: false,
    latency_ms: 12.4,
    ...over,
  };
}

export const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

export const apiError = (status: number, code: string, headers: Record<string, string> = {}) =>
  json({ error: { code, message: "server text that must not be shown", request_id: "req-1", details: {} } }, status, headers);

type Handler = (url: URL, init: RequestInit) => Response | Promise<Response>;

/** A fake fetch routed by "METHOD /path". Unrouted calls fail the test loudly. */
export function mockFetch(routes: Record<string, Handler>) {
  const fn = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input));
    const path = url.pathname
      .replace(/\/searches\/[^/]+\/feedback\/[^/]+/, "/searches/:id/feedback/:id")
      .replace(/\/searches\/[^/]+\/(results|clicks)/, "/searches/:id/$1");
    const key = `${(init.method ?? "GET").toUpperCase()} ${path}`;
    const handler = routes[key];
    if (!handler) throw new Error(`unrouted request: ${key}`);
    return handler(url, init);
  });
  return fn;
}

export function clientWith(routes: Record<string, Handler>) {
  const fetchImpl = mockFetch(routes);
  return { client: createClient({ baseUrl: BASE, fetchImpl: fetchImpl as unknown as typeof fetch }), fetchImpl };
}

export const bodyOf = (init: RequestInit) => JSON.parse(String(init.body));
