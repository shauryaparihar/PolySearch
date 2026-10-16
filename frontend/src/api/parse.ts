import { ApiError, BAD_RESPONSE } from "./errors";
import type {
  ClickResponse,
  Language,
  Mode,
  ResultItem,
  ResultPage,
  SearchResponse,
  SourceAttribution,
  Suggestion,
} from "./types";

/** Runtime checks: the UI never trusts the shape of a network response. */
const bad = (what: string): never => {
  throw new ApiError(BAD_RESPONSE, `Unexpected response: ${what}`);
};

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, what: string): string => (typeof v === "string" ? v : bad(what));
const num = (v: unknown, what: string): number =>
  typeof v === "number" && Number.isFinite(v) ? v : bad(what);
const optNum = (v: unknown, what: string): number | null =>
  v === null || v === undefined ? null : num(v, what);
const optStr = (v: unknown, what: string): string | null =>
  v === null || v === undefined ? null : str(v, what);
const lang = (v: unknown, what: string): Language =>
  v === "en" || v === "es" || v === "fr" ? v : bad(what);
const mode = (v: unknown): Mode =>
  v === "dense" || v === "lexical" || v === "hybrid" ? v : bad("mode");

function parseResult(raw: unknown): ResultItem {
  if (!isRecord(raw)) return bad("result");
  return {
    rank: num(raw.rank, "rank"),
    document_id: str(raw.document_id, "document_id"),
    title: str(raw.title, "title"),
    snippet: str(raw.snippet, "snippet"),
    language: lang(raw.language, "language"),
    source: str(raw.source, "source"),
    source_url: str(raw.source_url, "source_url"),
    source_revision: optStr(raw.source_revision, "source_revision"),
    score: num(raw.score, "score"),
    dense_rank: optNum(raw.dense_rank, "dense_rank"),
    lexical_rank: optNum(raw.lexical_rank, "lexical_rank"),
  };
}

function parseAttribution(raw: unknown): Record<string, SourceAttribution> {
  if (!isRecord(raw)) return bad("attribution");
  const out: Record<string, SourceAttribution> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (!isRecord(value)) return bad("attribution entry");
    out[key] = {
      name: str(value.name, "attribution.name"),
      licence_name: str(value.licence_name, "attribution.licence_name"),
      licence_url: str(value.licence_url, "attribution.licence_url"),
      attribution_text: str(value.attribution_text, "attribution.attribution_text"),
    };
  }
  return out;
}

export function parseResultPage(raw: unknown): ResultPage {
  if (!isRecord(raw) || !Array.isArray(raw.results)) return bad("result page");
  return {
    search_id: str(raw.search_id, "search_id"),
    results: raw.results.map(parseResult),
    next_cursor: optStr(raw.next_cursor, "next_cursor"),
    total_results: num(raw.total_results, "total_results"),
    attribution: parseAttribution(raw.attribution),
  };
}

export function parseSearchResponse(raw: unknown): SearchResponse {
  const page = parseResultPage(raw);
  const r = raw as Record<string, unknown>;
  if (typeof r.cache_hit !== "boolean") bad("cache_hit");
  return {
    ...page,
    index_version: str(r.index_version, "index_version"),
    mode: mode(r.mode),
    cache_hit: r.cache_hit as boolean,
    latency_ms: num(r.latency_ms, "latency_ms"),
  };
}

export function parseSuggestions(raw: unknown): Suggestion[] {
  if (!isRecord(raw) || !Array.isArray(raw.suggestions)) return bad("suggestions");
  return raw.suggestions.map((s) => {
    if (!isRecord(s) || !Array.isArray(s.languages)) return bad("suggestion");
    return { text: str(s.text, "suggestion.text"), languages: s.languages.map((l) => lang(l, "language")) };
  });
}

export function parseClick(raw: unknown): ClickResponse {
  if (!isRecord(raw) || typeof raw.duplicate !== "boolean") return bad("click");
  return {
    click_id: str(raw.click_id, "click_id"),
    rank: num(raw.rank, "rank"),
    duplicate: raw.duplicate,
  };
}
