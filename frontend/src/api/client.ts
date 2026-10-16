import { ABORTED, ApiError, BAD_RESPONSE, NETWORK } from "./errors";
import { parseClick, parseResultPage, parseSearchResponse, parseSuggestions } from "./parse";
import type {
  ClickResponse,
  FeedbackLabel,
  Language,
  ResultPage,
  SearchParams,
  SearchResponse,
  Suggestion,
} from "./types";

export interface ApiClient {
  search(params: SearchParams, signal?: AbortSignal): Promise<SearchResponse>;
  page(searchId: string, cursor: string, limit: number, signal?: AbortSignal): Promise<ResultPage>;
  suggestions(prefix: string, signal?: AbortSignal): Promise<Suggestion[]>;
  /** Fire-and-forget safe: resolves/rejects but the UI never depends on it. */
  click(searchId: string, documentId: string, idempotencyKey: string): Promise<ClickResponse>;
  feedback(searchId: string, documentId: string, label: FeedbackLabel): Promise<void>;
}

export interface ClientOptions {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export const defaultBaseUrl = (): string =>
  (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Turn an error envelope ({error: {code, message, request_id}}) into an ApiError. */
async function toApiError(response: Response): Promise<ApiError> {
  const retryHeader = response.headers.get("retry-after");
  const retry = retryHeader && /^\d+$/.test(retryHeader) ? Number(retryHeader) : null;
  try {
    const body = await response.json();
    const err = body?.error;
    if (err && typeof err.code === "string") {
      return new ApiError(
        err.code,
        typeof err.message === "string" ? err.message : "Request failed",
        response.status,
        retry,
        typeof err.request_id === "string" ? err.request_id : null,
      );
    }
  } catch {
    /* not JSON: fall through */
  }
  return new ApiError(`HTTP_${response.status}`, "Request failed", response.status, retry);
}

export function createClient(options: ClientOptions = {}): ApiClient {
  const baseUrl = (options.baseUrl ?? defaultBaseUrl()).replace(/\/+$/, "");
  const doFetch = options.fetchImpl ?? ((...args: Parameters<typeof fetch>) => fetch(...args));

  async function request(path: string, init: RequestInit = {}): Promise<unknown> {
    let response: Response;
    try {
      response = await doFetch(`${baseUrl}${path}`, {
        ...init,
        headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}) },
      });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        throw new ApiError(ABORTED, "Request cancelled");
      }
      throw new ApiError(NETWORK, "Could not reach the search service");
    }
    if (!response.ok) throw await toApiError(response);
    if (response.status === 204) return null;
    try {
      return await response.json();
    } catch {
      throw new ApiError(BAD_RESPONSE, "Unexpected response: not JSON");
    }
  }

  return {
    async search(params, signal) {
      const body: Record<string, unknown> = {
        query: params.query,
        query_language: params.queryLanguage,
        target_languages: params.targetLanguages,
        sources: ["wikipedia"],
        page_size: params.pageSize,
      };
      if (params.mode) body.mode = params.mode; // otherwise let the index's default decide
      return parseSearchResponse(await request("/api/v1/search", { method: "POST", body: JSON.stringify(body), signal }));
    },
    async page(searchId, cursor, limit, signal) {
      const qs = new URLSearchParams({ cursor, limit: String(limit) });
      return parseResultPage(await request(`/api/v1/searches/${encodeURIComponent(searchId)}/results?${qs}`, { signal }));
    },
    async suggestions(prefix, signal) {
      const qs = new URLSearchParams({ prefix, limit: "8" });
      return parseSuggestions(await request(`/api/v1/suggestions?${qs}`, { signal }));
    },
    async click(searchId, documentId, idempotencyKey) {
      return parseClick(
        await request(`/api/v1/searches/${encodeURIComponent(searchId)}/clicks`, {
          method: "POST",
          body: JSON.stringify({ document_id: documentId, idempotency_key: idempotencyKey }),
          keepalive: true, // lets the request finish even while the browser navigates away
        }),
      );
    },
    async feedback(searchId, documentId, label) {
      await request(`/api/v1/searches/${encodeURIComponent(searchId)}/feedback/${encodeURIComponent(documentId)}`, {
        method: "PUT",
        body: JSON.stringify({ label }),
      });
    },
  };
}

export type { Language };
