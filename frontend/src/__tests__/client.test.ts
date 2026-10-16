import { describe, expect, it } from "vitest";
import { ApiError, describeError } from "../api/errors";
import { createClient } from "../api/client";
import { apiError, bodyOf, clientWith, json, result, searchResponse } from "../testing/fixtures";

const params = { query: "volcano", queryLanguage: "en" as const, targetLanguages: ["es" as const, "fr" as const], mode: null, pageSize: 10 };

describe("API response parsing", () => {
  it("parses a valid search response", async () => {
    const { client } = clientWith({ "POST /api/v1/search": () => json(searchResponse()) });
    const response = await client.search(params);
    expect(response.results).toHaveLength(3);
    expect(response.results[0]).toMatchObject({ rank: 1, language: "es", source: "wikipedia" });
    expect(response.attribution.wikipedia.licence_name).toMatch(/Creative Commons/);
    expect(response.cache_hit).toBe(false);
  });

  it.each([
    ["results is not a list", { ...searchResponse(), results: "nope" }],
    ["a result is missing its title", { ...searchResponse(), results: [{ ...result(1), title: undefined }] }],
    ["unknown language", { ...searchResponse(), results: [result(1, { language: "xx" as never })] }],
    ["unknown mode", { ...searchResponse(), mode: "magic" }],
    ["cache_hit is not a boolean", { ...searchResponse(), cache_hit: "yes" }],
    ["a number is not finite", { ...searchResponse(), latency_ms: "fast" }],
    ["not an object", [1, 2, 3]],
  ])("rejects a malformed response: %s", async (_name, body) => {
    const { client } = clientWith({ "POST /api/v1/search": () => json(body) });
    await expect(client.search(params)).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });

  it("rejects a body that is not JSON", async () => {
    const { client } = clientWith({ "POST /api/v1/search": () => new Response("<html>oops</html>", { status: 200 }) });
    await expect(client.search(params)).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });

  it("parses suggestions and rejects bad ones", async () => {
    const good = clientWith({ "GET /api/v1/suggestions": () => json({ prefix: "vo", suggestions: [{ text: "Volcán", languages: ["es", "fr"] }] }) });
    expect(await good.client.suggestions("vo")).toEqual([{ text: "Volcán", languages: ["es", "fr"] }]);
    const bad = clientWith({ "GET /api/v1/suggestions": () => json({ suggestions: [{ text: 5 }] }) });
    await expect(bad.client.suggestions("vo")).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });
});

describe("requests", () => {
  it("sends the search body the API expects and omits mode when it is the default", async () => {
    const { client, fetchImpl } = clientWith({ "POST /api/v1/search": () => json(searchResponse()) });
    await client.search(params);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(String(url)).toBe("http://api.test/api/v1/search");
    expect(bodyOf(init!)).toEqual({
      query: "volcano", query_language: "en", target_languages: ["es", "fr"], sources: ["wikipedia"], page_size: 10,
    });
    await client.search({ ...params, mode: "hybrid" });
    expect(bodyOf(fetchImpl.mock.calls[1][1]!).mode).toBe("hybrid");
  });

  it("never sends an admin header or calls an admin endpoint", async () => {
    const { client, fetchImpl } = clientWith({
      "POST /api/v1/search": () => json(searchResponse()),
      "GET /api/v1/suggestions": () => json({ prefix: "vo", suggestions: [] }),
    });
    await client.search(params);
    await client.suggestions("vo");
    for (const [url, init] of fetchImpl.mock.calls) {
      expect(String(url)).not.toContain("/admin");
      expect(JSON.stringify(init?.headers ?? {}).toLowerCase()).not.toContain("admin");
    }
  });

  it("builds URLs correctly from a base URL with trailing slashes", async () => {
    const seen: string[] = [];
    const client = createClient({
      baseUrl: "http://api.test///",
      fetchImpl: (async (url: RequestInfo | URL) => {
        seen.push(String(url));
        return json({ prefix: "vo", suggestions: [] });
      }) as unknown as typeof fetch,
    });
    await client.suggestions("vo");
    expect(seen).toEqual(["http://api.test/api/v1/suggestions?prefix=vo&limit=8"]);
  });
});

describe("error handling", () => {
  it("turns the API error envelope into an ApiError with the retry delay", async () => {
    const { client } = clientWith({ "POST /api/v1/search": () => apiError(429, "RATE_LIMITED", { "retry-after": "17" }) });
    const error = await client.search(params).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ code: "RATE_LIMITED", status: 429, retryAfterSeconds: 17, requestId: "req-1" });
  });

  it("maps a network failure to NETWORK", async () => {
    const client = createClient({ baseUrl: "http://api.test", fetchImpl: (async () => { throw new TypeError("Failed to fetch"); }) as unknown as typeof fetch });
    await expect(client.search(params)).rejects.toMatchObject({ code: "NETWORK" });
  });

  it("maps an aborted request to ABORTED", async () => {
    const client = createClient({ baseUrl: "http://api.test", fetchImpl: (async () => { throw new DOMException("x", "AbortError"); }) as unknown as typeof fetch });
    await expect(client.suggestions("vo")).rejects.toMatchObject({ code: "ABORTED" });
  });

  it("handles a non-JSON error page", async () => {
    const { client } = clientWith({ "POST /api/v1/search": () => new Response("Bad gateway", { status: 502 }) });
    await expect(client.search(params)).rejects.toMatchObject({ code: "HTTP_502", status: 502 });
  });

  it("describes errors without leaking server text", () => {
    const messages = [
      describeError(new ApiError("RATE_LIMITED", "server text", 429, 3)).message,
      describeError(new ApiError("QUERY_INVALID", "server text", 422)).message,
      describeError(new ApiError("NETWORK", "server text")).message,
      describeError(new ApiError("BUSY", "server text", 503)).message,
      describeError(new ApiError("SEARCH_NOT_FOUND", "server text", 404)).message,
      describeError(new Error("secret internal detail")).message,
    ];
    for (const m of messages) {
      expect(m).not.toMatch(/server text|secret/);
    }
    expect(messages[0]).toContain("3 seconds");
  });
});
