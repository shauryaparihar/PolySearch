import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "../App";
import { apiError, bodyOf, clientWith, json, result, searchResponse } from "../testing/fixtures";

type Routes = Parameters<typeof clientWith>[0];

function setup(routes: Routes = {}) {
  const harness = clientWith({
    "GET /api/v1/suggestions": () => json({ prefix: "", suggestions: [] }),
    "POST /api/v1/search": () => json(searchResponse()),
    ...routes,
  });
  const user = userEvent.setup();
  render(<App client={harness.client} />);
  return { ...harness, user };
}

const box = () => screen.getByRole("combobox", { name: "Search query" });
const searchCalls = (f: ReturnType<typeof clientWith>["fetchImpl"]) =>
  f.mock.calls.filter(([u, i]) => String(u).endsWith("/search") && i?.method === "POST");

async function search(user: ReturnType<typeof userEvent.setup>, text = "volcano") {
  await user.clear(box());
  await user.type(box(), text);
  await user.click(screen.getByRole("button", { name: "Search" }));
}

describe("search flow", () => {
  it("shows ranked results with title, language, source, summary, link and measured metadata", async () => {
    const { user } = setup();
    await search(user);
    const cards = await screen.findAllByTestId("result");
    expect(cards).toHaveLength(3);
    const first = within(cards[0]);
    expect(first.getByRole("heading", { name: "Título 1" })).toBeInTheDocument();
    expect(first.getByText("Español")).toBeInTheDocument();
    expect(first.getAllByText("Wikipedia").length).toBeGreaterThan(0);
    expect(first.getByText("Resumen del documento 1.")).toBeInTheDocument();
    expect(first.getByRole("link", { name: /Open on Wikipedia/ })).toHaveAttribute("href", "https://es.wikipedia.org/wiki/Titulo_1");
    const meta = screen.getByRole("status");
    expect(meta).toHaveTextContent("3 results");
    expect(meta).toHaveTextContent("dense mode");
    expect(meta).toHaveTextContent("server 12 ms");
    expect(meta).toHaveTextContent(/round trip \d+ ms/);
    expect(meta).toHaveTextContent("cache miss");
    expect(meta).toHaveTextContent("idx-test");
    expect(meta.textContent).not.toMatch(/accura|precision|best|quality score/i); // no quality claims
  });

  it("shows the source and licence attribution with a link", async () => {
    const { user } = setup();
    await search(user);
    const note = await screen.findByText(/Text is available under a Creative Commons licence/);
    expect(note).toBeVisible();
    expect(note.textContent).not.toMatch(/Source:.*Source:/); // the API text is shown once, not repeated
    const licence = within(note.closest("p")!).getByRole("link", { name: /Creative Commons Attribution-Share Alike 4.0/ });
    expect(licence).toHaveAttribute("href", "https://creativecommons.org/licenses/by-sa/4.0/deed.en");
    expect(licence).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("shows a cache hit and the component ranks", async () => {
    const response = searchResponse({ cache_hit: true, mode: "hybrid", results: [result(1, { dense_rank: 2, lexical_rank: 1 })], total_results: 1 });
    const { user } = setup({ "POST /api/v1/search": () => json(response) });
    await search(user);
    expect(await screen.findByRole("status")).toHaveTextContent("cache hit");
    expect(screen.getByText("meaning #2 · keywords #1")).toBeInTheDocument();
  });

  it("explains an empty result set", async () => {
    const { user } = setup({ "POST /api/v1/search": () => json(searchResponse({ results: [], total_results: 0 })) });
    await search(user);
    expect(await screen.findByText(/No results/)).toBeInTheDocument();
    expect(screen.queryAllByTestId("result")).toHaveLength(0);
  });

  it("shows a loading state while waiting", async () => {
    let release!: (r: Response) => void;
    const { user } = setup({ "POST /api/v1/search": () => new Promise<Response>((resolve) => (release = resolve)) });
    await search(user);
    expect(await screen.findByText("Searching…", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Searching…" })).toHaveAttribute("aria-busy", "true");
    release(json(searchResponse()));
    expect(await screen.findAllByTestId("result")).toHaveLength(3);
  });

  it("a newer search cancels and replaces an older one", async () => {
    const signals: AbortSignal[] = [];
    let n = 0;
    const { user } = setup({
      "POST /api/v1/search": (_u, init) => {
        signals.push(init.signal as AbortSignal);
        return ++n === 1 ? new Promise<Response>(() => undefined) : Promise.resolve(json(searchResponse({ results: [result(7)], total_results: 1 })));
      },
    });
    await search(user, "first");
    await waitFor(() => expect(signals).toHaveLength(1));
    // the first search is still loading; the user refines the query and submits again
    await user.clear(box());
    await user.type(box(), "second{Enter}");
    expect(await screen.findByRole("heading", { name: "Título 7" })).toBeInTheDocument();
    expect(signals[0].aborted).toBe(true);
  });
});

describe("filters and validation", () => {
  it("sends the chosen languages, mode and page size", async () => {
    const { user, fetchImpl } = setup();
    await user.selectOptions(screen.getByLabelText("Query language"), "es");
    // default targets follow the query language: the other two
    expect(screen.getByRole("checkbox", { name: "English" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Español" })).not.toBeChecked();
    await user.click(screen.getByRole("checkbox", { name: "Français" })); // untick
    await user.selectOptions(screen.getByLabelText("Search mode"), "hybrid");
    await user.selectOptions(screen.getByLabelText("Results per page"), "20");
    await search(user, "volcán");
    await screen.findAllByTestId("result");
    expect(bodyOf(searchCalls(fetchImpl)[0][1]!)).toEqual({
      query: "volcán", query_language: "es", target_languages: ["en"], sources: ["wikipedia"], mode: "hybrid", page_size: 20,
    });
  });

  it("omits the mode when the default is selected", async () => {
    const { user, fetchImpl } = setup();
    await search(user);
    await screen.findAllByTestId("result");
    expect(bodyOf(searchCalls(fetchImpl)[0][1]!)).not.toHaveProperty("mode");
  });

  it("keeps the user's language choices when the query language changes later", async () => {
    const { user } = setup();
    await user.click(screen.getByRole("checkbox", { name: "Français" })); // now touched
    await user.selectOptions(screen.getByLabelText("Query language"), "fr");
    expect(screen.getByRole("checkbox", { name: "Français" })).not.toBeChecked();
  });

  it("rejects a too-short query without calling the API", async () => {
    const { user, fetchImpl } = setup();
    await search(user, "a");
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter between 2 and 500 characters.");
    expect(searchCalls(fetchImpl)).toHaveLength(0);
  });

  it("rejects a too-long query without calling the API", async () => {
    const { user, fetchImpl } = setup();
    await user.click(box());
    await user.paste("x".repeat(520));
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("between 2 and 500");
    expect(searchCalls(fetchImpl)).toHaveLength(0);
  });

  it("requires at least one language", async () => {
    const { user, fetchImpl } = setup();
    await user.click(screen.getByRole("checkbox", { name: "Español" }));
    await user.click(screen.getByRole("checkbox", { name: "Français" }));
    await search(user);
    expect(await screen.findByRole("alert")).toHaveTextContent("at least one language");
    expect(searchCalls(fetchImpl)).toHaveLength(0);
  });

  it("collapses whitespace in the query it sends", async () => {
    const { user, fetchImpl } = setup();
    await search(user, "  volcano    lava  ");
    await screen.findAllByTestId("result");
    expect(bodyOf(searchCalls(fetchImpl)[0][1]!).query).toBe("volcano lava");
  });

  it("offers Wikipedia as the only source and does not let it be unchecked", () => {
    setup();
    expect(screen.getByRole("checkbox", { name: "Wikipedia" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Wikipedia" })).toBeDisabled();
  });

  it("is keyboard operable: Enter in the box submits", async () => {
    const { user, fetchImpl } = setup();
    await user.type(box(), "volcano{Enter}");
    await screen.findAllByTestId("result");
    expect(searchCalls(fetchImpl)).toHaveLength(1);
  });
});

describe("pagination", () => {
  const first = searchResponse({ results: [result(1), result(2)], next_cursor: "cursor-2", total_results: 4 });

  it("loads the next page with the cursor and appends it; the button disappears on the last page", async () => {
    const pages: URL[] = [];
    const { user } = setup({
      "POST /api/v1/search": () => json(first),
      "GET /api/v1/searches/:id/results": (url) => {
        pages.push(url);
        return json({ search_id: first.search_id, results: [result(3), result(4)], next_cursor: null, total_results: 4, attribution: first.attribution });
      },
    });
    await search(user);
    await screen.findAllByTestId("result");
    await user.click(screen.getByRole("button", { name: "Load more results" }));
    await waitFor(() => expect(screen.getAllByTestId("result")).toHaveLength(4));
    expect(pages[0].searchParams.get("cursor")).toBe("cursor-2");
    expect(pages[0].searchParams.get("limit")).toBe("10");
    expect(screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent)).toEqual(["Título 1", "Título 2", "Título 3", "Título 4"]);
    expect(screen.queryByRole("button", { name: "Load more results" })).toBeNull();
  });

  it("explains an expired result pool and lets the user search again", async () => {
    let searches = 0;
    const { user } = setup({
      "POST /api/v1/search": () => (++searches, json(first)),
      "GET /api/v1/searches/:id/results": () => apiError(404, "SEARCH_NOT_FOUND"),
    });
    await search(user);
    await screen.findAllByTestId("result");
    await user.click(screen.getByRole("button", { name: "Load more results" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("These results have expired");
    expect(screen.getAllByTestId("result")).toHaveLength(2); // what was loaded stays visible
    await user.click(screen.getByRole("button", { name: "Search again" }));
    await waitFor(() => expect(searches).toBe(2));
  });

  it("shows a page error but keeps the results and the button to retry", async () => {
    const { user } = setup({
      "POST /api/v1/search": () => json(first),
      "GET /api/v1/searches/:id/results": () => apiError(429, "RATE_LIMITED", { "retry-after": "5" }),
    });
    await search(user);
    await screen.findAllByTestId("result");
    await user.click(screen.getByRole("button", { name: "Load more results" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("wait 5 seconds");
    expect(screen.getAllByTestId("result")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Load more results" })).toBeEnabled();
  });
});

describe("error states", () => {
  const cases: [string, () => Response, RegExp][] = [
    ["rate limit with a delay", () => apiError(429, "RATE_LIMITED", { "retry-after": "30" }), /wait 30 seconds/],
    ["rate limit without a delay", () => apiError(429, "RATE_LIMITED"), /Too many requests/],
    ["server-side query validation", () => apiError(422, "QUERY_INVALID"), /between 2 and 500/],
    ["server-side filter validation", () => apiError(422, "FILTER_INVALID"), /options are not valid/],
    ["busy", () => apiError(503, "BUSY", { "retry-after": "1" }), /busy/],
    ["no index", () => apiError(503, "INDEX_UNAVAILABLE"), /unavailable/],
    ["internal error", () => apiError(500, "INTERNAL_ERROR"), /unavailable/],
    ["a proxy error page", () => new Response("<html>502</html>", { status: 502 }), /unavailable/],
    ["malformed success body", () => json({ nonsense: true }), /unavailable/],
  ];

  it.each(cases)("shows a clear message for: %s", async (_name, make, expected) => {
    const { user } = setup({ "POST /api/v1/search": make });
    await search(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(expected);
    expect(alert.textContent).not.toContain("server text that must not be shown");
    expect(screen.queryAllByTestId("result")).toHaveLength(0);
  });

  it("shows 'unavailable' when the backend cannot be reached", async () => {
    const { user } = setup({
      "POST /api/v1/search": () => {
        throw new TypeError("Failed to fetch");
      },
    });
    await search(user);
    expect(await screen.findByRole("alert")).toHaveTextContent("Search is unavailable right now");
  });

  it("recovers: a successful search after an error clears the error", async () => {
    let fail = true;
    const { user } = setup({ "POST /api/v1/search": () => (fail ? apiError(503, "BUSY") : json(searchResponse())) });
    await search(user);
    await screen.findByRole("alert");
    fail = false;
    await user.click(screen.getByRole("button", { name: "Search" }));
    await screen.findAllByTestId("result");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
