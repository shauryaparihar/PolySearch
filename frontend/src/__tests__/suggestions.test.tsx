import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { SUGGEST_DELAY_MS } from "../hooks/useSuggestions";
import { clientWith, json, searchResponse } from "../testing/fixtures";

const SUGGESTIONS = {
  prefix: "vol",
  suggestions: [
    { text: "Volcán", languages: ["es", "fr"] },
    { text: "Volcano", languages: ["en"] },
  ],
};

function setup(extra: Record<string, Parameters<typeof clientWith>[0][string]> = {}) {
  const harness = clientWith({
    "GET /api/v1/suggestions": () => json(SUGGESTIONS),
    "POST /api/v1/search": () => json(searchResponse()),
    ...extra,
  });
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<App client={harness.client} />);
  return { ...harness, user, input: screen.getByRole("combobox", { name: "Search query" }) as HTMLInputElement };
}

const suggestionCalls = (fetchImpl: ReturnType<typeof clientWith>["fetchImpl"]) =>
  fetchImpl.mock.calls.filter(([url]) => String(url).includes("/suggestions"));

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => vi.useRealTimers());

describe("autocomplete", () => {
  it("waits for typing to pause (debounce) and then makes a single request", async () => {
    const { user, input, fetchImpl } = setup();
    await user.type(input, "volc");
    expect(suggestionCalls(fetchImpl)).toHaveLength(0); // nothing yet: still within the delay
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await waitFor(() => expect(suggestionCalls(fetchImpl)).toHaveLength(1));
    expect(String(suggestionCalls(fetchImpl)[0][0])).toContain("prefix=volc");
    expect(await screen.findByRole("listbox")).toBeInTheDocument();
  });

  it("does not request suggestions for a single character", async () => {
    const { user, input, fetchImpl } = setup();
    await user.type(input, "v");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS * 2));
    expect(suggestionCalls(fetchImpl)).toHaveLength(0);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("cancels the previous request when the prefix changes, so stale answers are ignored", async () => {
    const signals: AbortSignal[] = [];
    const { user, input } = setup({
      "GET /api/v1/suggestions": (_url, init) => {
        signals.push(init.signal as AbortSignal);
        return new Promise<Response>(() => undefined); // never answers
      },
    });
    await user.type(input, "vo");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await waitFor(() => expect(signals).toHaveLength(1));
    await user.type(input, "l");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await waitFor(() => expect(signals).toHaveLength(2));
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it("selecting a suggestion with the mouse fills the input and searches", async () => {
    const { user, input, fetchImpl } = setup();
    await user.type(input, "vol");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await user.click(await screen.findByRole("option", { name: /Volcán/ }));
    expect(input.value).toBe("Volcán");
    await waitFor(() => expect(screen.getByRole("region", { name: "Search results" })).toBeInTheDocument());
    const search = fetchImpl.mock.calls.find(([url]) => String(url).endsWith("/search"))!;
    expect(JSON.parse(String(search[1]!.body)).query).toBe("Volcán");
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("supports the keyboard: arrows move, Enter selects, Escape closes", async () => {
    const { user, input, fetchImpl } = setup();
    await user.type(input, "vol");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await screen.findByRole("listbox");
    await user.keyboard("{ArrowDown}{ArrowDown}");
    expect(screen.getByRole("option", { name: /Volcano/ })).toHaveAttribute("aria-selected", "true");
    expect(input).toHaveAttribute("aria-activedescendant", expect.stringContaining("opt-1"));
    await user.keyboard("{ArrowUp}{Enter}");
    expect(input.value).toBe("Volcán");
    await waitFor(() => expect(fetchImpl.mock.calls.some(([u]) => String(u).endsWith("/search"))).toBe(true));
  });

  it("Escape closes the list without searching", async () => {
    const { user, input, fetchImpl } = setup();
    await user.type(input, "vol");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    await screen.findByRole("listbox");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(fetchImpl.mock.calls.some(([u]) => String(u).endsWith("/search"))).toBe(false);
  });

  it("fails quietly when suggestions are unavailable", async () => {
    const { user, input } = setup({ "GET /api/v1/suggestions": () => new Response("down", { status: 503 }) });
    await user.type(input, "vol");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("renders suggestion text as plain text", async () => {
    const hostile = { prefix: "x", suggestions: [{ text: "<img src=x onerror=alert(1)>", languages: ["en"] }] };
    const { user, input } = setup({ "GET /api/v1/suggestions": () => json(hostile) });
    await user.type(input, "xx");
    await act(async () => vi.advanceTimersByTime(SUGGEST_DELAY_MS + 10));
    const list = await screen.findByRole("listbox");
    expect(list.querySelector("img")).toBeNull();
    expect(list).toHaveTextContent("<img src=x onerror=alert(1)>");
  });
});
