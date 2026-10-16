import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "../App";
import { apiError, bodyOf, clientWith, json, result, searchResponse } from "../testing/fixtures";

type Routes = Parameters<typeof clientWith>[0];

async function open(routes: Routes = {}, response = searchResponse()) {
  const harness = clientWith({
    "GET /api/v1/suggestions": () => json({ prefix: "", suggestions: [] }),
    "POST /api/v1/search": () => json(response),
    "POST /api/v1/searches/:id/clicks": () => json({ click_id: "c1", rank: 1, duplicate: false }, 201),
    "PUT /api/v1/searches/:id/feedback/:id": () => json({ document_id: "x", label: "relevant", updated_at: "now" }),
    ...routes,
  });
  const user = userEvent.setup();
  render(<App client={harness.client} />);
  await user.type(screen.getByRole("combobox", { name: "Search query" }), "volcano{Enter}");
  await screen.findAllByTestId("result");
  return { ...harness, user };
}

const calls = (f: ReturnType<typeof clientWith>["fetchImpl"], suffix: string) =>
  f.mock.calls.filter(([u]) => String(u).includes(suffix));

describe("click tracking", () => {
  it("records the click with the document, an idempotency key and keepalive, then lets the link open", async () => {
    const { fetchImpl } = await open();
    const link = within(screen.getAllByTestId("result")[1]).getByRole("link", { name: /Open on Wikipedia/ });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    const notPrevented = fireEvent.click(link); // returns false if default navigation was prevented
    expect(notPrevented).toBe(true);
    await waitFor(() => expect(calls(fetchImpl, "/clicks")).toHaveLength(1));
    const [url, init] = calls(fetchImpl, "/clicks")[0];
    expect(String(url)).toContain("/searches/11111111-1111-1111-1111-111111111111/clicks");
    expect(init!.keepalive).toBe(true);
    const body = bodyOf(init!);
    expect(body.document_id).toBe(result(2).document_id);
    expect(body.idempotency_key).toMatch(/^click-[A-Za-z0-9_-]{8,}$/);
  });

  it("a failing click request never blocks opening the source or shows an error", async () => {
    const { fetchImpl } = await open({
      "POST /api/v1/searches/:id/clicks": () => {
        throw new TypeError("network down");
      },
    });
    const link = within(screen.getAllByTestId("result")[0]).getByRole("link", { name: /Open on Wikipedia/ });
    expect(fireEvent.click(link)).toBe(true); // navigation allowed
    await waitFor(() => expect(calls(fetchImpl, "/clicks")).toHaveLength(1));
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("a click answered with a server error is also silent", async () => {
    const { fetchImpl } = await open({ "POST /api/v1/searches/:id/clicks": () => apiError(404, "RESULT_NOT_FOUND") });
    fireEvent.click(within(screen.getAllByTestId("result")[0]).getByRole("link", { name: /Open on Wikipedia/ }));
    await waitFor(() => expect(calls(fetchImpl, "/clicks")).toHaveLength(1));
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("each click uses a fresh idempotency key", async () => {
    const { fetchImpl } = await open();
    const link = within(screen.getAllByTestId("result")[0]).getByRole("link", { name: /Open on Wikipedia/ });
    fireEvent.click(link);
    fireEvent.click(link);
    await waitFor(() => expect(calls(fetchImpl, "/clicks")).toHaveLength(2));
    const keys = calls(fetchImpl, "/clicks").map(([, i]) => bodyOf(i!).idempotency_key);
    expect(new Set(keys).size).toBe(2);
  });
});

describe("relevance feedback", () => {
  it("sends the label, marks the button pressed and lets the user change their mind", async () => {
    const { user, fetchImpl } = await open();
    const card = within(screen.getAllByTestId("result")[0]);
    const relevant = card.getByRole("button", { name: "Relevant" });
    const notRelevant = card.getByRole("button", { name: "Not relevant" });
    expect(relevant).toHaveAttribute("aria-pressed", "false");
    await user.click(relevant);
    await waitFor(() => expect(relevant).toHaveAttribute("aria-pressed", "true"));
    await user.click(notRelevant);
    await waitFor(() => expect(notRelevant).toHaveAttribute("aria-pressed", "true"));
    expect(relevant).toHaveAttribute("aria-pressed", "false");
    const sent = calls(fetchImpl, "/feedback/").map(([u, i]) => [String(u).split("/feedback/")[1], i!.method, bodyOf(i!).label]);
    expect(sent).toEqual([
      [result(1).document_id, "PUT", "relevant"],
      [result(1).document_id, "PUT", "not_relevant"],
    ]);
  });

  it("shows an error and reverts when saving fails", async () => {
    const { user } = await open({ "PUT /api/v1/searches/:id/feedback/:id": () => apiError(429, "RATE_LIMITED") });
    const card = within(screen.getAllByTestId("result")[0]);
    const relevant = card.getByRole("button", { name: "Relevant" });
    await user.click(relevant);
    expect(await card.findByRole("alert")).toHaveTextContent("Could not save your feedback");
    expect(relevant).toHaveAttribute("aria-pressed", "false"); // reverted
  });

  it("keeps feedback independent per result", async () => {
    const { user } = await open();
    await user.click(within(screen.getAllByTestId("result")[1]).getByRole("button", { name: "Relevant" }));
    expect(within(screen.getAllByTestId("result")[0]).getByRole("button", { name: "Relevant" })).toHaveAttribute("aria-pressed", "false");
    await waitFor(() => expect(within(screen.getAllByTestId("result")[1]).getByRole("button", { name: "Relevant" })).toHaveAttribute("aria-pressed", "true"));
  });
});

describe("safe rendering", () => {
  const hostile = searchResponse({
    results: [
      result(1, {
        title: "<img src=x onerror=alert(1)>Volcán",
        snippet: "<script>alert('xss')</script><b>bold</b> &lt;i&gt; text",
        language: "es",
      }),
      result(2, { title: "Unsafe link", source_url: "javascript:alert(1)" }),
      result(3, { title: "Data link", source_url: "data:text/html,<script>alert(1)</script>" }),
    ],
    total_results: 3,
  });

  it("renders titles and snippets as inert text, never as HTML", async () => {
    await open({}, hostile);
    const cards = screen.getAllByTestId("result");
    const first = cards[0];
    expect(first.querySelector("script, img, b, i")).toBeNull();
    expect(within(first).getByRole("heading", { level: 3 })).toHaveTextContent("<img src=x onerror=alert(1)>Volcán");
    expect(first).toHaveTextContent("<script>alert('xss')</script><b>bold</b> &lt;i&gt; text");
    expect(document.querySelector("script:not([src])")).toBeNull();
  });

  it("never turns non-http(s) source URLs into links", async () => {
    await open({}, hostile);
    const cards = screen.getAllByTestId("result");
    for (const card of cards.slice(1)) {
      expect(within(card).queryByRole("link", { name: /Open on Wikipedia/ })).toBeNull();
      expect(card).toHaveTextContent("Source link unavailable");
    }
    expect(document.querySelectorAll('a[href^="javascript:"], a[href^="data:"]')).toHaveLength(0);
  });

  it("keeps the source and licence attribution visible next to malicious content", async () => {
    await open({}, hostile);
    expect(screen.getByText(/Text is available under a Creative Commons licence/)).toBeVisible();
    expect(screen.getAllByRole("link", { name: /Creative Commons Attribution-Share Alike 4.0/ }).length).toBeGreaterThan(0);
  });

  it("does not use dangerouslySetInnerHTML anywhere in the source", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const dir = path.resolve(__dirname, "..");
    const files: string[] = [];
    const walk = (d: string) => {
      for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== "__tests__") walk(p);
        } else if (/\.(tsx?|jsx?)$/.test(entry.name)) files.push(p);
      }
    };
    walk(dir);
    expect(files.length).toBeGreaterThan(5);
    for (const file of files) expect(fs.readFileSync(file, "utf8")).not.toContain("dangerouslySetInnerHTML");
  });

  it("the public bundle source contains no admin key or admin endpoint", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const root = path.resolve(__dirname, "..");
    const scan = (d: string): string[] =>
      fs.readdirSync(d, { withFileTypes: true }).flatMap((e: import("node:fs").Dirent) => {
        const p = path.join(d, e.name);
        return e.isDirectory() ? (e.name === "__tests__" || e.name === "testing" ? [] : scan(p)) : [p];
      });
    for (const file of scan(root)) {
      const text = fs.readFileSync(file, "utf8").toLowerCase();
      expect(text, file).not.toContain("x-admin-key");
      expect(text, file).not.toContain("/admin/");
      expect(text, file).not.toContain("admin_api_key");
    }
  });
});
