import { useMemo, useRef, useState } from "react";
import { createClient, type ApiClient } from "./api/client";
import { ABORTED, ApiError, describeError } from "./api/errors";
import type { SearchParams } from "./api/types";
import { ResultsPanel, type ResultsView } from "./components/ResultsPanel";
import { SearchForm } from "./components/SearchForm";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; errorKind: string; message: string }
  | { kind: "done"; view: ResultsView };

export default function App({ client: injected }: { client?: ApiClient }) {
  const client = useMemo(() => injected ?? createClient(), [injected]);
  const [state, setState] = useState<State>({ kind: "idle" });
  const lastParams = useRef<SearchParams | null>(null);
  const inflight = useRef<AbortController | null>(null);

  async function runSearch(params: SearchParams) {
    inflight.current?.abort(); // a newer search replaces an older one
    const controller = new AbortController();
    inflight.current = controller;
    lastParams.current = params;
    setState({ kind: "loading" });
    const started = performance.now();
    try {
      const response = await client.search(params, controller.signal);
      if (controller.signal.aborted) return;
      setState({
        kind: "done",
        view: {
          response,
          results: response.results,
          nextCursor: response.next_cursor,
          roundTripMs: performance.now() - started,
          loadingMore: false,
          pageProblem: null,
          expired: false,
        },
      });
    } catch (error) {
      if (error instanceof ApiError && error.code === ABORTED) return;
      if (controller.signal.aborted) return;
      const { kind, message } = describeError(error);
      setState({ kind: "error", errorKind: kind, message });
    }
  }

  async function loadMore() {
    if (state.kind !== "done" || !lastParams.current) return;
    const { view } = state;
    const cursor = view.nextCursor;
    if (!cursor) return;
    setState({ kind: "done", view: { ...view, loadingMore: true, pageProblem: null } });
    try {
      const page = await client.page(view.response.search_id, cursor, lastParams.current.pageSize);
      setState({
        kind: "done",
        view: {
          ...view,
          results: [...view.results, ...page.results],
          nextCursor: page.next_cursor,
          loadingMore: false,
          pageProblem: null,
        },
      });
    } catch (error) {
      const { kind, message } = describeError(error);
      setState({
        kind: "done",
        view: { ...view, loadingMore: false, expired: kind === "expired", pageProblem: kind === "expired" ? null : message },
      });
    }
  }

  return (
    <div className="app">
      <header>
        <h1>PolySearch</h1>
        <p className="tagline">Search Wikipedia in English, Spanish or French and find the same topic in the other languages.</p>
      </header>
      <main>
        <SearchForm client={client} busy={state.kind === "loading"} onSearch={runSearch} />
        {state.kind === "loading" && (
          <p role="status" className="message info">
            Searching…
          </p>
        )}
        {state.kind === "error" && (
          <p role="alert" className={`message error ${state.errorKind}`}>
            {state.message}
          </p>
        )}
        {state.kind === "done" && (
          <ResultsPanel
            view={state.view}
            client={client}
            onLoadMore={loadMore}
            onSearchAgain={() => lastParams.current && runSearch(lastParams.current)}
          />
        )}
      </main>
      <footer>
        Results show how quickly this service answered and which ranking produced them; they are not a claim
        about quality.
      </footer>
    </div>
  );
}
