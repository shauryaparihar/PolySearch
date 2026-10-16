import type { ApiClient } from "../api/client";
import type { ResultItem, SearchResponse } from "../api/types";
import { ResultCard } from "./ResultCard";

export interface ResultsView {
  response: SearchResponse;
  results: ResultItem[];
  nextCursor: string | null;
  roundTripMs: number;
  loadingMore: boolean;
  pageProblem: string | null;
  expired: boolean;
}

interface Props {
  view: ResultsView;
  client: ApiClient;
  onLoadMore: () => void;
  onSearchAgain: () => void;
}

export function ResultsPanel({ view, client, onLoadMore, onSearchAgain }: Props) {
  const { response, results } = view;
  const source = Object.values(response.attribution)[0];
  return (
    <section aria-label="Search results">
      <p className="meta" role="status">
        {response.total_results} result{response.total_results === 1 ? "" : "s"} · {response.mode} mode ·{" "}
        server {Math.round(response.latency_ms)} ms · round trip {Math.round(view.roundTripMs)} ms ·{" "}
        {response.cache_hit ? "cache hit" : "cache miss"} · index {response.index_version}
      </p>
      {source && (
        <p className="attribution">
          {source.attribution_text}{" "}
          <a href={source.licence_url} target="_blank" rel="noopener noreferrer">
            {source.licence_name}
          </a>
        </p>
      )}
      {results.length === 0 ? (
        <p className="message info">No results. Try different words or other languages.</p>
      ) : (
        <ol className="results">
          {results.map((item) => (
            <ResultCard
              key={item.document_id}
              item={item}
              searchId={response.search_id}
              client={client}
              attribution={response.attribution[item.source]}
            />
          ))}
        </ol>
      )}
      {view.expired ? (
        <p role="alert" className="message error">
          These results have expired. Run the search again.{" "}
          <button type="button" onClick={onSearchAgain}>
            Search again
          </button>
        </p>
      ) : (
        <>
          {view.pageProblem && (
            <p role="alert" className="message error">
              {view.pageProblem}
            </p>
          )}
          {view.nextCursor && (
            <button type="button" className="more" onClick={onLoadMore} disabled={view.loadingMore}>
              {view.loadingMore ? "Loading…" : "Load more results"}
            </button>
          )}
        </>
      )}
    </section>
  );
}
