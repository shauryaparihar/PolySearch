import { useState } from "react";
import type { ApiClient } from "../api/client";
import { LANGUAGE_NAMES, type FeedbackLabel, type ResultItem, type SourceAttribution } from "../api/types";

/** Only plain http(s) links are ever rendered as links (never javascript:, data:, etc.). */
export function isSafeHttpUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch {
    return false;
  }
}

export function newIdempotencyKey(): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  return `click-${random}`; // matches the API's [A-Za-z0-9_-]{8,64}
}

interface Props {
  item: ResultItem;
  searchId: string;
  client: ApiClient;
  attribution?: SourceAttribution;
}

export function ResultCard({ item, searchId, client, attribution }: Props) {
  const [label, setLabel] = useState<FeedbackLabel | null>(null);
  const [feedbackError, setFeedbackError] = useState(false);
  const [saving, setSaving] = useState(false);

  /** Start recording the click before the browser follows the link. The request is made with
   * keepalive so it survives navigation, and any failure is ignored: opening the source never
   * depends on tracking. */
  function recordClick() {
    void client.click(searchId, item.document_id, newIdempotencyKey()).catch(() => undefined);
  }

  async function sendFeedback(next: FeedbackLabel) {
    if (saving || next === label) return;
    const previous = label;
    setLabel(next); // optimistic
    setFeedbackError(false);
    setSaving(true);
    try {
      await client.feedback(searchId, item.document_id, next);
    } catch {
      setLabel(previous);
      setFeedbackError(true);
    } finally {
      setSaving(false);
    }
  }

  const ranks = [
    item.dense_rank ? `meaning #${item.dense_rank}` : null,
    item.lexical_rank ? `keywords #${item.lexical_rank}` : null,
  ].filter(Boolean);

  return (
    <li className="result" data-testid="result">
      <div className="result-head">
        <span className="rank">{item.rank}</span>
        <h3>{item.title}</h3>
      </div>
      <p className="badges">
        <span className="badge">{LANGUAGE_NAMES[item.language]}</span>
        <span className="badge">{item.source === "wikipedia" ? "Wikipedia" : item.source}</span>
        {attribution && (
          <a className="badge licence" href={isSafeHttpUrl(attribution.licence_url) ? attribution.licence_url : undefined} target="_blank" rel="noopener noreferrer">
            {attribution.licence_name}
          </a>
        )}
      </p>
      <p className="snippet">{item.snippet}</p>
      <div className="result-foot">
        {isSafeHttpUrl(item.source_url) ? (
          <a href={item.source_url} target="_blank" rel="noopener noreferrer" onClick={recordClick}>
            Open on Wikipedia ↗
          </a>
        ) : (
          <span className="muted">Source link unavailable</span>
        )}
        {ranks.length > 0 && <span className="muted ranks">{ranks.join(" · ")}</span>}
        <span className="feedback" role="group" aria-label={`Was "${item.title}" relevant?`}>
          <button type="button" aria-pressed={label === "relevant"} disabled={saving} onClick={() => sendFeedback("relevant")}>
            Relevant
          </button>
          <button type="button" aria-pressed={label === "not_relevant"} disabled={saving} onClick={() => sendFeedback("not_relevant")}>
            Not relevant
          </button>
        </span>
      </div>
      {feedbackError && (
        <p role="alert" className="message error small">
          Could not save your feedback. Please try again.
        </p>
      )}
    </li>
  );
}
