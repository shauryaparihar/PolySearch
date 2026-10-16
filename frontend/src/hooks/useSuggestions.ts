import { useEffect, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Suggestion } from "../api/types";
import { useDebouncedValue } from "./useDebouncedValue";

export const SUGGEST_DELAY_MS = 250;
export const MIN_PREFIX = 2;

/**
 * Debounced title suggestions. Each new prefix cancels the previous in-flight request, so a
 * slow answer for an old prefix can never overwrite the answer for the current one.
 */
export function useSuggestions(client: ApiClient, query: string, enabled: boolean): Suggestion[] {
  const prefix = useDebouncedValue(query.trim(), SUGGEST_DELAY_MS);
  const [items, setItems] = useState<Suggestion[]>([]);

  useEffect(() => {
    if (!enabled || prefix.length < MIN_PREFIX) {
      setItems([]);
      return;
    }
    const controller = new AbortController();
    client
      .suggestions(prefix, controller.signal)
      .then((found) => {
        if (!controller.signal.aborted) setItems(found);
      })
      .catch(() => {
        if (!controller.signal.aborted) setItems([]); // suggestions are optional: fail quietly
      });
    return () => controller.abort();
  }, [client, prefix, enabled]);

  return items;
}
