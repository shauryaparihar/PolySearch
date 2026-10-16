/** A failure reported by the API, or a local failure to reach/understand it. */
export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number = 0,
    public readonly retryAfterSeconds: number | null = null,
    public readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Codes created in the browser (not returned by the server). */
export const NETWORK = "NETWORK";
export const BAD_RESPONSE = "BAD_RESPONSE";
export const ABORTED = "ABORTED";

/** What the user sees. Never shows raw server messages, stack traces or request bodies. */
export function describeError(error: unknown): { kind: string; message: string } {
  if (!(error instanceof ApiError)) {
    return { kind: "unknown", message: "Something went wrong. Please try again." };
  }
  switch (error.code) {
    case "QUERY_INVALID":
      return { kind: "validation", message: "Enter between 2 and 500 characters." };
    case "FILTER_INVALID":
    case "MODE_INVALID":
    case "PAGE_SIZE_INVALID":
    case "REQUEST_INVALID":
      return { kind: "validation", message: "Those search options are not valid. Please adjust them." };
    case "RATE_LIMITED": {
      const wait = error.retryAfterSeconds;
      return {
        kind: "rate-limit",
        message: wait
          ? `Too many requests. Please wait ${wait} second${wait === 1 ? "" : "s"} and try again.`
          : "Too many requests. Please wait a moment and try again.",
      };
    }
    case "BUSY":
      return { kind: "busy", message: "The search service is busy. Please try again in a moment." };
    case "SEARCH_NOT_FOUND":
      return { kind: "expired", message: "These results have expired. Run the search again." };
    case "REQUEST_TOO_LARGE":
      return { kind: "validation", message: "That request is too large." };
    case NETWORK:
    case "INDEX_UNAVAILABLE":
      return { kind: "unavailable", message: "Search is unavailable right now. Please try again later." };
    default:
      return error.status >= 500 || error.code === BAD_RESPONSE
        ? { kind: "unavailable", message: "Search is unavailable right now. Please try again later." }
        : { kind: "unknown", message: "Something went wrong. Please try again." };
  }
}
