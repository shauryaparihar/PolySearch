export type Language = "en" | "es" | "fr";
export type Mode = "dense" | "lexical" | "hybrid";

export const LANGUAGES: readonly Language[] = ["en", "es", "fr"];
export const LANGUAGE_NAMES: Record<Language, string> = {
  en: "English",
  es: "Español",
  fr: "Français",
};

export interface SearchParams {
  query: string;
  queryLanguage: Language;
  targetLanguages: Language[];
  mode: Mode | null; // null = the index's evidence-selected default
  pageSize: number;
}

export interface SourceAttribution {
  name: string;
  licence_name: string;
  licence_url: string;
  attribution_text: string;
}

export interface ResultItem {
  rank: number;
  document_id: string;
  title: string;
  snippet: string; // plain text: always rendered as text, never as HTML
  language: Language;
  source: string;
  source_url: string;
  source_revision: string | null;
  score: number;
  dense_rank: number | null;
  lexical_rank: number | null;
}

export interface ResultPage {
  search_id: string;
  results: ResultItem[];
  next_cursor: string | null;
  total_results: number;
  attribution: Record<string, SourceAttribution>;
}

export interface SearchResponse extends ResultPage {
  index_version: string;
  mode: Mode;
  cache_hit: boolean;
  latency_ms: number;
}

export interface Suggestion {
  text: string;
  languages: Language[];
}

export interface ClickResponse {
  click_id: string;
  rank: number;
  duplicate: boolean;
}

export type FeedbackLabel = "relevant" | "not_relevant";
