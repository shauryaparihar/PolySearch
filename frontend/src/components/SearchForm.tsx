import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { ApiClient } from "../api/client";
import { LANGUAGE_NAMES, LANGUAGES, type Language, type Mode, type SearchParams, type Suggestion } from "../api/types";
import { MIN_PREFIX, useSuggestions } from "../hooks/useSuggestions";

export const MAX_QUERY = 500;
const PAGE_SIZES = [5, 10, 20];

interface Props {
  client: ApiClient;
  busy: boolean;
  onSearch: (params: SearchParams) => void;
}

const othersOf = (lang: Language): Language[] => LANGUAGES.filter((l) => l !== lang);

export function SearchForm({ client, busy, onSearch }: Props) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [queryLanguage, setQueryLanguage] = useState<Language>("en");
  const [targets, setTargets] = useState<Language[]>(othersOf("en"));
  const [targetsTouched, setTargetsTouched] = useState(false);
  const [mode, setMode] = useState<Mode | "default">("default");
  const [pageSize, setPageSize] = useState(10);
  const [problem, setProblem] = useState<string | null>(null);

  const [focused, setFocused] = useState(false);
  const [suppress, setSuppress] = useState(false); // hide the list right after a choice
  const [active, setActive] = useState(-1);
  const items = useSuggestions(client, query, focused && !suppress);
  const open = focused && !suppress && items.length > 0 && query.trim().length >= MIN_PREFIX;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => setActive(-1), [items]);

  function changeQueryLanguage(next: Language) {
    setQueryLanguage(next);
    if (!targetsTouched) setTargets(othersOf(next)); // sensible default until the user chooses
  }

  function toggleTarget(lang: Language) {
    setTargetsTouched(true);
    setTargets((current) => (current.includes(lang) ? current.filter((l) => l !== lang) : [...current, lang]));
  }

  function submit(text: string = query) {
    const trimmed = text.trim().replace(/\s+/g, " ");
    if (trimmed.length < MIN_PREFIX || trimmed.length > MAX_QUERY) {
      setProblem(`Enter between ${MIN_PREFIX} and ${MAX_QUERY} characters.`);
      return;
    }
    if (targets.length === 0) {
      setProblem("Choose at least one language to search in.");
      return;
    }
    setProblem(null);
    setSuppress(true);
    onSearch({
      query: trimmed,
      queryLanguage,
      targetLanguages: LANGUAGES.filter((l) => targets.includes(l)),
      mode: mode === "default" ? null : mode,
      pageSize,
    });
  }

  function choose(suggestion: Suggestion) {
    setQuery(suggestion.text);
    submit(suggestion.text);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" && items.length) {
      event.preventDefault();
      setSuppress(false);
      setActive((i) => (i + 1) % items.length);
    } else if (event.key === "ArrowUp" && items.length) {
      event.preventDefault();
      setActive((i) => (i <= 0 ? items.length - 1 : i - 1));
    } else if (event.key === "Enter" && open && active >= 0) {
      event.preventDefault();
      choose(items[active]);
    } else if (event.key === "Escape") {
      setSuppress(true);
      setActive(-1);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  const listId = `${id}-suggestions`;
  return (
    <form className="search-form" onSubmit={onSubmit} role="search" noValidate>
      <div className="query-row">
        <div className="combobox">
          <label htmlFor={`${id}-q`} className="visually-hidden">
            Search query
          </label>
          <input
            ref={inputRef}
            id={`${id}-q`}
            type="search"
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={open && active >= 0 ? `${id}-opt-${active}` : undefined}
            autoComplete="off"
            placeholder="Search in English, Spanish or French"
            value={query}
            maxLength={MAX_QUERY + 50}
            onChange={(e) => {
              setQuery(e.target.value);
              setSuppress(false);
              setProblem(null);
            }}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={onKeyDown}
          />
          {open && (
            <ul id={listId} role="listbox" aria-label="Suggestions" className="suggestions">
              {items.map((s, i) => (
                <li
                  key={s.text}
                  id={`${id}-opt-${i}`}
                  role="option"
                  aria-selected={i === active}
                  className={i === active ? "active" : undefined}
                  onMouseDown={(e) => {
                    e.preventDefault(); // keep focus in the input
                    choose(s);
                  }}
                >
                  <span>{s.text}</span>
                  <small>{s.languages.map((l) => l.toUpperCase()).join(" · ")}</small>
                </li>
              ))}
            </ul>
          )}
        </div>
        {/* Not disabled while busy: a newer search must be able to replace a slow one, and a
            disabled default button would also stop Enter from submitting the form. */}
        <button type="submit" aria-busy={busy}>
          {busy ? "Searching…" : "Search"}
        </button>
      </div>

      <div className="filters">
        <label>
          Query language
          <select value={queryLanguage} onChange={(e) => changeQueryLanguage(e.target.value as Language)}>
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>
                {LANGUAGE_NAMES[l]}
              </option>
            ))}
          </select>
        </label>

        <fieldset>
          <legend>Search in</legend>
          {LANGUAGES.map((l) => (
            <label key={l} className="check">
              <input type="checkbox" checked={targets.includes(l)} onChange={() => toggleTarget(l)} />
              {LANGUAGE_NAMES[l]}
            </label>
          ))}
        </fieldset>

        <fieldset>
          <legend>Source</legend>
          <label className="check">
            <input type="checkbox" checked disabled readOnly />
            Wikipedia
          </label>
        </fieldset>

        <label>
          Search mode
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode | "default")}>
            <option value="default">Default (recommended)</option>
            <option value="dense">Meaning (dense)</option>
            <option value="lexical">Keywords (lexical)</option>
            <option value="hybrid">Combined (hybrid)</option>
          </select>
        </label>

        <label>
          Results per page
          <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
            {PAGE_SIZES.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      {problem && (
        <p role="alert" className="message validation">
          {problem}
        </p>
      )}
    </form>
  );
}
