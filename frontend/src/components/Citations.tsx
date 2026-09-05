import type { ReactNode } from "react";

import type { Citation } from "../types";

const CITATION = /\[(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\]/g;

/** Renders answer text with each [4.2] marker as a button that highlights its source. */
export function AnswerText({
  text,
  active,
  onCite,
}: {
  text: string;
  active: string | null;
  onCite: (section: string) => void;
}) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const start = match.index ?? 0;
    parts.push(text.slice(last, start));
    for (const section of match[1].split(",").map((s) => s.trim())) {
      parts.push(
        <button
          key={`${start}-${section}`}
          type="button"
          className={`cite${active === section ? " cite-active" : ""}`}
          onClick={() => onCite(section)}
          aria-label={`Show section ${section}`}
        >
          {section}
        </button>,
      );
    }
    last = start + match[0].length;
  }
  parts.push(text.slice(last));
  return <p className="answer-text">{parts}</p>;
}

export function SourceList({
  citations,
  active,
  onCite,
}: {
  citations: Citation[];
  active?: string | null;
  onCite?: (section: string) => void;
}) {
  if (citations.length === 0) return null;
  return (
    <div className="sources">
      <h4>Booklet sections</h4>
      <ul>
        {citations.map((citation) => (
          <li
            key={citation.section}
            className={active === citation.section ? "source source-active" : "source"}
          >
            <button
              type="button"
              className="source-head"
              onClick={() => onCite?.(citation.section)}
              disabled={!onCite}
            >
              <span className="source-number">{citation.section}</span>
              <span className="source-title">{citation.title}</span>
            </button>
            <p className="source-snippet">{citation.snippet}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
