import { useEffect, useState } from "react";

import { api } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { ClaimPanel } from "./components/ClaimPanel";
import { formatDate } from "./format";
import type { Health } from "./types";

const PROVIDER_LABEL: Record<Health["llm_provider"], string> = {
  offline: "Offline planner",
  azure: "Azure OpenAI",
  openai: "OpenAI",
};

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth, (error: Error) => setHealthError(error.message));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            CD
          </span>
          <div>
            <h1>ClaimDesk</h1>
            <p>Benefits answers and claim intake for the fictional Harbourline Mutual plan</p>
          </div>
        </div>
        <div className="status" aria-live="polite">
          {health && (
            <>
              <span className={`pill provider-${health.llm_provider}`}>
                {PROVIDER_LABEL[health.llm_provider]}
              </span>
              <span className="pill">
                {health.retrieval === "hybrid" ? "Hybrid search" : "BM25 search"}
              </span>
              <span className="muted">Demo date {formatDate(health.as_of)}</span>
            </>
          )}
          {!health && healthError && <span className="pill pill-bad">API unreachable</span>}
          {!health && !healthError && <span className="pill">Connecting…</span>}
        </div>
      </header>

      <main className="layout">
        <ChatPanel />
        <ClaimPanel />
      </main>

      <footer className="footer">
        All people, plans and amounts are fictional. ClaimDesk never approves or denies a claim:
        a claims examiner reviews every one.
      </footer>
    </div>
  );
}
