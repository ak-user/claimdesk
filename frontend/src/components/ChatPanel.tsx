import { useRef, useState } from "react";

import { api } from "../api";
import { describeFlag } from "../format";
import type { Answer } from "../types";
import { AnswerText, SourceList } from "./Citations";
import { Trace } from "./Trace";

const MEMBERS = ["M-1001", "M-1002", "M-1003"];

const EXAMPLES: { question: string; member: string | null }[] = [
  { question: "How much physio do I have left?", member: "M-1001" },
  { question: "What would I get back for a $120 physio session?", member: "M-1001" },
  { question: "Can I get a crown covered now?", member: "M-1002" },
  { question: "How long do I have to submit a claim?", member: null },
  { question: "Should I take ibuprofen for back pain?", member: null },
];

interface Exchange {
  id: number;
  question: string;
  member: string | null;
  answer?: Answer;
  error?: string;
}

export function ChatPanel() {
  const [member, setMember] = useState<string>("M-1001");
  const [question, setQuestion] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);

  async function ask(text: string, memberId: string | null) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    const id = nextId.current++;
    setExchanges((current) => [{ id, question: trimmed, member: memberId }, ...current]);
    setQuestion("");
    setBusy(true);
    try {
      const answer = await api.ask(trimmed, memberId);
      setExchanges((current) => current.map((e) => (e.id === id ? { ...e, answer } : e)));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Something went wrong.";
      setExchanges((current) => current.map((e) => (e.id === id ? { ...e, error: message } : e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="ask-title">
      <div className="panel-head">
        <h2 id="ask-title">Ask about benefits</h2>
        <p>Answers come from the plan booklet and the member's claims, with section numbers.</p>
      </div>

      <form
        className="ask-form"
        onSubmit={(event) => {
          event.preventDefault();
          void ask(question, member || null);
        }}
      >
        <label className="field-label" htmlFor="member">
          Member
        </label>
        <select id="member" value={member} onChange={(e) => setMember(e.target.value)}>
          <option value="">No member (policy questions only)</option>
          {MEMBERS.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <label className="field-label" htmlFor="question">
          Question
        </label>
        <div className="ask-row">
          <input
            id="question"
            value={question}
            maxLength={1000}
            placeholder="For example: Is massage therapy covered?"
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" className="primary" disabled={busy || !question.trim()}>
            {busy ? "Asking…" : "Ask"}
          </button>
        </div>
      </form>

      <div className="examples" role="group" aria-label="Example questions">
        {EXAMPLES.map((example) => (
          <button
            key={example.question}
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => {
              setMember(example.member ?? "");
              void ask(example.question, example.member);
            }}
          >
            {example.question}
            {example.member && <span className="chip-meta">{example.member}</span>}
          </button>
        ))}
      </div>

      <div className="exchanges">
        {exchanges.length === 0 && (
          <p className="empty">
            Pick an example or ask your own question. Balance questions need a member.
          </p>
        )}
        {exchanges.map((exchange) => (
          <ExchangeView key={exchange.id} exchange={exchange} />
        ))}
      </div>
    </section>
  );
}

function ExchangeView({ exchange }: { exchange: Exchange }) {
  const [active, setActive] = useState<string | null>(null);
  const toggle = (section: string) => setActive((current) => (current === section ? null : section));
  const { answer } = exchange;

  return (
    <article className="exchange">
      <p className="question">
        {exchange.question}
        {exchange.member && <span className="chip-meta">{exchange.member}</span>}
      </p>
      {!answer && !exchange.error && <p className="muted">Working…</p>}
      {exchange.error && <p className="error-text">{exchange.error}</p>}
      {answer && (
        <div className={answer.refused ? "answer answer-refused" : "answer"}>
          {answer.refused && <span className="tag">Outside what ClaimDesk decides</span>}
          <AnswerText text={answer.answer} active={active} onCite={toggle} />
          <SourceList citations={answer.citations} active={active} onCite={toggle} />
          {answer.flags.length > 0 && (
            <ul className="flags">
              {answer.flags.map((flag) => (
                <li key={flag}>{describeFlag(flag)}</li>
              ))}
            </ul>
          )}
          <Trace steps={answer.trace} provider={answer.provider} />
        </div>
      )}
    </article>
  );
}
