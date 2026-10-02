import { useEffect, useRef, useState } from "react";

import { api } from "../api";
import { describeFlag, formatDate, money } from "../format";
import type {
  ClaimAssessment,
  ClaimFields,
  ClaimStatus,
  Issue,
  QueuedClaim,
  Sample,
} from "../types";
import { SourceList } from "./Citations";

export const STATUS_LABEL: Record<ClaimStatus, string> = {
  ready_for_review: "Ready for examiner review",
  needs_correction: "Needs correction",
  not_payable: "Not payable",
};

const FIELD_LABEL: Record<keyof ClaimFields, string> = {
  member_id: "Member ID",
  patient_name: "Patient",
  service_type: "Service",
  service_date: "Service date",
  provider_name: "Provider",
  provider_registration: "Registration #",
  amount: "Amount charged",
};

const SEVERITY_MARK: Record<Issue["severity"], string> = { error: "✕", warning: "!", info: "i" };

export function ClaimPanel({ onQueued }: { onQueued?: (claim: QueuedClaim) => void }) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [text, setText] = useState("");
  const [result, setResult] = useState<ClaimAssessment | null>(null);
  // The text behind `result`, so it can be queued; uploads only keep the assessment.
  const [checkedText, setCheckedText] = useState<string | null>(null);
  const [queued, setQueued] = useState<QueuedClaim | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then(setSamples, () => setSamples([]));
  }, []);

  async function run(task: () => Promise<ClaimAssessment>, source: string | null = null) {
    setBusy(true);
    setError(null);
    setQueued(null);
    try {
      setResult(await task());
      setCheckedText(source);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSample(name: string) {
    try {
      const sample = await api.sample(name);
      setText(sample.text);
      await run(() => api.assess(sample.text), sample.text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't load the sample.");
    }
  }

  async function sendToQueue(source: string) {
    setBusy(true);
    try {
      const claim = await api.submit(source);
      setQueued(claim);
      onQueued?.(claim);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't queue the claim.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="claim-title">
      <div className="panel-head">
        <h2 id="claim-title">Claim intake</h2>
        <p>Reads a claim form, checks it against the plan rules and estimates the payment.</p>
      </div>

      <div className="examples" role="group" aria-label="Sample claims">
        {samples.map((sample) => (
          <button
            key={sample.name}
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => void loadSample(sample.name)}
          >
            {sampleLabel(sample.name)}
            <span className="chip-meta">{sample.name.match(/M-\d{4}/)?.[0]}</span>
          </button>
        ))}
      </div>

      <form
        className="claim-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim()) void run(() => api.assess(text), text);
        }}
      >
        <label className="field-label" htmlFor="claim-text">
          Claim text
        </label>
        <textarea
          id="claim-text"
          rows={8}
          value={text}
          maxLength={20000}
          placeholder="Paste a claim form, or pick a sample above."
          onChange={(e) => setText(e.target.value)}
        />
        <div className="claim-actions">
          <button type="submit" className="primary" disabled={busy || !text.trim()}>
            {busy ? "Checking…" : "Check claim"}
          </button>
          <button type="button" disabled={busy} onClick={() => fileInput.current?.click()}>
            Upload .txt or .pdf
          </button>
          <input
            ref={fileInput}
            id="claim-file"
            type="file"
            accept=".txt,.md,.pdf"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void run(() => api.upload(file));
            }}
          />
        </div>
      </form>

      {error && <p className="error-text">{error}</p>}
      {result && <Assessment result={result} />}
      {result && checkedText && (
        <div className="queue-action">
          {queued ? (
            <p className="muted">Sent to the examiner queue as claim #{queued.id}.</p>
          ) : (
            <button type="button" disabled={busy} onClick={() => void sendToQueue(checkedText)}>
              Send to examiner queue
            </button>
          )}
        </div>
      )}
      {!result && !error && (
        <p className="empty">Pick a sample claim to see the checks: each one shows a different rule.</p>
      )}
    </section>
  );
}

export function Assessment({ result }: { result: ClaimAssessment }) {
  const { fields, estimate } = result;
  return (
    <div className="assessment">
      <div className={`status-banner status-${result.status}`}>
        <strong>{STATUS_LABEL[result.status]}</strong>
        <span>{result.summary}</span>
      </div>

      {result.flags.includes("possible_prompt_injection") && (
        <p className="injection">{describeFlag("possible_prompt_injection")} It was ignored.</p>
      )}

      <div className="assessment-grid">
        <table className="fields">
          <caption>
            Fields read by the {result.extractor === "llm" ? "model" : "rule-based reader"}
          </caption>
          <tbody>
            {(Object.keys(FIELD_LABEL) as (keyof ClaimFields)[]).map((key) => (
              <tr key={key}>
                <th scope="row">{FIELD_LABEL[key]}</th>
                <td>{fieldValue(key, fields[key])}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">Benefit</th>
              <td>{result.benefit_display ?? <span className="missing">Not recognized</span>}</td>
            </tr>
          </tbody>
        </table>

        <div className="estimate">
          <h4>Estimated reimbursement</h4>
          {estimate ? (
            <>
              <p className="estimate-amount">{money(estimate.estimate)}</p>
              <p className="muted">
                {estimate.coinsurance_percent}% of {money(fields.amount ?? 0)} ={" "}
                {money(estimate.covered_share)}
                {estimate.capped_by_maximum && ", limited by the remaining maximum"}
              </p>
              {estimate.remaining_before !== null && estimate.remaining_after !== null && (
                <p className="muted">
                  {estimate.benefit_display} left: {money(estimate.remaining_before)} →{" "}
                  {money(estimate.remaining_after)}
                </p>
              )}
              <p className="muted">Pending examiner review.</p>
            </>
          ) : (
            <p className="muted">No estimate: the claim isn't payable as submitted.</p>
          )}
        </div>
      </div>

      {result.issues.length > 0 && (
        <ul className="issues">
          {result.issues.map((issue, index) => (
            <li key={index} className={`issue issue-${issue.severity}`}>
              <span className="issue-mark" aria-hidden="true">
                {SEVERITY_MARK[issue.severity]}
              </span>
              <span>
                {issue.message}
                {issue.section && <span className="section-ref">§{issue.section}</span>}
              </span>
            </li>
          ))}
        </ul>
      )}

      <SourceList citations={result.citations} />
    </div>
  );
}

function fieldValue(key: keyof ClaimFields, value: string | number | null) {
  if (value === null || value === "") return <span className="missing">Missing</span>;
  if (key === "amount" && typeof value === "number") return money(value);
  if (key === "service_date" && typeof value === "string") return formatDate(value);
  return value;
}

function sampleLabel(name: string): string {
  const kind = name.split("_")[0];
  const labels: Record<string, string> = {
    physio: "Physio receipt",
    dental: "Dental crown",
    massage: "Massage, odd note",
    vision: "Glasses, sent late",
  };
  return labels[kind] ?? name;
}
