import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { money } from "../format";
import type { QueuedClaim, ReviewDecision, ReviewStatus } from "../types";
import { Assessment, STATUS_LABEL } from "./ClaimPanel";

const FILTERS: { status: ReviewStatus | null; label: string }[] = [
  { status: "pending", label: "Pending" },
  { status: "approved", label: "Approved" },
  { status: "denied", label: "Denied" },
  { status: "returned", label: "Returned" },
  { status: null, label: "All" },
];

const REVIEW_LABEL: Record<ReviewStatus, string> = {
  pending: "Pending",
  approved: "Approved",
  denied: "Denied",
  returned: "Returned to member",
};

const timestamp = new Intl.DateTimeFormat("en-CA", { dateStyle: "medium", timeStyle: "short" });

export function QueuePanel({ onReviewed }: { onReviewed?: () => void }) {
  const [filter, setFilter] = useState<ReviewStatus | null>("pending");
  const [claims, setClaims] = useState<QueuedClaim[]>([]);
  const [selected, setSelected] = useState<QueuedClaim | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setClaims(await api.queue(filter));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't load the queue.");
    }
  }, [filter]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="panel queue" aria-labelledby="queue-title">
      <div className="panel-head">
        <h2 id="queue-title">Examiner queue</h2>
        <p>Claims sent from intake. ClaimDesk prepares each file; an examiner decides.</p>
      </div>

      <div className="examples" role="group" aria-label="Filter by review status">
        {FILTERS.map(({ status, label }) => (
          <button
            key={label}
            type="button"
            className={filter === status ? "chip chip-active" : "chip"}
            aria-pressed={filter === status}
            onClick={() => {
              setFilter(status);
              setSelected(null);
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {error && <p className="error-text">{error}</p>}
      {!error && claims.length === 0 && (
        <p className="empty">
          Nothing here. Check a claim in intake and choose "Send to examiner queue".
        </p>
      )}

      <div className="queue-layout">
        {claims.length > 0 && (
          <ul className="queue-list">
            {claims.map((claim) => (
              <li key={claim.id}>
                <button
                  type="button"
                  className={selected?.id === claim.id ? "queue-row queue-row-active" : "queue-row"}
                  onClick={() => setSelected(claim)}
                >
                  <span className="queue-id">#{claim.id}</span>
                  <span>
                    {claim.assessment.benefit_display ?? "Unrecognized service"}
                    <span className="chip-meta">{claim.assessment.fields.member_id ?? "no ID"}</span>
                  </span>
                  <span className="muted">
                    {claim.assessment.fields.amount !== null
                      ? money(claim.assessment.fields.amount)
                      : "—"}
                  </span>
                  <span className={`pill review-${claim.review_status}`}>
                    {REVIEW_LABEL[claim.review_status]}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {selected && (
          <div className="queue-detail">
            <p className="muted">
              Claim #{selected.id}, submitted {timestamp.format(new Date(selected.submitted_at))}.
              Intake: {STATUS_LABEL[selected.assessment.status].toLowerCase()}.
            </p>
            <Assessment result={selected.assessment} />
            {selected.review_status === "pending" ? (
              <ReviewForm
                claim={selected}
                onDone={(claim) => {
                  setSelected(claim);
                  void load();
                  onReviewed?.();
                }}
              />
            ) : (
              <p className={`review-result review-${selected.review_status}`}>
                {REVIEW_LABEL[selected.review_status]} by {selected.reviewer}
                {selected.reviewed_at && ` on ${timestamp.format(new Date(selected.reviewed_at))}`}
                {selected.note && `: ${selected.note}`}
              </p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function ReviewForm({
  claim,
  onDone,
}: {
  claim: QueuedClaim;
  onDone: (claim: QueuedClaim) => void;
}) {
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(decision: ReviewDecision["decision"]) {
    setBusy(true);
    setError(null);
    try {
      const body = { decision, reviewer: reviewer.trim(), note: note.trim() || null };
      onDone(await api.review(claim.id, body));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't save the decision.");
    } finally {
      setBusy(false);
    }
  }

  const ready = !busy && reviewer.trim().length > 0;
  return (
    <form className="review-form" onSubmit={(event) => event.preventDefault()}>
      <label className="field-label" htmlFor="reviewer">
        Examiner
      </label>
      <input
        id="reviewer"
        value={reviewer}
        maxLength={80}
        placeholder="Your name"
        onChange={(e) => setReviewer(e.target.value)}
      />
      <label className="field-label" htmlFor="review-note">
        Note (optional)
      </label>
      <textarea
        id="review-note"
        rows={2}
        value={note}
        maxLength={2000}
        onChange={(e) => setNote(e.target.value)}
      />
      <div className="claim-actions">
        <button
          type="button"
          className="primary"
          disabled={!ready}
          onClick={() => void decide("approved")}
        >
          Approve
        </button>
        <button type="button" disabled={!ready} onClick={() => void decide("returned")}>
          Return to member
        </button>
        <button type="button" disabled={!ready} onClick={() => void decide("denied")}>
          Deny
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}
    </form>
  );
}
