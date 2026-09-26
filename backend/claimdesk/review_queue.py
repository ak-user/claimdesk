"""The examiner queue: assessed claims wait here until a person records a decision.

ClaimDesk prepares the file; the decision is always an examiner's, so `review` needs a reviewer
name and is refused once a claim has been decided. SQLite keeps it to one file with no server;
the default path, ":memory:", is enough for the demo and the tests.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime

from .models import ClaimAssessment, QueuedClaim, ReviewDecision

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submitted_at TEXT NOT NULL,
    text TEXT NOT NULL,
    assessment TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    note TEXT,
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS claims_by_status ON claims (review_status, id);
"""


class QueueError(ValueError):
    pass


class ReviewQueue:
    def __init__(self, path: str = ":memory:") -> None:
        # One connection shared by FastAPI's worker threads, serialised by a lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def submit(self, text: str, assessment: ClaimAssessment) -> QueuedClaim:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO claims (submitted_at, text, assessment) VALUES (?, ?, ?)",
                (_now(), text, assessment.model_dump_json()),
            )
        claim = self.get(cursor.lastrowid or 0)
        assert claim is not None
        return claim

    def get(self, claim_id: int) -> QueuedClaim | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return _to_claim(row) if row else None

    def list(self, review_status: str | None = None, limit: int = 50) -> list[QueuedClaim]:
        query, params = "SELECT * FROM claims", []
        if review_status:
            query += " WHERE review_status = ?"
            params.append(review_status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [_to_claim(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        return {status: count for status, count in rows}

    def review(self, claim_id: int, decision: ReviewDecision) -> QueuedClaim:
        with self._lock, self._db:
            updated = self._db.execute(
                "UPDATE claims SET review_status = ?, reviewer = ?, note = ?, reviewed_at = ? "
                "WHERE id = ? AND review_status = 'pending'",
                (decision.decision, decision.reviewer, decision.note, _now(), claim_id),
            ).rowcount
        claim = self.get(claim_id)
        if claim is None:
            raise QueueError(f"claim {claim_id} not found")
        if not updated:
            raise QueueError(f"claim {claim_id} was already reviewed by {claim.reviewer}")
        return claim

    def close(self) -> None:
        self._db.close()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_claim(row: sqlite3.Row) -> QueuedClaim:
    return QueuedClaim(
        id=row["id"],
        submitted_at=row["submitted_at"],
        text=row["text"],
        assessment=ClaimAssessment.model_validate(json.loads(row["assessment"])),
        review_status=row["review_status"],
        reviewer=row["reviewer"],
        note=row["note"],
        reviewed_at=row["reviewed_at"],
    )
