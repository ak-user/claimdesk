"""Per-client rate limiting and request IDs for the API.

Every question and claim can cost a model call, so each client gets a token bucket: a burst of
`per_minute` requests, refilled continuously. It is in memory and per process, which suits one
container; several replicas would need a shared store such as Redis.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

request_id: ContextVar[str] = ContextVar("request_id", default="-")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Forget clients that have been idle long enough to have a full bucket again.
_PRUNE_EVERY = 1000


def new_request_id(incoming: str | None) -> str:
    """Keeps a caller's X-Request-ID if it is safe to log, otherwise makes one."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


class RequestIdFilter(logging.Filter):
    """Adds the current request ID to every log record as `request_id`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._calls = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def check(self, key: str) -> float:
        """Takes a token for `key`. Returns 0 if allowed, else seconds until the next token."""
        if not self.enabled:
            return 0.0
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % _PRUNE_EVERY == 0:
                self._prune(now)
            bucket = self._buckets.setdefault(key, _Bucket(self.capacity, now))
            bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return 0.0
            return (1 - bucket.tokens) / self.rate

    def _prune(self, now: float) -> None:
        full_after = self.capacity / self.rate
        idle = [key for key, b in self._buckets.items() if now - b.updated > full_after]
        for key in idle:
            del self._buckets[key]
