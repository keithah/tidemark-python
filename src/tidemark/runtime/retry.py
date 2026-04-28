"""Deterministic retry policy primitives for runtime loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class RetryDecision:
    """One scheduled retry attempt."""

    attempt: int
    delay_seconds: float
    next_retry_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.delay_seconds, (int, float)) or isinstance(self.delay_seconds, bool):
            raise ValueError("delay_seconds must be a finite non-negative number")
        delay = float(self.delay_seconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("delay_seconds must be a finite non-negative number")
        if not isinstance(self.next_retry_at, datetime):
            raise ValueError("next_retry_at must be a datetime")
        object.__setattr__(self, "delay_seconds", delay)
        object.__setattr__(self, "next_retry_at", _as_utc(self.next_retry_at))


@dataclass(frozen=True)
class RetryPolicy:
    """Capped exponential backoff policy.

    ``max_attempts=0`` disables retry. Attempts are one-based: attempt 1 uses
    ``initial_backoff_seconds``, attempt 2 multiplies once, and so on.
    """

    max_attempts: int = 0
    initial_backoff_seconds: float = 0.0
    max_backoff_seconds: float = 0.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 0:
            raise ValueError("max_attempts must be a non-negative integer")
        initial = _finite_non_negative_float(self.initial_backoff_seconds, "initial_backoff_seconds")
        maximum = _finite_non_negative_float(self.max_backoff_seconds, "max_backoff_seconds")
        multiplier = _finite_float(self.multiplier, "multiplier")
        if multiplier < 1.0:
            raise ValueError("multiplier must be >= 1")
        object.__setattr__(self, "initial_backoff_seconds", initial)
        object.__setattr__(self, "max_backoff_seconds", maximum)
        object.__setattr__(self, "multiplier", multiplier)

    def decision_for_attempt(self, attempt: int, *, now: datetime) -> RetryDecision | None:
        """Return the retry decision for one-based ``attempt``, or ``None`` when exhausted."""
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(now, datetime):
            raise ValueError("now must be a datetime")
        if self.max_attempts == 0 or attempt > self.max_attempts:
            return None

        delay = self.initial_backoff_seconds * (self.multiplier ** (attempt - 1))
        if self.max_backoff_seconds > 0:
            delay = min(delay, self.max_backoff_seconds)
        next_retry_at = _as_utc(now) + timedelta(seconds=delay)
        return RetryDecision(attempt=attempt, delay_seconds=delay, next_retry_at=next_retry_at)


def _finite_non_negative_float(value: object, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _finite_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return parsed


__all__ = ["RetryDecision", "RetryPolicy"]
