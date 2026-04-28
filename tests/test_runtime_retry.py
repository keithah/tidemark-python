from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tidemark.runtime.retry import RetryDecision, RetryPolicy


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_retry_policy_is_disabled_by_default() -> None:
    policy = RetryPolicy()

    assert policy.decision_for_attempt(1, now=utc("2026-04-28T12:00:00+00:00")) is None


def test_retry_policy_returns_deterministic_decision_for_attempt() -> None:
    now = utc("2026-04-28T12:00:00+00:00")
    policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=2.5, max_backoff_seconds=30.0)

    decision = policy.decision_for_attempt(1, now=now)

    assert decision == RetryDecision(
        attempt=1,
        delay_seconds=2.5,
        next_retry_at=utc("2026-04-28T12:00:02.500000+00:00"),
    )


def test_retry_policy_exponential_delay_is_capped() -> None:
    now = utc("2026-04-28T12:00:00+00:00")
    policy = RetryPolicy(max_attempts=5, initial_backoff_seconds=3.0, max_backoff_seconds=10.0, multiplier=2.0)

    assert policy.decision_for_attempt(1, now=now).delay_seconds == 3.0  # type: ignore[union-attr]
    assert policy.decision_for_attempt(2, now=now).delay_seconds == 6.0  # type: ignore[union-attr]
    assert policy.decision_for_attempt(3, now=now).delay_seconds == 10.0  # type: ignore[union-attr]
    assert policy.decision_for_attempt(5, now=now).delay_seconds == 10.0  # type: ignore[union-attr]


def test_retry_policy_returns_none_when_attempts_are_exhausted() -> None:
    policy = RetryPolicy(max_attempts=2, initial_backoff_seconds=1.0, max_backoff_seconds=5.0)

    assert policy.decision_for_attempt(3, now=utc("2026-04-28T12:00:00+00:00")) is None


def test_retry_policy_zero_initial_backoff_schedules_immediate_retry() -> None:
    now = utc("2026-04-28T12:00:00+00:00")
    policy = RetryPolicy(max_attempts=1, initial_backoff_seconds=0.0, max_backoff_seconds=5.0)

    decision = policy.decision_for_attempt(1, now=now)

    assert decision == RetryDecision(attempt=1, delay_seconds=0.0, next_retry_at=now)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": -1},
        {"max_attempts": True},
        {"initial_backoff_seconds": -0.1},
        {"initial_backoff_seconds": float("nan")},
        {"max_backoff_seconds": -0.1},
        {"max_backoff_seconds": float("nan")},
        {"multiplier": 0.99},
        {"multiplier": float("nan")},
    ],
)
def test_retry_policy_rejects_malformed_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_retry_policy_rejects_malformed_attempt_and_clock_values() -> None:
    policy = RetryPolicy(max_attempts=1)

    with pytest.raises(ValueError):
        policy.decision_for_attempt(0, now=utc("2026-04-28T12:00:00+00:00"))
    with pytest.raises(ValueError):
        policy.decision_for_attempt(True, now=utc("2026-04-28T12:00:00+00:00"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        policy.decision_for_attempt(1, now="not-a-datetime")  # type: ignore[arg-type]
