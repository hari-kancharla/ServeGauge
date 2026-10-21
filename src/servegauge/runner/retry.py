"""Bounded retry delays honor Retry-After and add seeded jitter."""

from __future__ import annotations

import random
from contextlib import suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from servegauge.core.models import Attempt


def retryable(attempt: Attempt) -> bool:
    return (
        attempt.error in {"transport_timeout", "transport_failure"}
        or attempt.status == 429
        or (attempt.status is not None and 500 <= attempt.status < 600)
    )


def delay_seconds(attempt: Attempt, base: float, maximum: float, rng: random.Random) -> float:
    header = attempt.response_headers.get("retry-after")
    advertised = 0.0
    if header:
        try:
            advertised = max(0, float(header))
        except ValueError:
            with suppress(ValueError, TypeError):
                advertised = max(
                    0, (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds()
                )
    jitter = rng.uniform(0.5, 1.5) * min(maximum, base * 2 ** (attempt.n - 1))
    # Never shorten an explicit server wait to our local jitter cap.
    return float(max(advertised, jitter))
