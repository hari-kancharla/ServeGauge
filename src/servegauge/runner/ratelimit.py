"""Per-endpoint token buckets with injectable time for deterministic tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    def __init__(
        self,
        per_minute: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.rate = per_minute / 60
        self.capacity = capacity
        self.tokens = capacity
        self.clock, self.sleep = clock, sleep
        self.updated = clock()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: float = 1) -> None:
        if amount > self.capacity or amount < 0:
            raise ValueError("request_exceeds_rate_bucket_capacity")
        async with self.lock:
            while True:
                now = self.clock()
                self.tokens = min(
                    self.capacity, self.tokens + max(0, now - self.updated) * self.rate
                )
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await self.sleep((amount - self.tokens) / self.rate)
