from typing import Any

from servegauge.core.ids import exchange_id
from servegauge.core.models import Attempt, CheckResult, Exchange, Timings


def fingerprint_cards(checks: list[CheckResult]) -> list[CheckResult]:
    """Keep every FP assertion while explicitly checking the added mock routing card."""
    others = [c for c in checks if not c.check_id.startswith("FP")]
    assert len(others) == 1 and others[0].check_id == "S8.routing.provider"
    assert others[0].verdict == "NOT_APPLICABLE" and not others[0].details["operational_alarm"]
    return [c for c in checks if c.check_id.startswith("FP")]


def exchange(sample: int = 0, **updates: Any) -> Exchange:
    data: dict[str, Any] = {
        "exchange_id": exchange_id("run-a", "mock:one", "S1", "P01-a", sample),
        "run_id": "run-a",
        "endpoint_id": "mock:one",
        "suite_id": "S1",
        "item_id": "P01-a",
        "sample_idx": sample,
        "request": {"messages": [{"role": "user", "content": "hello"}]},
        "attempts": [Attempt(n=1, status=200, timings=Timings(start_unix=1, total_ms=2))],
        "status": 200,
        "response": {"choices": [{"message": {"content": "hello"}}]},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "cost_usd": 0,
        "cost_source": "estimate",
    }
    data.update(updates)
    return Exchange(**data)
