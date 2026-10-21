"""Proxy integration uses the same reservations, manifest, captures and cumulative ledger."""

from pathlib import Path
from typing import Any

import httpx

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.core.models import ChatRequest, Item
from servegauge.mock.backends import ReservedForward
from servegauge.runner.plan import RunPlan
from servegauge.runner.scheduler import run_plan
from servegauge.storage.raw_store import RawStore


def reserved_forwarder(
    plan: RunPlan,
    *,
    root: Path = Path("."),
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
) -> ReservedForward:
    if len(plan.endpoints) != 1:
        raise ValueError("proxy_plan_requires_exactly_one_endpoint")
    endpoint = plan.endpoints[0]

    async def forward(url: str, body: dict[str, Any]) -> dict[str, Any]:
        if (
            url.rstrip("/") != endpoint.base_url.rstrip("/")
            or body.get("model", endpoint.model) != endpoint.model
        ):
            raise ValueError("proxy_endpoint_or_model_mismatch")
        standard = {"messages", "tools", "tool_choice", "response_format", "stream", "model"}
        request = ChatRequest(
            messages=body["messages"],
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            response_format=body.get("response_format"),
            stream=bool(body.get("stream")),
            params={k: v for k, v in body.items() if k not in standard},
        )
        item = Item(
            item_id="proxy-" + content_hash(body),
            suite_id="S0",
            version="proxy-v1",
            payload=request.model_dump(mode="json"),
            expected={},
        )
        config = plan.model_copy(update={"suites": ["S0"]})
        result = await run_plan(
            config, root=root, approved=True, transports=transports, items_override=[item]
        )
        records = [e for e in RawStore(root / "data/raw").exchanges() if e.run_id == result.run_id]
        if result.errors or not records or records[0].response is None:
            raise SafeError("proxy_upstream_failed; inspect captured run")
        return records[0].response

    return forward
