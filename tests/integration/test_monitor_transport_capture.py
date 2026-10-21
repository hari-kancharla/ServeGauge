"""Reserved native wire-to-first-attempt extraction, without temporal qualification."""

import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.observatory.changes import load_transport_policy
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.transport import transport_batch
from servegauge.runner.check import _mock_metadata
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.raw_store import RawStore
from servegauge.suites.s1_protocol import protocol_items


@pytest.mark.slow
async def test_actual_first_attempts_match_wire_and_ignore_successful_retries(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="transport-native",
        profile="gpt-oss-120b",
        level="core",
        seed=1307,
        reference={"tier": "MOCK"},
        max_usd=0,
        endpoints=[endpoint],
        suites=["S1"],
        requests_per_minute=600000,
    )
    role_seed = int(content_hash([plan.seed, endpoint.id])[:15], 16)
    faults = FaultConfig(seed=role_seed, error_rate=0.1)
    service = create_app(seed=role_seed, faults=faults, sampling_version="v2", context_length=512)
    calls = []

    @service.middleware("http")
    async def observe(request, call_next):
        body = await request.json() if request.method == "POST" else None
        response = await call_next(request)
        if body is not None:
            calls.append((content_hash(body), response.status_code))
        return response

    transports = {endpoint.id: httpx.ASGITransport(app=service)}
    await _mock_metadata(plan, transports)
    # Canonical 400 P01 items; no omitted requested probe or reduced trial count.
    items = [i for i in protocol_items("core", plan.seed) if i.expected.get("check_id") == "P01"]
    estimate = estimate_plan(plan, items_override=items)
    assert not calls and estimate["total_reserved_cost_usd"] == "0"
    assert estimate["cost_table"][0]["n_requests"] == 400
    source = implementation_hash()
    run = await execute_plan(
        plan,
        root=tmp_path,
        run_id="native-first-attempt",
        approved=True,
        inventory=items,
        budget_scope="transport-window",
        transports=transports,
    )
    analysis = reconstruct(tmp_path / "data/raw", run.run_id)
    result = transport_batch(
        analysis,
        endpoint.id,
        producer_source_hash=source,
        resumed=False,
        faults=faults,
        policy=load_transport_policy(),
    )
    assert result.ready and result.n == 400 and not result.qualified
    first_by_request = {}
    for body_hash, status in calls:
        first_by_request.setdefault(body_hash, status)
    assert len(first_by_request) == 400
    records = sorted(analysis.records, key=lambda r: r.item_id)
    expected = tuple(int(first_by_request[content_hash(r.request)] == 200) for r in records)
    assert result.values == expected
    assert sum(result.values) < sum(r.status == 200 for r in records)
    assert len(calls) == sum(len(r.attempts) for r in records) > 400
    assert len(list(RawStore(tmp_path / "data/raw").exchanges())) == 400
    before = [r.model_dump(mode="json") for r in records]
    assert not transport_batch(
        analysis,
        endpoint.id,
        producer_source_hash=source,
        resumed=True,
        faults=faults,
        policy=load_transport_policy(),
    ).ready
    assert [
        r.model_dump(mode="json") for r in sorted(analysis.records, key=lambda r: r.item_id)
    ] == before
    assert implementation_hash() == source
    (tmp_path / "transport-capture-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "candidate": asdict(result),
                "wire_post_calls": len(calls),
                "logical_units": 400,
                "estimate": estimate,
                "first_attempt_successes": sum(result.values),
                "final_http_successes": sum(r.status == 200 for r in records),
                "new_provider_api_calls": 0,
                "new_spend_usd": 0,
                "qualified": False,
            },
            indent=2,
        )
        + "\n"
    )
