"""Two reserved canonical wire windows through the unqualified rolling replay."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import first_count_cut, load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.transport import transport_batch
from servegauge.runner.check import _mock_metadata
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.suites.s1_protocol import protocol_items


@pytest.mark.slow
async def test_two_actual_wire_windows_match_raw_and_candidate_replay(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    source = implementation_hash()
    policy = load_rolling_policy()
    frames, counts, wire_rows = [], [], []
    for index, error_rate in enumerate([0.05, 0.5]):
        endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
        plan = RunPlan(
            plan_id="rolling-native",
            profile="gpt-oss-120b",
            level="core",
            seed=1973 + index,
            reference={"tier": "MOCK"},
            max_usd=0,
            endpoints=[endpoint],
            suites=["S1"],
            requests_per_minute=600000,
        )
        seed = int(content_hash([plan.seed, endpoint.id])[:15], 16)
        faults = FaultConfig(seed=seed, error_rate=error_rate)
        service = create_app(seed=seed, faults=faults, sampling_version="v2", context_length=512)
        calls = []

        @service.middleware("http")
        async def observe(request, call_next, observed=calls):
            body = await request.json() if request.method == "POST" else None
            response = await call_next(request)
            if body is not None:
                observed.append((content_hash(body), response.status_code))
            return response

        transports = {endpoint.id: httpx.ASGITransport(app=service)}
        await _mock_metadata(plan, transports)
        items = [
            i for i in protocol_items("core", plan.seed) if i.expected.get("check_id") == "P01"
        ]
        estimate = estimate_plan(plan, items_override=items)
        assert not calls and estimate["total_reserved_cost_usd"] == "0"
        assert estimate["cost_table"][0]["n_requests"] == 400
        run = await execute_plan(
            plan,
            root=tmp_path,
            run_id=f"rolling-wire-{index}",
            approved=True,
            inventory=items,
            budget_scope=f"rolling-window-{index}",
            transports=transports,
        )
        analysis = reconstruct(tmp_path / "data/raw", run.run_id)
        batch = transport_batch(
            analysis,
            endpoint.id,
            producer_source_hash=source,
            resumed=False,
            faults=faults,
            policy=policy,
        )
        assert batch.ready and batch.n == 400 and not batch.qualified
        by_request = {}
        for request_hash, status in calls:
            by_request.setdefault(request_hash, status)
        records = sorted(analysis.records, key=lambda r: r.item_id)
        assert len(by_request) == 400
        assert batch.values == tuple(
            int(by_request[content_hash(r.request)] == 200) for r in records
        )
        assert len(calls) == sum(len(r.attempts) for r in records)
        counts.append(sum(batch.values))
        wire_rows.append(
            {
                "run_id": run.run_id,
                "first_successes": counts[-1],
                "final_successes": sum(r.status == 200 for r in records),
                "wire_attempts": len(calls),
                "batch": asdict(batch),
                "estimate": estimate,
            }
        )
        start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
        frames.append(
            TemporalFrame(
                "wire-stream",
                f"wire-window-{index}",
                "daily",
                start,
                start + timedelta(days=1),
                source,
                batch,
            )
        )
    rows = replay_candidates(frames, policy, expected_source_hash=source)
    cut = first_count_cut(counts, 400, policy)
    assert cut and rows[1].event and rows[1].event.difference == cut.difference
    assert rows[1].event.ci_high < 0 and all(not r.qualified for r in rows)
    assert rows == replay_candidates(
        list(reversed(frames)) + frames, policy, expected_source_hash=source
    )
    assert implementation_hash() == source
    (tmp_path / "rolling-wire-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "wire": wire_rows,
                "candidate_steps": [asdict(r) for r in rows],
                "qualified": False,
                "new_spend_usd": 0,
                "provider_api_calls": 0,
            },
            indent=2,
        )
        + "\n"
    )
