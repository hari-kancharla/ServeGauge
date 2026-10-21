"""Synthetic P06/T06 sources for full matched two-procedure contract acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import run_check
from servegauge.runner.plan import RunPlan
from servegauge.storage.comparison import (
    CaseRequest,
    ComparisonRequest,
    _guard,
    prepare_sources,
    propose_selection,
)
from servegauge.storage.comparison_controls import (
    ControlRequest,
    compile_controls,
    propose_controls,
)
from servegauge.storage.comparison_program import ComparisonProgram, compile_program
from servegauge.storage.workspace import initialize
from servegauge.suites.s0_capabilities import capability_items
from servegauge.suites.s1_protocol import probe
from servegauge.suites.s2_tools import tool_items
from servegauge.validation.capture import endpoint


class ByteMismatch(httpx.AsyncBaseTransport):
    """Explicit synthetic wire mutation; no real model or performance claim."""

    def __init__(self):
        self.base = httpx.ASGITransport(app=create_app(seed=31))

    async def handle_async_request(self, request):
        response = await self.base.handle_async_request(request)
        body = await response.aread()
        sent = json.loads(request.content) if request.content else {}
        if sent.get("logprobs") is True and not sent.get("stream"):
            value = json.loads(body)
            value["choices"][0]["logprobs"]["content"][0]["bytes"] = [120]
            return httpx.Response(response.status_code, json=value)
        return httpx.Response(response.status_code, content=body, headers=response.headers)


async def prepare_comparison_fixture(
    root: Path, *, trials: int = 4
) -> tuple[Path, ComparisonProgram, ComparisonProgram]:
    _guard(root)
    if root.exists():
        raise ValueError("fresh_comparison_fixture_required")
    workspace = root / "source"
    initialize(workspace)
    requests = []
    for name, check in [("bytes", "P06.structure"), ("tools", "T06")]:
        target, baseline = endpoint(name, 31, clean=False), endpoint("baseline", 32, clean=True)
        plan = RunPlan(
            plan_id="comparison-source-" + name,
            profile="gpt-oss-120b",
            level="core",
            seed=31,
            reference={"tier": "BASELINE"},
            max_usd=0,
            endpoints=[target, baseline],
            suites=["S0", "S1" if name == "bytes" else "S2"],
            concurrency_per_endpoint=1,
            requests_per_minute=1e9,
            retry_base_seconds=0,
            retry_max_seconds=0,
        )
        items = (
            [
                probe(
                    "P06.structure",
                    index,
                    "logprobs",
                    f"Trial {index}. Reply exactly OK.",
                    {"type": "echo", "text": "OK"},
                    params={"max_tokens": 48, "logprobs": True, "top_logprobs": 1},
                    extra={"requires": ["logprobs"]},
                )
                for index in range(trials)
            ]
            if name == "bytes"
            else [
                item
                for item in tool_items(
                    "smoke", 31, {"items": 1, "structural_repeats": trials, "agent_tasks": 1}
                )
                if item.expected.get("probe") == "parallel"
            ]
        )
        caps = capability_items()
        caps.append(
            probe(
                "S0.unavailable",
                0,
                "bounded",
                "Unavailable synthetic capability.",
                {"type": "echo", "text": "unused"},
                extra={"minimum_output_tokens": 1000000, "capability": "bounded_fixture"},
            )
        )
        transport = (
            ByteMismatch()
            if name == "bytes"
            else httpx.ASGITransport(
                app=create_app(seed=31, faults=FaultConfig(stream_delta_without_index=True))
            )
        )
        result = await run_check(
            plan,
            root=workspace,
            approved=True,
            tests=check,
            baseline_id=baseline.id,
            inventory=items,
            capability_inventory=caps,
            transports={
                target.id: transport,
                baseline.id: httpx.ASGITransport(app=create_app(seed=32)),
            },
        )
        finding = next(row for row in result.checks if row.check_id == check)
        assert finding.verdict == "FAIL" and finding.n == trials and finding.violations == trials
        assert finding.details["replay_confirmed"]
        requests.append(
            CaseRequest(
                case_id=name,
                run_id=result.run_id,
                endpoint_id=target.id,
                suite_id=finding.suite_id,
                check_id=check,
            )
        )
    selection = propose_selection(workspace, ComparisonRequest(cases=requests))
    anchor = content_hash(selection.model_dump(mode="json"))
    snapshot = workspace / ".cache/comparisons/reviewed"
    prepare_sources(workspace, selection, snapshot)
    failures = compile_program(snapshot, selection_hash=anchor)
    controls = propose_controls(
        snapshot,
        selection_hash=anchor,
        requests=[
            ControlRequest(
                case_id="control-" + case.case_id,
                failure_case_id=case.case_id,
                endpoint_id="baseline",
            )
            for case in selection.cases
        ],
    )
    control_program = compile_controls(
        snapshot,
        controls,
        selection_hash=anchor,
        controls_hash=content_hash(controls.model_dump(mode="json")),
    )
    (root / "control-selection.json").write_text(controls.model_dump_json(indent=2) + "\n")
    return snapshot, failures, control_program
