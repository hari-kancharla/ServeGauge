import json
from pathlib import Path

import httpx
import pytest

from servegauge.core.ids import safe_filename
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.check import expand_requirements, normalize_tests, run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.raw_store import RawStore


def check_plan(*, baseline=True):
    endpoints = [resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")]
    if baseline:
        endpoints.append(
            resolve_endpoint(
                "http://127.0.0.1:8012/v1", "mock", "gpt-oss-120b", identifier="baseline"
            )
        )
    return RunPlan.model_validate(
        {
            "plan_id": "check-test",
            "profile": "gpt-oss-120b",
            "level": "smoke",
            "seed": 8,
            "reference": {"tier": "BASELINE"},
            "max_usd": 0,
            "endpoints": endpoints,
            "suites": ["S0", "S1"],
            "requests_per_minute": 60000,
            "overrides": {
                "S1": {
                    "invariant_trials": 2,
                    "diversity_prompts": 2,
                    "diversity_repeats": 3,
                    "effort_items": 3,
                    "seed_trials": 2,
                }
            },
        }
    )


@pytest.fixture(autouse=True)
def budget(tmp_path):
    path = tmp_path / "configs"
    path.mkdir()
    (path / "budgets.yaml").write_text(Path("configs/budgets.yaml").read_text())


async def test_faulted_check_performs_real_reserved_replay(tmp_path: Path) -> None:
    plan = check_plan()
    transports = {
        plan.endpoints[0].id: httpx.ASGITransport(
            app=create_app(faults=FaultConfig(ignore_max_tokens=True), context_length=512)
        ),
        plan.endpoints[1].id: httpx.ASGITransport(app=create_app(seed=2, context_length=512)),
    }
    result = await run_check(
        plan,
        tests="P01",
        root=tmp_path,
        approved=True,
        baseline_id=plan.endpoints[1].id,
        transports=transports,
    )
    assert result.code == 1
    finding = next(r for r in result.checks if r.check_id == "P01")
    assert finding.verdict == "FAIL" and finding.violations == finding.n == 2
    assert finding.details["replay_confirmed"] and finding.ci_high > 0
    assert result.report_dir.joinpath("checks.md").exists()
    run_dir = tmp_path / "data/runs" / safe_filename(result.run_id)
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    replay_id = evaluation["replay_runs"]["P01"]
    replay_records = list(RawStore(tmp_path / "data/raw").exchanges(run_id=replay_id))
    assert len(replay_records) == 1 and replay_records[0].attempts
    replay_manifest = json.loads(
        (tmp_path / "data/runs" / safe_filename(replay_id) / "manifest.json").read_text()
    )
    assert replay_manifest["budget_scope"] == result.run_id
    assert evaluation["synthetic_mock"]


async def test_no_reference_smoke_and_dry_run_are_honest(tmp_path: Path) -> None:
    plan = check_plan(baseline=False)
    transports = {plan.endpoints[0].id: httpx.ASGITransport(app=create_app(context_length=512))}
    dry = await run_check(plan, root=tmp_path, dry_run=True)
    assert dry.checks == [] and not (tmp_path / "data/raw").exists()
    with pytest.raises(BudgetExceeded, match="explicit_yes"):
        await run_check(plan, root=tmp_path)
    result = await run_check(plan, tests="P03", root=tmp_path, approved=True, transports=transports)
    check = next(r for r in result.checks if r.check_id == "P03")
    assert result.code == 2 and check.reason == "no baseline or reference"
    assert check.n == 2 and check.ci_low is not None
    assert next(r for r in result.checks if r.check_id == "P14").verdict == "NOT_APPLICABLE"


def test_selection_and_live_identity_fail_closed():
    assert normalize_tests("protocol") == {"S1"}
    assert normalize_tests("fingerprint") == {"S6"}
    for invalid in ["P08", "not-a-check"]:
        with pytest.raises(ValueError):
            normalize_tests(invalid)
    for url in ["https://user:password@example.com/v1", "https://example.com/v1?api_key=private"]:
        with pytest.raises(ValueError):
            resolve_endpoint(url, "model", "gpt-oss-120b")
    from servegauge.runner.plan import Requirements

    requirements = Requirements.model_validate(
        {"app": "test", "requires": [{"id": "choice", "tests": ["P09"]}]}
    )
    expanded = expand_requirements(requirements)
    assert expanded.requires[0].tests == ["P09.named", "P09.none", "P09.required"]
    requirements.requires[0].params = {"unknown": True}
    with pytest.raises(ValueError, match="known_positive_context_requirement"):
        expand_requirements(requirements)


def test_context_requirement_maps_to_measured_qualifying_length():
    from servegauge.runner.plan import Requirements

    requirement = Requirements.model_validate(
        {
            "app": "context",
            "requires": [
                {
                    "id": "capacity",
                    "tests": ["L04", "L05"],
                    "params": {"min_effective_context": 64000},
                }
            ],
        }
    )
    assert expand_requirements(requirement).requires[0].tests == ["L04.64000", "L05.64000"]
    with pytest.raises(ValueError, match="not_covered"):
        expand_requirements(requirement, level="smoke")


async def test_saved_baseline_pack_is_used_at_matched_items_and_n(tmp_path: Path) -> None:
    from servegauge.refs.pack import ReferenceRun, build_pack

    plan = check_plan()
    supplied = {
        endpoint.id: httpx.ASGITransport(app=create_app(seed=index, context_length=512))
        for index, endpoint in enumerate(plan.endpoints, 1)
    }
    first = await run_check(
        plan,
        tests="P03",
        root=tmp_path,
        approved=True,
        baseline_id=plan.endpoints[1].id,
        transports=supplied,
    )
    path = tmp_path / "data/runs" / safe_filename(first.run_id) / "outcomes.json"
    source = ReferenceRun.model_validate_json(path.read_bytes())
    pack = build_pack(source, "BASELINE", seed=8)
    solo = plan.model_copy(update={"endpoints": plan.endpoints[:1]})
    second = await run_check(
        solo,
        tests="P03",
        root=tmp_path,
        approved=True,
        reference_pack=pack,
        transports={
            solo.endpoints[0].id: httpx.ASGITransport(app=create_app(seed=3, context_length=512))
        },
    )
    check = next(result for result in second.checks if result.check_id == "P03")
    assert check.reference_tier == "BASELINE" and check.n == 2
    assert check.details["reference_calibration"]["n"] == 2
    assert check.details["reference_unavailable"] is None
