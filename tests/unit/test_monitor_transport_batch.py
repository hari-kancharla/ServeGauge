"""First-attempt units, censoring and provenance guards without operating claims."""

import copy

import pytest

from servegauge.core.hashing import content_hash
from servegauge.core.models import Attempt, Exchange, Timings
from servegauge.mock.faults import FaultConfig
from servegauge.observatory.changes import load_transport_policy
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.transport import transport_batch
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import Analysis
from servegauge.suites.registry import prepare_item
from servegauge.suites.s1_protocol import protocol_items


@pytest.fixture
def captured_batch():
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="candidate",
        profile="gpt-oss-120b",
        level="core",
        seed=1307,
        reference={"tier": "MOCK"},
        max_usd=0,
        endpoints=[endpoint],
        suites=["S1"],
    )
    endpoint = plan.endpoints[0]
    endpoint.metadata["mock_declaration"] = {
        "backend": "scripted",
        "sampling_version": "v2",
        "healthy_logit_jitter": 0.0,
        "seed": int(content_hash([plan.seed, endpoint.id])[:15], 16),
        "clean": False,
    }
    items = [i for i in protocol_items("core", plan.seed) if i.expected.get("check_id") == "P01"]
    records = []
    for index, item in enumerate(items):
        attempts = [
            Attempt(
                n=1, status=503 if index < 10 else 200, timings=Timings(start_unix=0, total_ms=1)
            )
        ]
        if index < 10:
            attempts.append(Attempt(n=2, status=200, timings=Timings(start_unix=1, total_ms=1)))
        records.append(
            Exchange(
                exchange_id=f"unit-{index}",
                run_id="candidate-run",
                endpoint_id=endpoint.id,
                suite_id="S1",
                item_id=item.item_id,
                sample_idx=0,
                request=prepare_item(item, endpoint).body,
                attempts=attempts,
                status=200,
                request_url=endpoint.base_url + "/chat/completions",
            )
        )
    analysis = Analysis(
        plan=plan,
        items=items,
        records=records,
        checks=[],
        samples={},
        metadata={"synthetic_mock": True},
        manifest={"run_id": "candidate-run"},
        reference_pack=None,
    )
    return (
        analysis,
        endpoint.id,
        {
            "producer_source_hash": implementation_hash(),
            "resumed": False,
            "faults": FaultConfig(error_rate=0.05),
            "policy": load_transport_policy(),
        },
    )


def test_first_attempts_are_not_replaced_by_retry_success(captured_batch):
    analysis, endpoint, arguments = captured_batch
    result = transport_batch(analysis, endpoint, **arguments)
    assert result.ready and result.qualified is False and result.verdict == "INCONCLUSIVE"
    assert result.n == 400 and result.observed_units == 400 and sum(result.values) == 390
    assert result.values[:10] == (0,) * 10 and result.values[10:] == (1,) * 390
    assert result.ci_low < 390 / 400 < result.ci_high
    assert result.reference_tier == "MOCK" and result.request_design_hash
    assert all(r.status == 200 for r in analysis.records)


def test_continuous_policy_keeps_identical_first_attempt_units(captured_batch):
    from servegauge.observatory.rolling import load_rolling_policy

    analysis, endpoint, arguments = captured_batch
    original = transport_batch(analysis, endpoint, **arguments)
    arguments["policy"] = load_rolling_policy()
    assert transport_batch(analysis, endpoint, **arguments) == original


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "repeat",
        "attempt_missing",
        "attempt_number",
        "status_unknown",
        "request_changed",
        "item_changed",
        "resumed",
        "resume_unknown",
        "source_changed",
        "source_unknown",
        "real_endpoint",
        "other_fault",
        "wrong_seed",
        "missing_declaration",
        "malformed_declaration",
        "unsupported_http_status",
        "wrong_tier",
        "synthetic_marker_missing",
        "defaults_changed",
    ],
)
def test_unsupported_incomplete_or_changed_evidence_abstains(captured_batch, case):
    analysis, endpoint, arguments = captured_batch
    analysis = copy.deepcopy(analysis)
    if case == "missing":
        analysis.records.pop()
    elif case == "duplicate":
        analysis.records.append(analysis.records[0])
    elif case == "repeat":
        analysis.records[0].sample_idx = 1
    elif case == "attempt_missing":
        analysis.records[0].attempts = []
    elif case == "attempt_number":
        analysis.records[0].attempts[0].n = 2
    elif case == "status_unknown":
        analysis.records[0].attempts[0].status = None
    elif case == "request_changed":
        analysis.records[0].request["max_tokens"] = 17
    elif case == "item_changed":
        analysis.items[0].payload["params"]["max_tokens"] = 17
    elif case == "resumed":
        arguments["resumed"] = True
    elif case == "resume_unknown":
        arguments["resumed"] = None
    elif case == "source_changed":
        arguments["producer_source_hash"] = "different"
    elif case == "source_unknown":
        arguments["producer_source_hash"] = None
    elif case == "real_endpoint":
        analysis.plan.endpoints[0].kind = "openai_compat"
    elif case == "other_fault":
        arguments["faults"] = FaultConfig(error_rate=0.05, stale_cache=0.2)
    elif case == "wrong_seed":
        analysis.plan.endpoints[0].metadata["mock_declaration"]["seed"] = 1
    elif case == "missing_declaration":
        analysis.plan.endpoints[0].metadata.pop("mock_declaration")
    elif case == "malformed_declaration":
        analysis.plan.endpoints[0].metadata["mock_declaration"] = None
    elif case == "unsupported_http_status":
        analysis.records[0].attempts[0].status = 429
    elif case == "wrong_tier":
        analysis.plan.reference.tier = "R3"
    elif case == "synthetic_marker_missing":
        analysis.metadata.pop("synthetic_mock")
    elif case == "defaults_changed":
        analysis.plan.endpoints[0].defaults["temperature"] = 0
    result = transport_batch(analysis, endpoint, **arguments)
    assert not result.ready and not result.qualified and result.verdict == "INCONCLUSIVE"
    assert not result.values and result.reason


def test_diagnostic_does_not_mutate_evidence_or_use_fault_labels(captured_batch):
    analysis, endpoint, arguments = captured_batch
    before = copy.deepcopy(analysis)
    first = transport_batch(analysis, endpoint, **arguments)
    arguments["faults"].fault_id = "unrelated-label"
    second = transport_batch(analysis, endpoint, **arguments)
    assert first == second and analysis == before
