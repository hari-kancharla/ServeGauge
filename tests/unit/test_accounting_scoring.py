import pytest
from tests_support import exchange

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import collect, evaluate
from servegauge.runner.plan import RunPlan
from servegauge.scoring.accounting_scoring import accounting_samples
from servegauge.suites.config import suite_config
from servegauge.suites.registry import prepare_item
from servegauge.suites.s7_accounting import accounting_items
from servegauge.verdicts.config import load_policy


def test_accounting_requires_matched_inputs_calibrates_overhead_and_detects_usage():
    endpoints = [
        resolve_endpoint(
            f"http://127.0.0.1:{port}/v1", "mock", "gpt-oss-120b", identifier=str(port)
        )
        for port in (8011, 8012)
    ]
    plan = RunPlan(
        plan_id="accounting",
        profile="gpt-oss-120b",
        endpoints=endpoints,
        suites=["S7"],
        max_usd=0,
        level="core",
        seed=9,
        reference={"tier": "BASELINE"},
    )
    items = accounting_items("core", 9)
    records = []
    for endpoint, faults in zip(
        endpoints, (FaultConfig(misreport_usage_pct=30), FaultConfig()), strict=True
    ):
        engine = ScriptedBackend(1, faults)
        for item in items:
            body = prepare_item(item, endpoint).body
            response = engine.respond(body)
            records.append(
                exchange(
                    endpoint_id=endpoint.id,
                    item_id=item.item_id,
                    request=body,
                    response=response,
                    usage=response["usage"],
                )
            )
    samples = collect(plan, items, records, load_profile(plan.profile), load_policy())
    result = evaluate(
        plan,
        "accounting",
        items,
        samples,
        load_policy(),
        target_ids=[endpoints[0].id],
        baseline_id=endpoints[1].id,
    )[0]
    assert result.verdict == "FAIL" and result.n == 200
    assert result.reference_tier == "BASELINE" and result.delta_ci_low > 0.03
    samples[endpoints[1].id]["B01"].details["input_hashes"] = {}
    unmatched = evaluate(
        plan,
        "accounting",
        items,
        samples,
        load_policy(),
        target_ids=[endpoints[0].id],
        baseline_id=endpoints[1].id,
    )[0]
    assert unmatched.verdict == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "failure", ["missing", "transport", "unsent", "reject", "usage", "redacted", "effort", "opaque"]
)
def test_incomplete_accounting_never_becomes_failure_or_pass(failure):
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    items = accounting_items("smoke", 9, {"items": 1})
    body = prepare_item(items[0], endpoint).body
    response = ScriptedBackend(1, FaultConfig()).respond(body)
    record = exchange(
        endpoint_id=endpoint.id,
        item_id=items[0].item_id,
        request=body,
        response=response,
        usage=response["usage"],
    )
    if failure == "transport":
        record.error = "transport_timeout"
    elif failure == "unsent":
        record.metadata = {"request_not_sent": True}
    elif failure == "reject":
        record.status = 400
    elif failure == "usage":
        record.usage = {}
    elif failure == "redacted":
        from servegauge.core.redaction import REDACTED

        record.request["messages"][0]["content"] = REDACTED
    elif failure == "effort":
        record.request["reasoning_effort"] = None
    elif failure == "opaque":
        record.request["messages"].insert(
            0,
            {
                "role": "assistant",
                "content": "old",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
            },
        )
    samples = accounting_samples(
        items,
        [] if failure == "missing" else [record],
        endpoint.id,
        load_profile("gpt-oss-120b"),
        suite_config()["accounting"],
    )
    assert samples["B01"].status in {"ERROR", "INCONCLUSIVE"} and not samples["B01"].values
