"""M15 source-scope gate only; these fixtures do not qualify a real reference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import EndpointSpec, OpenRouterPin
from servegauge.runner.plan import RunPlan
from servegauge.storage.comparison import ComparisonSelection, SourcePin
from servegauge.storage.comparison_kit import _source_scope


def source_scope(tmp_path, monkeypatch, *, providers=8, problem=None):
    endpoints = [
        EndpointSpec(
            id="route-" + str(i),
            kind="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-oss-120b",
            claimed_profile="gpt-oss-120b",
            openrouter_pin=OpenRouterPin(tag=f"provider-{i % providers}/route-{i}"),
            labels={
                "provider_name": f"Provider {i % providers}",
                "quantization": "bf16" if i % 2 else "fp8",
            },
        )
        for i in range(8)
    ]
    suites = ["S0", "S1", "S2", "S3", "S4", "S7"]
    if problem in {"unknown_label", "one_label"}:
        for target in endpoints:
            target.labels["quantization"] = "bf16"
        if problem == "unknown_label":
            endpoints[0].labels["quantization"] = "unknown"
    if problem == "duplicate":
        endpoints[-1].id = endpoints[0].id
    if problem == "model":
        endpoints[0].model = "different-model"
    if problem == "fallback":
        endpoints[0].openrouter_pin.allow_fallbacks = True
    plan = RunPlan(
        plan_id="scope-only",
        profile="gpt-oss-120b",
        level="core",
        seed=1,
        max_usd=0,
        reference={"tier": "R3"},
        endpoints=endpoints,
        suites=suites,
    )
    targets = ["route-" + str(i) for i in range(7 if problem == "seven" else 8)]
    records = {
        str(i): SimpleNamespace(
            endpoint_id=e.id, run_id="round", suite_id=s, status=200, error=None, cost_usd=0
        )
        for i, (e, s) in enumerate(
            (e, s)
            for e in endpoints
            for s in suites
            if not (problem == "missing_suite" and s == "S7")
        )
    }
    source = SimpleNamespace(
        analysis=SimpleNamespace(
            plan=plan,
            metadata={"target_ids": targets},
            reference_pack=None if problem == "reference" else SimpleNamespace(tier="R3"),
        ),
        records=records,
    )
    value = {"manifest": {"config": plan.model_dump(mode="json")}}
    blob = canonical_json({"value": value, "hash": content_hash(value)})
    name = safe_filename("round") + "/capture.json"
    import hashlib

    pin = SourcePin(run_id="round", files_sha256={name: hashlib.sha256(blob).hexdigest()})
    path = tmp_path / "sources" / safe_filename("round") / "data/raw" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(blob)
    monkeypatch.setattr("servegauge.storage.comparison_kit._load", lambda *a, **k: source)
    # Selection validation belongs to the sealed loader; this fixture tests only
    # the derived scope cardinalities and declared provider coverage.
    selection = ComparisonSelection.model_construct(sources=[pin], cases=[])
    return _source_scope(tmp_path, selection)


def test_eight_distinct_routes_with_distinct_provider_and_label_coverage_are_eligible(
    tmp_path, monkeypatch
):
    synthetic, scopes = source_scope(tmp_path, monkeypatch, providers=8)
    assert synthetic is False and scopes["round"]["eligible_source_scope"] is True


def test_scope_does_not_invent_an_eight_unique_provider_requirement(tmp_path, monkeypatch):
    _, scopes = source_scope(tmp_path, monkeypatch, providers=7)
    assert scopes["round"]["eligible_source_scope"] is True


def test_single_provider_does_not_supply_distinct_provider_coverage(tmp_path, monkeypatch):
    _, scopes = source_scope(tmp_path, monkeypatch, providers=1)
    assert scopes["round"]["eligible_source_scope"] is False


def test_unknown_is_a_disclosed_label_category_not_a_weight_verification(tmp_path, monkeypatch):
    _, scopes = source_scope(tmp_path, monkeypatch, problem="unknown_label")
    assert scopes["round"]["eligible_source_scope"] is True
    assert scopes["round"]["declared_quantization_is_not_verified_weights"] is True


def test_duplicate_source_routes_are_rejected_by_the_plan_before_scope_evaluation(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="unique"):
        source_scope(tmp_path, monkeypatch, problem="duplicate")


@pytest.mark.parametrize(
    "problem", ["seven", "one_label", "model", "fallback", "missing_suite", "reference"]
)
def test_incomplete_or_ambiguous_recorded_scope_cannot_become_m15_eligible(
    tmp_path, monkeypatch, problem
):
    _, scopes = source_scope(tmp_path, monkeypatch, problem=problem)
    assert scopes["round"]["eligible_source_scope"] is False
