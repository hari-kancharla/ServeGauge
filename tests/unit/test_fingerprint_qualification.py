import copy
import json
from pathlib import Path

import pytest

from servegauge.core.models import EndpointSpec
from servegauge.refs.fingerprint_qualification import (
    load_qualification,
    qualify_data,
    target_binding,
)
from servegauge.scoring.common import Samples
from servegauge.verdicts.config import DecisionPolicy

pytestmark = pytest.mark.qualified_runtime


@pytest.fixture(scope="module")
def captured():
    value = json.loads(
        Path("docs/report/fingerprint-wire-v1/calibration/pack-00.json").read_text()
    )["value"]
    source = value["sources"][0]
    refs = list(source["fingerprints"].values())
    controls = [next(iter(s["fingerprints"].values())) for s in value["sources"][1:]]
    declarations = source["manifest"]["endpoint_metadata"] + [
        s["manifest"]["endpoint_metadata"][0] for s in value["sources"][1:]
    ]
    return (
        value,
        [
            [{m: Samples.model_validate(v) for m, v in group.items()} for group in part]
            for part in [refs, controls]
        ],
        [EndpointSpec.model_validate(d) for d in declarations],
    )


@pytest.fixture(scope="module")
def fresh_target():
    from servegauge.validation.fingerprint_wire import wire_items
    from servegauge.validation.fingerprint_wire_capture import capture_run

    value = json.loads(Path("docs/report/fingerprint-wire-v1/wire.json").read_text())["value"]
    cfg = value["configuration"]
    policy = DecisionPolicy.model_validate(cfg["policy"])
    source, _ = capture_run(
        wire_items(cfg["design"], cfg["wire"]),
        policy,
        seeds=[202610192731],
        source_id="qualification-unit-independent",
        healthy_jitter=0.3,
    )
    endpoint = EndpointSpec.model_validate(source.manifest["endpoint_metadata"][0])
    return source.fingerprints[endpoint.id]["FP1"], endpoint


def test_admitted_evidence_is_pinned_and_exact_scope_is_explicit():
    proof = load_qualification("captured-scripted-text-v1")
    assert proof["family"] == ["FP1", "FP2", "FP3"]
    assert proof["n"] == {"FP1": 3600, "FP2": 750, "FP3": 400}
    assert proof["margins"] == {"FP1": 0.03, "FP2": 0.03}


def test_qualification_preserves_numbers_and_does_not_mutate_original(captured, fresh_target):
    original, (refs, controls), declarations = captured
    before = copy.deepcopy(original["pack"]["fingerprints"])
    result = qualify_data(
        before, refs, controls, declarations, tier="MOCK", name="captured-scripted-text-v1"
    )
    assert before == original["pack"]["fingerprints"]
    metrics = result["views"]["*"]["metrics"]
    for method in ["FP1", "FP2"]:
        assert metrics[method]["equivalence_qualified"] is True
        assert (
            metrics[method]["calibration"] == before["views"]["*"]["metrics"][method]["calibration"]
        )
    assert metrics["FP3-mean"]["equivalence_qualified"] is False
    policy = DecisionPolicy.model_validate(original["pack"]["fingerprints"]["policy"])
    binding = target_binding(result, *fresh_target, policy)
    assert binding["family"] == ["FP1", "FP2", "FP3"]


@pytest.mark.parametrize(
    "case",
    [
        "tier",
        "provider",
        "backend",
        "jitter",
        "version",
        "session",
        "cache",
        "counts",
        "requests",
        "missing",
        "policy",
    ],
)
def test_unsupported_capture_cannot_be_qualified(captured, case):
    original, (refs, controls), declarations = captured
    refs = [dict(group) for group in refs]
    refs[0] = {m: s.model_copy(deep=True) for m, s in refs[0].items()}
    declarations = [d.model_copy(deep=True) for d in declarations]
    tier = "MOCK"
    sample = refs[0]["FP1"]
    if case == "tier":
        tier = "R3"
    elif case == "provider":
        declarations[0].kind = "openrouter"
    elif case == "backend":
        declarations[0].metadata["mock_declaration"]["backend"] = "unknown"
    elif case == "jitter":
        declarations[0].metadata["mock_declaration"]["healthy_logit_jitter"] = 0
    elif case == "version":
        declarations[0].metadata["mock_declaration"]["sampling_version"] = "v1"
    elif case == "session":
        sample.details["sampling_design"]["session_size"] = 2
    elif case == "cache":
        sample.details["sampling_design"]["cache_groups"] = {"x": "cache"}
    elif case == "counts":
        sample.details["design"]["sizes"]["fp1_samples"] = 30
    elif case == "requests":
        sample.details["observations"][0]["sampling_request_hash"] = "changed"
    elif case == "missing":
        sample.status = "INCONCLUSIVE"
    else:
        sample.details["policy_hash"] = "changed"
    with pytest.raises(ValueError):
        qualify_data(
            original["pack"]["fingerprints"],
            refs,
            controls,
            declarations,
            tier=tier,
            name="captured-scripted-text-v1",
        )


def test_changed_registry_or_code_pin_is_refused(tmp_path, monkeypatch):
    import servegauge.refs.fingerprint_qualification as module

    value = json.loads(Path("configs/fingerprint_qualification.json").read_text())
    value["implementation"][next(iter(value["implementation"]))] = "0" * 64
    target = tmp_path / "registry.json"
    target.write_text(json.dumps(value))
    monkeypatch.setattr(module, "REGISTRY", target)
    with pytest.raises(ValueError, match="qualification"):
        load_qualification("captured-scripted-text-v1")


def test_target_endpoint_and_valid_binding_are_required(captured, fresh_target):
    original, (refs, controls), declarations = captured
    result = qualify_data(
        original["pack"]["fingerprints"],
        refs,
        controls,
        declarations,
        tier="MOCK",
        name="captured-scripted-text-v1",
    )
    policy = DecisionPolicy.model_validate(result["policy"])
    for endpoint in [None, declarations[0].model_copy(update={"kind": "openrouter"})]:
        with pytest.raises(ValueError):
            target_binding(result, refs[0]["FP1"], endpoint, policy)
    result["qualification"]["binding"] = "changed"
    with pytest.raises(ValueError):
        target_binding(result, *fresh_target, policy)


@pytest.mark.parametrize("case", ["same_seed", "shared_capture"])
def test_target_cannot_reuse_a_reference_random_stream_or_capture(captured, fresh_target, case):
    original, (refs, controls), declarations = captured
    data = qualify_data(
        original["pack"]["fingerprints"],
        refs,
        controls,
        declarations,
        tier="MOCK",
        name="captured-scripted-text-v1",
    )
    sample = fresh_target[0].model_copy(deep=True)
    endpoint = fresh_target[1].model_copy(deep=True)
    if case == "same_seed":
        endpoint.metadata["mock_declaration"]["seed"] = declarations[0].metadata[
            "mock_declaration"
        ]["seed"]
    else:
        sample.evidence[0] = refs[0]["FP1"].evidence[0]
        sample.details["observations"][0]["exchange_id"] = sample.evidence[0]
    with pytest.raises(ValueError, match="independent"):
        target_binding(data, sample, endpoint, DecisionPolicy.model_validate(data["policy"]))


@pytest.mark.parametrize(
    "case", ["supported", "missing_logprobs", "extra_method", "stricter_margin"]
)
def test_equivalence_requires_the_validated_complete_family_and_margin(case):
    from servegauge.core.models import CheckResult
    from servegauge.verdicts.engine import decide_family

    trial = json.loads(
        Path("docs/report/fingerprint-wire-v1/trials/pack-00-clean-0000.json").read_text()
    )["value"]
    results = [CheckResult.model_validate(r["check"]) for r in trial["checks"]]
    proof = load_qualification("captured-scripted-text-v1")
    for result in results[:2]:
        result.reason = "awaiting family decision"
        result.details.update(
            equivalence_qualified=True,
            qualification={k: proof[k] for k in ["id", "binding", "family", "margins", "scope"]},
        )
    if case == "missing_logprobs":
        results[-1].p_value = None
        results[-1].reference_tier = None
    elif case == "extra_method":
        extra = results[-1].model_copy(deep=True)
        extra.check_id = "FP4"
        extra.p_value = 1
        extra.details = {"family_test": True}
        results.append(extra)
    elif case == "stricter_margin":
        results[0].margin = 0.02
        results[0].details["declared_margin"] = 0.02
    policy = DecisionPolicy.model_validate(
        json.loads(Path("docs/report/fingerprint-wire-v1/wire.json").read_text())["value"][
            "configuration"
        ]["policy"]
    )
    final = decide_family([], policy, external=results)
    if case == "supported":
        assert [r.verdict for r in final] == ["PASS", "PASS", "INFO"]
    else:
        assert final[0].verdict == "INCONCLUSIVE"
