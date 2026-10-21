import copy

import pytest

from servegauge.core.hashing import content_hash
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.refs.fingerprint import captured_controls, compare_fingerprint
from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun, build_pack, load_pack, save_pack
from servegauge.scoring.fingerprint_collection import fingerprint_samples
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import load_policy


def source_run(name, seeds, control=None):
    policy = load_policy()
    items = fingerprint_items(
        "core",
        88,
        {
            "methods": ["FP1", "FP2", "FP3"],
            "fp1_cells": 2,
            "fp1_samples": 16,
            "fp2_cells": 2,
            "fp2_samples": 16,
            "fp3_samples": 32,
        },
    )
    fingerprint = {}
    endpoints = []
    declarations = []
    for seed in seeds:
        ep = endpoint(f"mock:{name}-{seed}", seed, clean=control is None)
        ep.metadata["mock_declaration"]["backend"] = "other:" + control if control else "scripted"
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
        backend = ScriptedBackend(seed, FaultConfig(), model_control=control, sampling_version="v2")
        rows = captures(items, backend, ep, name)
        fingerprint[ep.id] = fingerprint_samples(items, rows, ep, policy)
        endpoints.append(
            ReferenceEndpoint(
                id=ep.id, profile=ep.claimed_profile, kind="mock", seed=seed, clean=control is None
            )
        )
        declarations.append(ep.model_dump(mode="json"))
    return ReferenceRun(
        run_id=name,
        profile="gpt-oss-120b",
        configuration={"policy_hash": content_hash(policy.model_dump(mode="json"))},
        item_set_hashes={"S6": content_hash([i.model_dump(mode="json") for i in items])},
        endpoints=endpoints,
        checks={},
        fingerprints=fingerprint,
        fingerprint_policy=policy,
        manifest={"endpoint_metadata": declarations},
    )


def sources():
    return source_run("reference", [1, 2, 3]), [
        source_run("control-a", [11], "control-a"),
        source_run("control-b", [12], "control-b"),
    ]


@pytest.mark.parametrize("reused_seed", [1, 11])
def test_known_mock_random_stream_reuse_abstains_even_without_equivalence(reused_seed):
    from servegauge.core.models import EndpointSpec

    source, controls = sources()
    pack = build_pack(source, "MOCK", seed=8, controls=controls)
    target = source_run("different-run-reused-stream", [reused_seed])
    endpoint = EndpointSpec.model_validate(target.manifest["endpoint_metadata"][0])
    with pytest.raises(ValueError, match="mock_target_seed_must_be_independent"):
        compare_fingerprint(
            pack,
            target.fingerprints[endpoint.id]["FP1"],
            target=endpoint.id,
            method="FP1",
            policy=load_policy(),
            seed=8,
            endpoint=endpoint,
        )


def test_fingerprint_only_pack_uses_real_captured_controls_and_preserves_inputs(tmp_path):
    source, controls = sources()
    original = copy.deepcopy((source, controls))
    pack = build_pack(source, "MOCK", seed=8, controls=controls)
    assert pack.fingerprints["version"] and pack.fingerprints["views"]["*"]["metrics"]
    assert len(pack.fingerprints["controls"]) == 2
    assert all(v["n"] >= 30 for v in pack.fingerprints["views"]["*"]["metrics"].values())
    assert (source, controls) == original
    path = save_pack(pack, tmp_path)
    assert load_pack(str(path)) == pack and save_pack(pack, tmp_path) == path


@pytest.mark.parametrize(
    "case",
    [
        "missing_control",
        "duplicate_control",
        "same_backend",
        "missing_declaration",
        "shared_capture",
        "descriptor_mismatch",
    ],
)
def test_unqualified_controls_are_refused_instead_of_invented(case):
    source, controls = sources()
    if case == "missing_control":
        controls = controls[:1]
    elif case == "duplicate_control":
        controls = [controls[0], controls[0]]
    elif case == "same_backend":
        controls[1].manifest["endpoint_metadata"][0]["metadata"]["mock_declaration"]["backend"] = (
            "other:control-a"
        )
    elif case == "missing_declaration":
        controls[1].manifest = {}
    elif case == "descriptor_mismatch":
        controls[1].manifest["endpoint_metadata"][0]["kind"] = "selfhosted"
    else:
        a = next(iter(controls[0].fingerprints.values()))["FP1"].details["observations"][0]
        b = next(iter(source.fingerprints.values()))["FP1"].details["observations"][0]
        a["exchange_id"] = b["exchange_id"]
    with pytest.raises(ValueError):
        build_pack(source, "MOCK", seed=8, controls=controls)


def test_reference_calibration_is_unaffected_by_target_data():
    source, controls = sources()
    source.fingerprints["mock:future-target"] = copy.deepcopy(
        next(iter(source.fingerprints.values()))
    )
    source.endpoints.append(
        ReferenceEndpoint(
            id="mock:future-target", profile=source.profile, kind="mock", seed=99, clean=False
        )
    )
    before = build_pack(source, "MOCK", seed=8, controls=controls)
    source.fingerprints["mock:future-target"]["FP1"].details["observations"][0]["text"] = (
        "unseen target answer"
    )
    after = build_pack(source, "MOCK", seed=8, controls=controls)
    assert before.fingerprints["views"] == after.fingerprints["views"]


@pytest.fixture(scope="module")
def qualified_pack():
    source, controls = sources()
    return build_pack(source, "MOCK", seed=8, controls=controls)


def target_sample():
    source = source_run("fresh-target", [104])
    return next(iter(source.fingerprints.values()))["FP1"]


def test_fresh_target_returns_intervals_and_conservative_orientation_test(qualified_pack):
    sample = target_sample()
    before = copy.deepcopy((qualified_pack, sample))
    result = compare_fingerprint(
        qualified_pack, sample, target="fresh", method="FP1", policy=load_policy(), seed=33
    )
    assert result["n"] == 32
    assert result["delta_ci_low"] <= result["delta_ci_high"]
    assert result["p_value"] == max(result["orientation_p_values"])
    assert result["equivalence_qualified"] is False
    assert result["position"] in {"same-model-band", "between-bands", "calibration-overlap"}
    positions = result["control_positions"]
    assert len(positions) == len(qualified_pack.fingerprints["controls"]) == 2
    for position, control in zip(positions, qualified_pack.fingerprints["controls"], strict=True):
        assert position["model_identity"] == control["model_identity"]
        assert position["capture_hash"] == control["capture_hash"]
        assert len(position["different_model_band"]) == 2
    assert (qualified_pack, sample) == before


@pytest.mark.parametrize("case", ["unknown_design", "changed_n", "reused", "tampered", "policy"])
def test_target_comparison_refuses_unqualified_evidence(qualified_pack, case):
    sample = target_sample()
    policy = load_policy()
    if case == "unknown_design":
        sample.details["sampling_design"] = {}
    elif case == "changed_n":
        sample.details["design"]["sizes"]["fp1_samples"] += 1
    elif case == "reused":
        sample = next(iter(qualified_pack.fingerprints["views"]["*"]["references"].values()))["FP1"]
        from servegauge.scoring.common import Samples

        sample = Samples.model_validate(sample)
    elif case == "tampered":
        sample.details["observations"][0]["text"] = "changed"
    else:
        policy.alpha = 0.01
    with pytest.raises(ValueError):
        compare_fingerprint(
            qualified_pack, sample, target="fresh", method="FP1", policy=policy, seed=33
        )


def test_low_effective_n_and_missing_logprobs_are_explicit_unavailable_metrics():
    source, controls = sources()
    # Preserve nominal draws but declare their actual dependent grouping.
    for run in [source, *controls]:
        for methods in run.fingerprints.values():
            for sample in methods.values():
                sample.details["sampling_design"]["session_size"] = 2
    pack = build_pack(source, "MOCK", seed=8, controls=controls)
    view = pack.fingerprints["views"]["*"]
    assert not view["metrics"]
    assert set(view["unavailable"]) == {"FP1", "FP2", "FP3-mean", "FP3-dispersion"}
    assert all("insufficient" in reason for reason in view["unavailable"].values())

    source, controls = sources()
    sample = next(iter(controls[0].fingerprints.values()))["FP3"]
    sample.details["observations"][0]["response"]["choices"][0]["logprobs"] = None
    sample.details["observation_hash"] = content_hash(sample.details["observations"])
    pack = build_pack(source, "MOCK", seed=8, controls=controls)
    view = pack.fingerprints["views"]["*"]
    assert set(view["metrics"]) == {"FP1", "FP2"}
    assert set(view["unavailable"]) == {"FP3-mean", "FP3-dispersion"}


def test_r3_retains_only_other_captures_and_baseline_requires_three():
    source = source_run("r3-reference", [1, 2, 3, 4])
    _, controls = sources()
    pack = build_pack(source, "R3", seed=8, controls=controls)
    for target, view in pack.fingerprints["views"].items():
        assert target not in view["references"] and len(view["references"]) == 3
        assert all(target not in row for m in view["metrics"].values() for row in m["orientations"])
        result = compare_fingerprint(
            pack,
            source.fingerprints[target]["FP1"],
            target=target,
            method="FP1",
            policy=load_policy(),
            seed=33,
        )
        assert result["n"] == 32
    source.baseline_endpoint_id = source.endpoints[0].id
    with pytest.raises(ValueError, match="three_independent"):
        build_pack(source, "BASELINE", seed=8, controls=controls)


def test_selected_reference_and_model_declarations_are_required():
    source, controls = sources()
    source.fingerprints.pop(source.endpoints[0].id)
    with pytest.raises(ValueError, match="captures_missing"):
        build_pack(source, "MOCK", seed=8, controls=controls)
    source, controls = sources()
    source.fingerprint_policy = None
    with pytest.raises(ValueError, match="policy_required"):
        build_pack(source, "MOCK", seed=8, controls=controls)


def combined_source():
    source, controls = sources()
    for control in controls:
        source.endpoints.extend(copy.deepcopy(control.endpoints))
        source.fingerprints.update(copy.deepcopy(control.fingerprints))
        source.manifest["endpoint_metadata"].extend(
            copy.deepcopy(control.manifest["endpoint_metadata"])
        )
    return source


def test_same_run_captured_controls_are_discovered_without_labels_or_new_data():
    source = combined_source()
    before = copy.deepcopy(source)
    pack = build_pack(source, "MOCK", seed=8)
    assert len(pack.fingerprints["controls"]) == 2
    assert all(m["n"] == 32 for m in pack.fingerprints["views"]["*"]["metrics"].values())
    assert source == before
    assert {c["model_identity"] for c in pack.fingerprints["controls"]} == {
        "mock:other:control-a",
        "mock:other:control-b",
    }
    assert all(
        not p.startswith("mock:control") for p in pack.fingerprints["views"]["*"]["references"]
    )


def test_automatic_controls_choose_one_predetermined_capture_per_model_and_snapshot():
    source = combined_source()
    duplicate = source_run("extra-control-a", [31], "control-a")
    source.endpoints.extend(duplicate.endpoints)
    source.fingerprints.update(duplicate.fingerprints)
    source.manifest["endpoint_metadata"].extend(duplicate.manifest["endpoint_metadata"])
    before = copy.deepcopy(source)
    controls = captured_controls(source, "MOCK")
    assert len(controls) == 2
    next(iter(controls[0].fingerprints.values()))["FP1"].details["observations"][0]["text"] = (
        "changed"
    )
    assert source == before
    pack = build_pack(source, "MOCK", seed=8)
    assert len(pack.fingerprints["controls"]) == 2


def test_mock_seed_reuse_or_inconsistent_declarations_refuse_qualification():
    for inconsistent in (False, True):
        source, controls = sources()
        controls[0].endpoints[0].seed = 1
        if not inconsistent:
            controls[0].manifest["endpoint_metadata"][0]["metadata"]["mock_declaration"]["seed"] = 1
        with pytest.raises(ValueError, match="mock"):
            build_pack(source, "MOCK", seed=8, controls=controls)


def baseline_sources():
    source = source_run("trusted", [1])
    source.baseline_endpoint_id = source.endpoints[0].id
    references = [source_run("repeat-a", [2]), source_run("repeat-b", [3])]
    _, controls = sources()
    return source, references, controls


def test_trusted_baseline_uses_three_independent_captures_and_keeps_provenance():
    source, references, controls = baseline_sources()
    original = copy.deepcopy((source, references, controls))
    pack = build_pack(source, "BASELINE", seed=8, controls=controls, reference_runs=references)
    view = pack.fingerprints["views"]["*"]
    assert len(view["references"]) == 3 and len(view["metrics"]) == 4
    assert {s["run_id"] for s in pack.fingerprints["reference_sources"].values()} == {
        "trusted",
        "repeat-a",
        "repeat-b",
    }
    assert all(m["n"] == 32 for m in view["metrics"].values())
    target = next(iter(source_run("future-baseline-target", [104]).fingerprints.values()))["FP1"]
    comparison = compare_fingerprint(
        pack, target, target="future", method="FP1", policy=load_policy(), seed=33
    )
    assert comparison["n"] == 32
    assert (source, references, controls) == original


@pytest.mark.parametrize(
    "case", ["one", "reused", "duplicate", "url", "model", "pin", "seed", "policy", "ambiguous"]
)
def test_baseline_refuses_missing_reused_changed_or_ambiguous_reference_captures(case):
    source, references, controls = baseline_sources()
    if case == "one":
        references = references[:1]
    elif case == "reused":
        references[1] = source
    elif case == "duplicate":
        references[1] = references[0]
    elif case == "url":
        references[0].manifest["endpoint_metadata"][0]["base_url"] = "http://127.0.0.2/v1"
    elif case == "model":
        references[0].manifest["endpoint_metadata"][0]["model"] = "other-model"
    elif case == "pin":
        references[0].manifest["endpoint_metadata"][0]["openrouter_pin"] = {"tag": "other/route"}
    elif case == "seed":
        references[0].endpoints[0].seed = 1
        references[0].manifest["endpoint_metadata"][0]["metadata"]["mock_declaration"]["seed"] = 1
    elif case == "policy":
        references[0].fingerprint_policy.alpha = 0.01
    else:
        additional = source_run("extra", [20])
        references[0].endpoints.extend(additional.endpoints)
        references[0].fingerprints.update(additional.fingerprints)
        references[0].manifest["endpoint_metadata"].extend(additional.manifest["endpoint_metadata"])
    with pytest.raises(ValueError):
        build_pack(source, "BASELINE", seed=8, controls=controls, reference_runs=references)


def test_capture_selection_does_not_turn_design_changes_into_calibration():
    source, references, controls = baseline_sources()
    for methods in references[0].fingerprints.values():
        methods["FP1"].details["design"]["sizes"]["fp1_samples"] += 1
    pack = build_pack(source, "BASELINE", seed=8, controls=controls, reference_runs=references)
    view = pack.fingerprints["views"]["*"]
    assert "FP1" in view["unavailable"] and "FP1" not in view["metrics"]
    assert "FP2" in view["metrics"]
