import copy

import pytest

from servegauge.core.hashing import content_hash
from servegauge.validation.fingerprint_product import configuration
from servegauge.validation.fingerprint_wire import (
    checked_trial,
    measure_target,
    prepare_pack,
    prerequisites,
    wire_configuration,
    wire_items,
)
from servegauge.verdicts.config import load_policy


def small_configuration():
    cfg = copy.deepcopy(configuration())
    cfg["design"].update(fp1_cells=2, fp1_samples=16, fp2_cells=2, fp2_samples=16, fp3_samples=32)
    return cfg


def test_declared_wire_design_is_fresh_per_pack_and_does_not_weaken_gates():
    cfg = wire_configuration()
    assert cfg["reference_packs"] == 3
    assert cfg["replicates"] == {"clean": 60, "swap": 30, "mix": 30, "qsim": 30}
    assert cfg["minimum_clean_equivalence_lower"] == cfg["minimum_fault_power_lower"] == 0.8
    proof = prerequisites(configuration(), cfg, load_policy())
    assert proof["interval_accepted"] is True and proof["null_accepted"] is True
    assert proof["primitive_files_verified"] == 22


@pytest.fixture(scope="module")
def calibration():
    cfg, wire = small_configuration(), wire_configuration()
    items = wire_items(cfg, wire)
    prepared = prepare_pack(items, cfg, wire, load_policy(), 0)
    return cfg, wire, items, prepared


def test_pack_comes_from_distinct_real_captures_and_two_captured_model_controls(calibration):
    cfg, wire, items, prepared = calibration
    assert prepared["pack"]["tier"] == "MOCK"
    assert len(prepared["capture_receipts"]) == 5
    assert len({r["seed"] for r in prepared["capture_receipts"]}) == 5
    assert sum(r["captures"] for r in prepared["capture_receipts"]) == 480
    assert all(
        m["equivalence_qualified"] is False
        for m in prepared["pack"]["fingerprints"]["views"]["*"]["metrics"].values()
    )
    assert len(prepared["sources"]) == 3


@pytest.mark.parametrize("case", ["clean", "swap", "mix", "qsim"])
def test_actual_trial_preserves_unqualified_checks_and_checkable_capture_hashes(calibration, case):
    cfg, wire, items, prepared = calibration
    row = measure_target(items, cfg, wire, load_policy(), prepared, 0, case, 0)
    assert checked_trial(row, cfg, wire, load_policy(), prepared, 0, case, 0) == row
    assert row["captures"] == 96 and len(row["capture_hash"]) == 64
    assert {c["check"]["check_id"] for c in row["checks"]} == {"FP1", "FP2", "FP3"}
    assert all(c["evidence_count"] > 0 and c["evidence_hash"] for c in row["checks"])
    assert all(c["check"]["verdict"] != "PASS" for c in row["checks"])
    assert "fault" not in row["checks"][0]["check"]["details"]
    assert prepared["pack_hash"] == content_hash(prepared["pack"])


@pytest.mark.parametrize("case", ["seed", "parent_pack", "derived_signal", "truth", "evidence"])
def test_saved_trials_refuse_mixed_or_relabelled_evidence(calibration, case):
    cfg, wire, items, prepared = calibration
    row = measure_target(items, cfg, wire, load_policy(), prepared, 0, "clean", 1)
    changed = copy.deepcopy(row)
    if case == "seed":
        changed["seed"] += 1
    elif case == "parent_pack":
        changed["pack_hash"] = "other-pack"
    elif case == "derived_signal":
        changed["signals"]["text_equivalence_candidate"] = not row["signals"][
            "text_equivalence_candidate"
        ]
    elif case == "truth":
        changed["case"] = "swap"
    else:
        changed["checks"][0]["evidence_count"] += 1
    with pytest.raises(ValueError):
        checked_trial(changed, cfg, wire, load_policy(), prepared, 0, "clean", 1)


def test_qualification_prerequisites_refuse_modified_result_or_runtime():
    cfg = wire_configuration()
    cfg["runtime"]["numpy"] = "different-version"
    with pytest.raises(ValueError):
        prerequisites(configuration(), cfg, load_policy())
    cfg = wire_configuration()
    cfg["evidence"]["null_result"]["sha256"] = "wrong"
    with pytest.raises(ValueError):
        prerequisites(configuration(), cfg, load_policy())
    cfg = wire_configuration()
    cfg["evidence"]["null_result"]["path"] = "../private.json"
    with pytest.raises(ValueError):
        prerequisites(configuration(), cfg, load_policy())


def test_wire_study_reuses_sealed_trials_and_refuses_corrupt_summary(monkeypatch, tmp_path):
    import json

    import servegauge.validation.fingerprint_wire as module

    base, wire = small_configuration(), copy.deepcopy(wire_configuration())
    wire.update(reference_packs=1, workers=1, replicates={c: 1 for c in module.CASES})
    monkeypatch.setattr(module, "configuration", lambda: base)
    monkeypatch.setattr(module, "wire_configuration", lambda: wire)
    monkeypatch.setattr(module, "prerequisites", lambda *args: {"test_fixture_only": True})
    first = module.wire_study(tmp_path)
    assert len(first["trials"]) == 4 and first["accepted"] is False
    assert first["captured_responses"] == 864
    assert not list(tmp_path.rglob("*.part"))

    def unexpected(*args):
        raise AssertionError("cached evidence must be reused")

    monkeypatch.setattr(module, "measure_target", unexpected)
    assert module.wire_study(tmp_path) == first
    path = tmp_path / "wire.json"
    path.unlink()
    assert module.wire_study(tmp_path) == first
    envelope = json.loads(path.read_text())
    envelope["value"]["accepted"] = True
    envelope["value_hash"] = content_hash(envelope["value"])
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="summary"):
        module.wire_study(tmp_path)


@pytest.mark.parametrize("accepted,exit_code", [(True, 0), (False, 1)])
def test_wire_cli_preserves_observed_scope_and_gate(monkeypatch, tmp_path, accepted, exit_code):
    import json

    from typer.testing import CliRunner

    import servegauge.validation.fingerprint_wire as module
    from servegauge.cli import app

    monkeypatch.setattr(
        module,
        "wire_study",
        lambda out: {"accepted": accepted, "scope": "captured scripted targets only"},
    )
    result = CliRunner().invoke(app, ["validate", "fingerprint-wire-power", "--out", str(tmp_path)])
    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["scope"] == "captured scripted targets only"


def test_wire_cli_masks_prerequisite_errors(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import servegauge.validation.fingerprint_wire as module
    from servegauge.cli import app

    def refuse(out):
        raise ValueError("private-input")

    monkeypatch.setattr(module, "wire_study", refuse)
    result = CliRunner().invoke(app, ["validate", "fingerprint-wire-power", "--out", str(tmp_path)])
    assert result.exit_code == 3 and "private-input" not in result.output
