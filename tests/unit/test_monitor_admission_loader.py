"""Admission requires bound local evidence; labels and booleans are insufficient."""

import hashlib
import json
from pathlib import Path

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.admission import load_admission
from servegauge.observatory.identity import producer_identity
from servegauge.observatory.rolling import load_rolling_policy


def qualification(root, *, alter=None):
    producer = producer_identity()
    policy = load_rolling_policy()
    plan = json.loads(Path("configs/monitor_operating_study_v1.json").read_text())
    study = {
        "family_cases": [
            {**c, "families": 2000, "any_event_families": 0} for c in plan["family_cases"]
        ],
        "readiness_cases": [
            {**c, "replicates": 2000, "detected": 2000} for c in plan["readiness_cases"]
        ],
        "held_out_public_replay": {"trajectories": 120, "full_family_paths": 2, "passed": True},
    }
    receipt = {
        "version": "monitor-operating-qualification-v1",
        "id": "scripted-transport-operating-v1",
        "scope": "scripted-independent-transport-v1",
        "reference_tier": "MOCK",
        "source_hash": producer["source_hash"],
        "policy_hash": content_hash(policy.model_dump(mode="json")),
        "generator": producer["generator"],
        "runtime": producer["runtime"],
        "producer_sources": [producer["source_hash"], "b" * 64],
        "study_plan_sha256": hashlib.sha256(
            Path("configs/monitor_operating_study_v1.json").read_bytes()
        ).hexdigest(),
        "statistical_implementation_pins": {},
        "all_source_gates_passed": True,
        "source_tests_passed": 1,
        "native": {
            "canonical_units": 800,
            "full_canonical_units": 800,
            "first_attempt_wire_equal": True,
            "preflight_all_error_abstains": True,
            "compatible_producer_bridge": True,
            "raw_only_equal": True,
            "idempotent": True,
        },
        "study": study,
        "artifacts": {
            "docs/report/monitor-operating-v1/loader-contract-fixture.json": hashlib.sha256(
                b"{}\n"
            ).hexdigest()
        },
    }
    # This is a synthetic loader-contract fixture, never an executed statistical claim.
    from servegauge.observatory.admission import STATISTICAL_PINS

    receipt["statistical_implementation_pins"] = {
        n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in STATISTICAL_PINS
    }
    if alter:
        alter(receipt)
    (root / "configs").mkdir()
    (root / "docs/validation").mkdir(parents=True)
    (root / "docs/report/monitor-operating-v1").mkdir(parents=True)
    (root / "docs/report/monitor-operating-v1/loader-contract-fixture.json").write_bytes(b"{}\n")
    relative = "docs/validation/product-monitor-operating-qualification-v1.json"
    data = json.dumps(receipt, sort_keys=True).encode()
    (root / relative).write_bytes(data)
    (root / "configs/monitor_operating_study_v1.json").write_bytes(
        Path("configs/monitor_operating_study_v1.json").read_bytes()
    )
    registry = {
        "version": "monitor-admission-registry-v1",
        "qualifications": [
            {"id": receipt["id"], "receipt": relative, "sha256": hashlib.sha256(data).hexdigest()}
        ],
    }
    (root / "configs/monitor_transport_admission_v1.json").write_text(json.dumps(registry))
    return producer, receipt


def test_absent_receipt_never_admits_or_writes(tmp_path):
    status = load_admission(tmp_path)
    assert status.admission is None and not list(tmp_path.iterdir())


def test_bound_contract_supports_only_reviewed_producers_and_retains_actual_identity(tmp_path):
    producer, receipt = qualification(tmp_path)
    status = load_admission(tmp_path)
    assert status.admission is not None
    admission = status.admission
    binding = {**producer}
    assert admission.matches_binding(binding)
    binding["source_hash"] = binding["loaded_source_hash"] = "b" * 64
    assert admission.matches_binding(binding)
    binding["source_hash"] = binding["loaded_source_hash"] = "c" * 64
    assert not admission.matches_binding(binding)
    assert admission.reference_tier == "MOCK"
    assert admission.policy_hash == receipt["policy_hash"]


@pytest.mark.parametrize(
    "case",
    [
        "source",
        "policy",
        "scope",
        "tier",
        "runtime",
        "generator",
        "null_power",
        "low_power",
        "too_few",
        "missing_replay",
        "native",
        "native_full",
        "checks",
        "pin",
        "path",
    ],
)
def test_changed_or_inadequate_claims_do_not_admit(tmp_path, case):
    def alter(r):
        if case == "source":
            r["source_hash"] = "c" * 64
        elif case == "policy":
            r["policy_hash"] = "c" * 64
        elif case == "scope":
            r["scope"] = "provider"
        elif case == "tier":
            r["reference_tier"] = "R3"
        elif case == "runtime":
            r["runtime"]["complete"] = False
        elif case == "generator":
            r["generator"]["pins"] = {}
        elif case == "null_power":
            r["study"]["family_cases"][0]["any_event_families"] = 1000
        elif case == "low_power":
            r["study"]["readiness_cases"][0]["detected"] = 1000
        elif case == "too_few":
            r["study"]["family_cases"][0]["families"] = 20
        elif case == "missing_replay":
            r["study"]["held_out_public_replay"]["passed"] = False
        elif case == "native":
            r["native"]["preflight_all_error_abstains"] = False
        elif case == "native_full":
            r["native"]["full_canonical_units"] = 0
        elif case == "checks":
            r["all_source_gates_passed"] = False
        elif case == "pin":
            r["statistical_implementation_pins"] = {}
        else:
            r["artifacts"] = {"../foreign.json": "a" * 64}

    qualification(tmp_path, alter=alter)
    assert load_admission(tmp_path).admission is None


def test_hash_tamper_and_changed_loaded_source_or_runtime_refuse(tmp_path):
    producer, _ = qualification(tmp_path)
    admission = load_admission(tmp_path).admission
    assert admission
    altered = {**producer, "loaded_source_hash": "b" * 64}
    assert not admission.matches_binding(altered)
    altered = json.loads(json.dumps(producer))
    altered["runtime"]["python"]["version"] = "unknown"
    assert not admission.matches_binding(altered)
    (tmp_path / "docs/report/monitor-operating-v1/loader-contract-fixture.json").write_text(
        "changed"
    )
    assert load_admission(tmp_path).admission is None


@pytest.mark.parametrize("key", ["study", "native", "producer_sources", "generator", "runtime"])
def test_malformed_nested_receipt_refuses_without_exception(tmp_path, key):
    qualification(tmp_path, alter=lambda receipt: receipt.update({key: None}))
    assert load_admission(tmp_path).admission is None
