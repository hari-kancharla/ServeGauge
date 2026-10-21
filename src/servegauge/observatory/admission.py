"""Strict local mock qualification receipts; no provider or label-based admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from servegauge.core.hashing import content_hash
from servegauge.observatory.identity import producer_identity
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.stats.intervals import wilson

STATISTICAL_PINS = (
    "src/servegauge/observatory/bounds.py",
    "src/servegauge/observatory/rolling.py",
    "src/servegauge/observatory/temporal.py",
    "src/servegauge/validation/transport_operating.py",
    "configs/monitor_transport_v2.yaml",
)
REGISTRY = "configs/monitor_transport_admission_v1.json"
QUALIFICATION = "scripted-transport-operating-v1"


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _identity(value: Any, version: str, key: str) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("version") == version
        and value.get("complete") is True
        and _digest(value.get(key))
        and value[key] == content_hash({k: v for k, v in value.items() if k != key})
    )


def _file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(p.lower() == "starter files" for p in [*root.parts, *relative.parts])
    ):
        raise ValueError("qualification_path_refused")
    path = root.joinpath(*relative.parts)
    if (
        any(p.is_symlink() for p in [*reversed(path.absolute().parents), path])
        or not path.is_file()
    ):
        raise ValueError("qualification_file_unavailable")
    return path


def _read(root: Path, name: str) -> dict[str, Any]:
    value = json.loads(_file(root, name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("qualification_object_required")
    return value


def statistical_pins() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    result = {}
    for name in STATISTICAL_PINS:
        path = (
            package.joinpath(*PurePosixPath(name).parts[2:])
            if name.startswith("src/servegauge/")
            else Path(name)
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError("qualification_implementation_unavailable")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@dataclass(frozen=True, init=False)
class VerifiedAdmission:
    id: str
    receipt_path: str
    receipt_hash: str
    current_source_hash: str
    policy_hash: str
    generator_hash: str
    runtime_hash: str
    measurement_source_hash: str
    producer_sources: frozenset[str]
    min_mock_context_length: int
    max_mock_context_length: int
    reference_tier: Literal["MOCK"]

    def admits_source(self, value: str | None) -> bool:
        return bool(_digest(value) and value in self.producer_sources)

    def matches_binding(self, value: dict[str, Any]) -> bool:
        if value.get("loaded_source_hash") != value.get("source_hash") or not self.admits_source(
            value.get("source_hash")
        ):
            return False
        runtime, generator = value.get("runtime"), value.get("generator")
        if not isinstance(runtime, dict) or not isinstance(generator, dict):
            return False
        return bool(
            _identity(runtime, "monitor-runtime-v1", "runtime_hash")
            and _identity(generator, "monitor-generator-v1", "generator_hash")
            and runtime["runtime_hash"] == self.runtime_hash
            and generator["generator_hash"] == self.generator_hash
        )


@dataclass(frozen=True)
class AdmissionState:
    admission: VerifiedAdmission | None
    reason: str


def _count(row: dict[str, Any], name: str, minimum: int, maximum: int | None = None) -> int:
    value = row.get(name)
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError("qualification_count_invalid")
    return value


def _study(receipt: dict[str, Any], plan: dict[str, Any], alpha: float) -> None:
    study = receipt["study"]
    if not isinstance(study, dict) or plan.get("version") != "monitor-operating-study-v1":
        raise ValueError("qualification_study_required")
    for key, count_key, event_key in [
        ("family_cases", "families", "any_event_families"),
        ("readiness_cases", "replicates", "detected"),
    ]:
        rows = study[key]
        if (
            not isinstance(rows, list)
            or any(not isinstance(r, dict) for r in rows)
            or not isinstance(plan[key], list)
            or any(not isinstance(r, dict) for r in plan[key])
        ):
            raise ValueError("qualification_case_inventory_mismatch")
        if len(rows) != len(plan[key]) or {r["id"] for r in rows} != {r["id"] for r in plan[key]}:
            raise ValueError("qualification_case_inventory_mismatch")
        for expected in plan[key]:
            row = next(r for r in rows if r["id"] == expected["id"])
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError("qualification_case_scope_mismatch")
            n = _count(row, count_key, max(2000, plan["replicates"]))
            k = _count(row, event_key, 0, n)
            low, high = wilson(k, n)
            if (
                key == "family_cases"
                and high > alpha
                or key == "readiness_cases"
                and low < max(0.8, plan["min_power"])
            ):
                raise ValueError("qualification_statistical_gate_failed")
    replay = study["held_out_public_replay"]
    if (
        not isinstance(replay, dict)
        or replay.get("passed") is not True
        or _count(replay, "full_family_paths", 2) < 2
        or _count(replay, "trajectories", max(120, plan["held_out_replays"])) < 120
    ):
        raise ValueError("qualification_public_replay_required")


def load_admission(root: Path = Path(".")) -> AdmissionState:
    """Read an immutable accepted receipt. Unknown or incomplete claims abstain."""
    try:
        registry = _read(root, REGISTRY)
        entries = registry["qualifications"]
        if (
            registry.get("version") != "monitor-admission-registry-v1"
            or not isinstance(entries, list)
            or len(entries) != 1
        ):
            raise ValueError("qualification_registry_scope_mismatch")
        entry = entries[0]
        if not isinstance(entry, dict):
            raise ValueError("qualification_registry_scope_mismatch")
        relative = entry["receipt"]
        if (
            entry.get("id") != QUALIFICATION
            or not _digest(entry.get("sha256"))
            or not isinstance(relative, str)
            or not relative.startswith("docs/validation/product-monitor-operating-qualification-v")
            or not relative.endswith(".json")
        ):
            raise ValueError("qualification_registry_scope_mismatch")
        if hashlib.sha256(_file(root, relative).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("qualification_receipt_hash_mismatch")
        receipt = _read(root, relative)
        producer = producer_identity()
        policy = load_rolling_policy()
        plan_path = _file(root, "configs/monitor_operating_study_v1.json")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        minimum_context = _count(plan, "min_mock_context_length", 512, 131072)
        maximum_context = _count(plan, "max_mock_context_length", minimum_context, 131072)
        if (
            receipt.get("version") != "monitor-operating-qualification-v1"
            or receipt.get("id") != QUALIFICATION
            or receipt.get("scope") != "scripted-independent-transport-v1"
            or receipt.get("reference_tier") != "MOCK"
            or receipt.get("source_hash") != producer["source_hash"]
            or producer["loaded_source_hash"] != producer["source_hash"]
            or receipt.get("policy_hash") != content_hash(policy.model_dump(mode="json"))
            or not _identity(receipt.get("generator"), "monitor-generator-v1", "generator_hash")
            or not _identity(receipt.get("runtime"), "monitor-runtime-v1", "runtime_hash")
            or receipt["generator"] != producer["generator"]
            or receipt["runtime"] != producer["runtime"]
            or receipt.get("statistical_implementation_pins") != statistical_pins()
            or receipt.get("study_plan_sha256")
            != hashlib.sha256(plan_path.read_bytes()).hexdigest()
            or receipt.get("all_source_gates_passed") is not True
            or _count(receipt, "source_tests_passed", 1) < 1
        ):
            raise ValueError("qualification_source_runtime_or_scope_mismatch")
        _study(receipt, plan, policy.family_alpha)
        native = receipt["native"]
        if not isinstance(native, dict):
            raise ValueError("qualification_native_acceptance_required")
        if (
            _count(native, "canonical_units", 800) < 800
            or _count(native, "full_canonical_units", 800) < 800
        ) or any(
            native.get(k) is not True
            for k in (
                "first_attempt_wire_equal",
                "preflight_all_error_abstains",
                "compatible_producer_bridge",
                "raw_only_equal",
                "idempotent",
            )
        ):
            raise ValueError("qualification_native_acceptance_required")
        sources = receipt["producer_sources"]
        if (
            not isinstance(sources, list)
            or not sources
            or any(not _digest(s) for s in sources)
            or producer["source_hash"] not in sources
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("qualification_producer_inventory_invalid")
        artifacts = receipt["artifacts"]
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("qualification_artifacts_required")
        for name, expected in artifacts.items():
            if (
                not name.startswith("docs/report/monitor-operating-v1/")
                or not _digest(expected)
                or hashlib.sha256(_file(root, name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("qualification_artifact_integrity_mismatch")
        fields = {
            "id": QUALIFICATION,
            "receipt_path": relative,
            "receipt_hash": entry["sha256"],
            "current_source_hash": producer["source_hash"],
            "policy_hash": receipt["policy_hash"],
            "generator_hash": producer["generator"]["generator_hash"],
            "runtime_hash": producer["runtime"]["runtime_hash"],
            "measurement_source_hash": content_hash(
                [
                    "scripted-independent-transport-v1",
                    producer["generator"]["generator_hash"],
                    producer["runtime"]["runtime_hash"],
                ]
            ),
            "producer_sources": frozenset(sources),
            "min_mock_context_length": minimum_context,
            "max_mock_context_length": maximum_context,
            "reference_tier": "MOCK",
        }
        verified = object.__new__(VerifiedAdmission)
        for name, value in fields.items():
            object.__setattr__(verified, name, value)
        return AdmissionState(verified, "qualified_for_declared_local_mock_model")
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return AdmissionState(None, "qualification_absent_inadequate_or_mismatched")
