"""Complete private comparison preparation; no procedure or owner action executes."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import EndpointSpec
from servegauge.mock.sealed_replay import _loopback_url
from servegauge.runner.plan import RunPlan
from servegauge.storage.comparison import (
    ComparisonSelection,
    _envelope,
    _files,
    _guard,
    _load,
    _private_json,
    _relative,
    _require_sanitized,
)
from servegauge.storage.comparison_controls import ControlSelection, compile_controls
from servegauge.storage.comparison_program import ComparisonProgram, compile_program


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _write(path: Path, blob: bytes) -> None:
    _guard(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)


def _mock_endpoint(endpoint: EndpointSpec) -> bool:
    try:
        _loopback_url(endpoint.base_url.rstrip("/") + "/chat/completions")
    except ValueError:
        return False
    return endpoint.kind == "mock" and endpoint.model == "mock" and endpoint.api_key_env is None


def _source_scope(snapshot: Path, selection: ComparisonSelection) -> tuple[bool, dict[str, Any]]:
    synthetic = True
    scopes = {}
    required_suites = {"S0", "S1", "S2", "S3", "S4", "S7"}
    for pin in selection.sources:
        source = _load(snapshot / "sources" / safe_filename(pin.run_id), pin)
        plan, metadata = source.analysis.plan, source.analysis.metadata
        targets = metadata.get("target_ids", [])
        selected = [endpoint for endpoint in plan.endpoints if endpoint.id in targets]
        reasons = []
        if (
            not isinstance(targets, list)
            or len(set(targets)) != len(targets)
            or len(targets) < 8
            or len(selected) != len(targets)
            or {endpoint.id for endpoint in selected} != set(targets)
        ):
            reasons.append("at_least_eight_explicit_source_targets_required")
        providers = {endpoint.labels.get("provider_name", "").casefold() for endpoint in selected}
        quantization = {
            endpoint.labels.get("quantization", "unknown").casefold() for endpoint in selected
        }
        if (
            len(providers - {""}) < 2
            or len(quantization - {""}) < 2
            or any(
                endpoint.kind != "openrouter"
                or endpoint.model != "openai/gpt-oss-120b"
                or endpoint.openrouter_pin is None
                or endpoint.openrouter_pin.allow_fallbacks
                or not endpoint.openrouter_pin.require_parameters
                for endpoint in selected
            )
        ):
            reasons.append("pinned_distinct_model_a_providers_and_declared_labels_required")
        if plan.level not in {"core", "full"} or not required_suites <= set(plan.suites):
            reasons.append("complete_core_model_a_plan_required")
        pack = source.analysis.reference_pack
        if pack is None or pack.tier != "R3":
            reasons.append("sealed_r3_reference_required")
        for endpoint in selected:
            rows = [row for row in source.records.values() if row.endpoint_id == endpoint.id]
            main = [row for row in rows if row.run_id == pin.run_id]
            if not required_suites <= {row.suite_id for row in rows} or not any(
                row.status == 200 and not row.error for row in main
            ):
                reasons.append("complete_recorded_target_suite_scope_required")
                break
        source_plans = []
        for name, digest in pin.files_sha256.items():
            if name.endswith("/capture.json"):
                path = _relative(snapshot, f"sources/{safe_filename(pin.run_id)}/data/raw/{name}")
                blob = path.read_bytes()
                if _sha(blob) != digest:
                    raise ValueError("comparison_source_context_changed")
                context = _private_json(blob)["value"]
                source_plans.append(RunPlan.model_validate(context["manifest"]["config"]))
        mock = (
            bool(source_plans)
            and all(
                part.max_usd == 0
                and part.endpoints
                and all(_mock_endpoint(endpoint) for endpoint in part.endpoints)
                for part in source_plans
            )
            and all(row.cost_usd in {None, 0} for row in source.records.values())
        )
        synthetic = synthetic and mock
        scopes[pin.run_id] = {
            "eligible_source_scope": not reasons,
            "reasons": sorted(set(reasons)),
            "target_count": len(selected),
            "source_kind": "synthetic" if mock else "recorded",
            "declared_quantization_is_not_verified_weights": True,
        }
    return synthetic, scopes


def _payloads(
    snapshot: Path,
    *,
    selection_hash: str,
    controls: ControlSelection,
    controls_hash: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    failures = compile_program(snapshot, selection_hash=selection_hash)
    passing = compile_controls(
        snapshot, controls, selection_hash=selection_hash, controls_hash=controls_hash
    )
    cases = failures.cases + passing.cases
    if len({case.case_id for case in cases}) != 4 or any(
        case.original_check.check_id not in {"P06.structure", "T06"} for case in cases
    ):
        raise ValueError("comparison_supported_distinct_four_case_kit_required")
    selection = ComparisonSelection.model_validate(
        _private_json(_relative(snapshot, "selection.json").read_bytes())
    )
    synthetic, scopes = _source_scope(snapshot, selection)
    program = ComparisonProgram(
        selection_hash=content_hash(
            {
                "failures": selection_hash,
                "controls": controls_hash,
            }
        ),
        cases=cases,
    )
    contexts = {}
    for case in cases:
        run = case.original_check.run_id
        sealed = _envelope(
            _relative(
                snapshot,
                f"sources/{safe_filename(run)}/data/raw/{safe_filename(run)}/analysis.json",
            )
        )
        metadata = sealed["metadata"]
        contexts[case.case_id] = {
            key: metadata[key] for key in ["profile", "policy", "suite_config"]
        }
    values = {
        "procedures/program.json": program.model_dump(mode="json"),
        "procedures/controls-selection.json": controls.model_dump(mode="json"),
        "procedures/contexts.json": contexts,
    }
    for value in values.values():
        _require_sanitized(value)
    payloads = {name: canonical_json(value) + b"\n" for name, value in values.items()}
    manifest = {
        "version": "comparison-kit-v1",
        "state": "prepared_only",
        "owner_comparison_executed": False,
        "setup_seconds": None,
        "selection_hash": selection_hash,
        "controls_hash": controls_hash,
        "program_hash": program.program_hash(),
        "synthetic_only": synthetic,
        "m15_eligible": bool(scopes)
        and all(value["eligible_source_scope"] for value in scopes.values()),
        "source_eligibility": scopes,
        "m15_report_site_completion": "requires_explicit_owner_attestation",
        "cases": [case.model_dump(mode="json", exclude={"phases"}) for case in cases],
        "procedure_limit": "Independent P06.structure/T06 assertions; other checks refuse.",
        "root_cause_limit": (
            "Observed raw signatures do not identify an internal serving parser or intent."
        ),
        "replay_adds_independent_n": False,
    }
    return payloads, manifest


def prepare_kit(
    root: Path,
    snapshot: Path,
    *,
    selection_hash: str,
    controls: ControlSelection,
    controls_hash: str,
    out: Path,
) -> dict[str, Any]:
    """Create an exclusive private kit, retaining the full original sealed snapshot."""
    # Reject every protected supplied path before any filesystem operation.
    for path in [root, snapshot, out]:
        if ".." in path.parts or any(part.casefold() == "starter files" for part in path.parts):
            raise ValueError("comparison_path_refused")
    for path in [root, snapshot, out]:
        _guard(path)
    private = root.absolute() / ".cache/comparisons"
    if (
        not out.absolute().is_relative_to(private)
        or out.absolute() == private
        or out.exists()
        or out.absolute().is_relative_to(snapshot.absolute())
        or snapshot.absolute().is_relative_to(out.absolute())
    ):
        raise ValueError("comparison_new_private_kit_destination_required")
    # Verify original anchors before copying any evidence.
    _payloads(
        snapshot, selection_hash=selection_hash, controls=controls, controls_hash=controls_hash
    )
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".kit-", dir=out.parent))
    try:
        for path in _files(snapshot):
            name = path.relative_to(snapshot).as_posix()
            _write(_relative(temporary, "snapshot/" + name), path.read_bytes())
        payloads, manifest = _payloads(
            temporary / "snapshot",
            selection_hash=selection_hash,
            controls=controls,
            controls_hash=controls_hash,
        )
        for name, blob in payloads.items():
            _write(_relative(temporary, name), blob)
        manifest["files_sha256"] = {
            path.relative_to(temporary).as_posix(): _sha(path.read_bytes())
            for path in _files(temporary)
        }
        _write(temporary / "manifest.json", canonical_json(manifest) + b"\n")
        verify_kit(temporary, kit_hash=content_hash(manifest))
        if out.exists():
            raise ValueError("comparison_kit_destination_created_during_preparation")
        temporary.rename(out)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_kit(kit: Path, *, kit_hash: str) -> dict[str, Any]:
    """Require an external manifest anchor and reconstruct every declared payload."""
    _guard(kit)
    manifest = _private_json(_relative(kit, "manifest.json").read_bytes())
    if not isinstance(manifest, dict) or content_hash(manifest) != kit_hash:
        raise ValueError("comparison_external_kit_anchor_changed")
    actual = {
        path.relative_to(kit).as_posix(): _sha(path.read_bytes())
        for path in _files(kit)
        if path != kit / "manifest.json"
    }
    if actual != manifest.get("files_sha256"):
        raise ValueError("comparison_kit_files_changed")
    controls = ControlSelection.model_validate(
        _private_json(_relative(kit, "procedures/controls-selection.json").read_bytes())
    )
    payloads, expected = _payloads(
        kit / "snapshot",
        selection_hash=manifest["selection_hash"],
        controls=controls,
        controls_hash=manifest["controls_hash"],
    )
    expected["files_sha256"] = actual
    if canonical_json(expected) != canonical_json(manifest) or any(
        _relative(kit, name).read_bytes() != blob for name, blob in payloads.items()
    ):
        raise ValueError("comparison_kit_reconstruction_changed")
    return manifest
