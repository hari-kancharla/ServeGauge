"""Explicit reusable reference snapshots, source rights and current upload preflight."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import ReferencePack
from servegauge.core.secrets import configured_secrets
from servegauge.observatory.notice_workflow import publication_status
from servegauge.observatory.notices import failure_subject
from servegauge.observatory.provenance import implementation_hash
from servegauge.refs.pack import load_pack
from servegauge.refs.provenance import OutcomeProvenanceError, outcome_items
from servegauge.refs.publication import VERSION, clean_public, outcomes_match, public_pack
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.notices import guard_path, notice_snapshots

LOCAL = "local-selection.json"
DATASHEET = """# ServeGauge reusable reference datasheet

This is a selected reference pack, not a provider ranking, weight/quantization or
hardware certificate. It retains the original per-item outcomes, configuration,
calibration and fingerprint reference observations without manufacturing missing
measurements. Counts, calibration bands, intervals and reference tiers remain in
the pack. Repeated and linked captures are not new independent observations.

The public copy has a distinct identity and original pack/content-hash provenance.
Private source/transport metadata and endpoint URLs are excluded; minimal declared
endpoint/model and mock-seed context is retained. Identifiers and generated model
text can be linkable. These exclusions are not an anonymity or security guarantee.
Known private fields/paths/credentials in semantic data refuse rather than being
silently rewritten. No engine, GPU, model-weight or unique-cause claim is added.

Published text requires current intact captures and declared generated CC-BY-4.0
item provenance. Unknown/external text rights block publication; no model assets,
external benchmark text, headers or account/generation records are intentionally
included. Reference samples may contain generated model text required for later
comparisons. This differs from the numeric dataset's content-removed summaries.

The standalone pack can be loaded with the ordinary guarded explicit-file loader.
Comparisons still require matching profile/items/configuration/n, reference peers,
method/runtime and declared assumptions. Optional FP4 needs separately pinned local
embedding assets/runtime. Scoped scripted qualification cannot certify providers.

All MOCK packs/captures remain local and upload-blocked. Source failures retain
the exact-revision notice policy; assertions are not independently verified delivery.
Upload preflight regenerates exact bytes and rechecks current source/rights/notices.
A local card or matching hashes alone never establish public eligibility.

The separate local-selection.json is private and never uploaded. Only the four
reviewed public files enter one explicit dataset commit; no inference, contact,
repository creation/deletion, GPU or schedule is enabled by staging. Configure a
dedicated dataset: its root card selects this snapshot. Older snapshot paths remain.
"""


def _guard(path: Path) -> None:
    if ".." in path.parts or any(p.casefold() == ".git" for p in path.parts):
        raise ValueError("reference_canonical_path_required")
    guard_path(path)


def _source(root: Path, identifier: str) -> tuple[ReferencePack, str]:
    if not re.fullmatch(r"ref-[a-f0-9]{20}", identifier):
        raise ValueError("reference_canonical_pack_id_required")
    directory = root / "refs"
    _guard(directory)
    matches = []
    for profile in directory.iterdir():
        # Inspect the owned namespace's names only; never stat or enter this entry.
        if profile.name.casefold() == "starter files":
            continue
        _guard(profile)
        path = profile / safe_filename(identifier) / "pack.json"
        _guard(path)
        if path.is_file():
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("reference_unique_saved_pack_required")
    pack = load_pack(str(matches[0]))
    if pack.pack_id != identifier or matches[0].parents[1].name != safe_filename(pack.profile):
        raise ValueError("reference_pack_location_identity_changed")
    return pack, hashlib.sha256(matches[0].read_bytes()).hexdigest()


def _mock(pack: ReferencePack) -> bool:
    declarations = list(pack.manifest.get("endpoint_declarations", []))
    declarations += pack.manifest.get("source_manifest", {}).get("endpoint_metadata", [])
    for control in pack.fingerprints.get("controls", []):
        declarations.extend(control.get("declarations", []))
    declarations.extend(
        row["declaration"] for row in pack.fingerprints.get("reference_sources", {}).values()
    )
    return pack.tier == "MOCK" or any(row.get("kind") == "mock" for row in declarations)


def _guard_capture(raw: Path, run: str) -> None:
    directory = raw / safe_filename(run)
    _guard(directory)
    descriptor = directory / "analysis.json"
    _guard(descriptor)
    if not descriptor.is_file():
        raise ValueError("reference_source_descriptor_unavailable")
    # Core reconstruction verifies this envelope and raw contents afterward. Read
    # only its declared namespace identities here, before any supporting traversal.
    sealed = json.loads(descriptor.read_bytes())["value"]
    supporting = set(sealed["run_ids"]) | {run}
    capability = sealed["metadata"].get("capability_run_id")
    if capability is not None:
        supporting.add(capability)
    for identifier in supporting:
        namespace = raw / safe_filename(identifier)
        _guard(namespace)
        for path in namespace.iterdir():
            _guard(path)
            if (
                path.name in {"analysis.json", "capture.json", "reference.json"}
                or path.name.endswith(".jsonl.zst")
            ) and not path.is_file():
                raise ValueError("reference_source_regular_file_required")


def source_state(root: Path, pack: ReferencePack) -> dict[str, Any]:
    """Bind intact local source captures and licenses, without storing their private data."""
    runs = {pack.manifest["source_run"]}
    runs.update(row["run_id"] for row in pack.fingerprints.get("controls", []))
    runs.update(row["run_id"] for row in pack.fingerprints.get("reference_sources", {}).values())
    wanted = {
        item
        for checks in pack.outcomes.values()
        for view in checks.values()
        for item in view["item_ids"]
    }
    evidence = {
        identifier
        for view in pack.fingerprints.get("views", {}).values()
        for samples in view["references"].values()
        for sample in samples.values()
        for identifier in sample.get("evidence", [])
    }
    states, blockers = [], set()
    captured: dict[str, dict[str, Any]] = {}
    rights_verified = True
    if pack.tier in {"R1", "R2"}:
        from servegauge.storage.owner_reference import validate_pack_origin

        try:
            validate_pack_origin(root, pack)
        except (OSError, ValueError, KeyError, TypeError, LookupError, ArithmeticError):
            rights_verified = False
            blockers.add("source_owner_origin_unverified")
    for run in sorted(runs):
        try:
            _guard_capture(root / "data/raw", run)
            analysis = reconstruct(root / "data/raw", run)
            source = export_outcomes(analysis)
            captured[run] = source
            if run == pack.manifest["source_run"] and (
                source["manifest"] != pack.manifest["source_manifest"]
                or source["configuration"] != pack.manifest["configuration"]
                or source["item_set_hashes"] != pack.manifest["item_set_hashes"]
            ):
                blockers.add("source_pack_context_mismatch")
                rights_verified = False
            if run == pack.manifest["source_run"] and not outcomes_match(pack, source):
                blockers.add("source_numeric_outcomes_mismatch")
                rights_verified = False
            selected = {r.item_id for r in analysis.records if r.exchange_id in evidence}
            provenance: dict[str, Any] = {}
            if run == pack.manifest["source_run"]:
                try:
                    links = outcome_items(pack, analysis.items, analysis.records, analysis.samples)
                    selected |= links.item_ids
                    if links.aggregate_links:
                        provenance["outcome_provenance"] = [
                            asdict(row) for row in links.aggregate_links
                        ]
                except OutcomeProvenanceError:
                    selected |= wanted
                    rights_verified = False
                    blockers.add("source_outcome_provenance_unverified")
                    provenance["outcome_provenance_unverified"] = True
            by_id = {item.item_id: item for item in analysis.items}
            known = bool(selected) and selected <= by_id.keys()
            if selected and (
                not known
                or any(
                    by_id[item].source != "generated" or by_id[item].license != "CC-BY-4.0"
                    for item in selected & by_id.keys()
                )
            ):
                rights_verified = False
            if any(e.kind == "mock" for e in analysis.plan.endpoints):
                blockers.add("scripted_mock_not_public")
            for check in analysis.checks:
                if (
                    check.verdict == "FAIL"
                    and not publication_status(
                        root, failure_subject(analysis, check.endpoint_id, check.check_id)
                    )["eligible"]
                ):
                    blockers.add("right_of_reply_required")
            states.append(
                content_hash(
                    {
                        "run": run,
                        "manifest": analysis.manifest,
                        "items": [i.model_dump(mode="json") for i in analysis.items],
                        "records": [r.model_dump(mode="json") for r in analysis.records],
                        "selected_items": sorted(selected),
                        **provenance,
                    }
                )
            )
        except (OSError, ValueError, KeyError, TypeError, LookupError):
            rights_verified = False
            blockers.add("source_capture_unavailable")
    for view in pack.fingerprints.get("views", {}).values():
        for key, samples in view["references"].items():
            descriptor = pack.fingerprints.get("reference_sources", {}).get(key, {})
            run = descriptor.get("run_id", pack.manifest["source_run"])
            endpoint = descriptor.get("endpoint_id", key)
            observed = captured.get(run, {}).get("fingerprints", {}).get(endpoint, {})
            if not samples or any(
                observed.get(method) != sample for method, sample in samples.items()
            ):
                blockers.add("source_fingerprint_observations_mismatch")
                rights_verified = False
    if not (wanted or evidence) or not rights_verified:
        rights_verified = False
        blockers.add("source_rights_unverified")
    notices = [
        row
        for row in notice_snapshots(root / "data/raw")
        if row["spec"]["subject"]["run_id"] in runs
    ]
    return {
        "binding": content_hash({"captures": states, "notices": notices}),
        "rights_verified": rights_verified,
        "blockers": sorted(blockers),
    }


def _payloads(pack: ReferencePack, rights: bool) -> dict[str, bytes]:
    blob = canonical_json(pack.model_dump(mode="json")) + b"\n"
    datasheet = (
        DATASHEET
        + "\nCurrent capture rights: "
        + ("declared generated CC-BY-4.0.\n" if rights else "unverified; no license granted.\n")
    ).encode()
    snapshot = content_hash(
        {
            "version": VERSION,
            "pack": hashlib.sha256(blob).hexdigest(),
            "datasheet": hashlib.sha256(datasheet).hexdigest(),
        }
    )
    prefix = f"references/{snapshot}"
    manifest = (
        canonical_json(
            {
                "version": VERSION,
                "snapshot": snapshot,
                "pack_id": pack.pack_id,
                "profile": pack.profile,
                "tier": pack.tier,
                "files": {
                    "pack.json": hashlib.sha256(blob).hexdigest(),
                    "DATASHEET.md": hashlib.sha256(datasheet).hexdigest(),
                },
            }
        )
        + b"\n"
    )
    card = (
        ("---\nlicense: cc-by-4.0\n---\n\n" if rights else "")
        + "# ServeGauge reusable reference\n\n"
        + f"Selected pack: `{pack.pack_id}`, declared tier `{pack.tier}`. "
        + "This is a local candidate until current owner preflight passes.\n\n"
        + f"[Pack]({prefix}/pack.json) | [Datasheet]({prefix}/DATASHEET.md) | "
        + f"[Manifest]({prefix}/manifest.json)\n\n"
        + "Load the explicit pack.json file with the ordinary reference loader. "
        + "Older reference snapshot paths remain separate; do not pool reused captures.\n"
    ).encode()
    payloads = {
        "README.md": card,
        f"{prefix}/pack.json": blob,
        f"{prefix}/DATASHEET.md": datasheet,
        f"{prefix}/manifest.json": manifest,
    }
    secrets = tuple(configured_secrets())
    for raw in payloads.values():
        # JSON semantics are checked before serialization; scan each document too.
        clean_public(raw.decode(), secrets=secrets)
    return payloads


def _inventory(out: Path, expected: set[str]) -> dict[str, bytes]:
    prefixes = {
        parent.as_posix()
        for name in expected
        for parent in Path(name).parents
        if str(parent) != "."
    }
    result, pending = {}, [out]
    while pending:
        directory = pending.pop()
        _guard(directory)
        for path in directory.iterdir():
            relative = path.relative_to(out).as_posix()
            if relative not in expected and relative not in prefixes:
                raise ValueError("reference_foreign_output_entry")
            _guard(path)
            if relative in expected and path.is_file():
                result[relative] = path.read_bytes()
            elif relative in prefixes and path.is_dir():
                pending.append(path)
            else:
                raise ValueError("reference_output_entry_type_invalid")
    if result.keys() != expected:
        raise ValueError("reference_output_inventory_incomplete")
    return result


def verify_reference(out: Path) -> dict[str, Any]:
    _guard(out)
    marker = out / LOCAL
    _guard(marker)
    if not marker.is_file():
        raise ValueError("reference_export_selection_unavailable")
    value = json.loads(marker.read_bytes())
    if (
        not isinstance(value, dict)
        or set(value)
        != {"version", "source_pack_id", "snapshot", "rights_verified", "input_hash", "files"}
        or value.get("version") != VERSION
        or not isinstance(value["source_pack_id"], str)
        or not re.fullmatch(r"ref-[a-f0-9]{20}", value["source_pack_id"])
        or any(
            not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{64}", value[key])
            for key in ["snapshot", "input_hash"]
        )
        or not isinstance(value["files"], dict)
        or any(
            not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
            for digest in value["files"].values()
        )
    ):
        raise ValueError("reference_export_manifest_invalid")
    prefix = f"references/{value['snapshot']}"
    expected = {
        "README.md",
        *(f"{prefix}/{name}" for name in ["pack.json", "DATASHEET.md", "manifest.json"]),
    }
    if set(value["files"]) != expected:
        raise ValueError("reference_export_inventory_invalid")
    inventory = _inventory(out, expected | {LOCAL})
    if any(
        hashlib.sha256(inventory[name]).hexdigest() != digest
        for name, digest in value["files"].items()
    ):
        raise ValueError("reference_export_file_changed")
    pack = load_pack(str(out / prefix / "pack.json"))
    if (
        pack.manifest.get("publication_version") != VERSION
        or pack.manifest.get("source_pack_id") != value["source_pack_id"]
    ):
        raise ValueError("reference_public_copy_provenance_invalid")
    rights = value["rights_verified"]
    if type(rights) is not bool or _payloads(pack, rights) != {n: inventory[n] for n in expected}:
        raise ValueError("reference_export_projection_changed")
    return dict(value)


def _project(root: Path, identifier: str) -> tuple[dict[str, bytes], dict[str, Any], list[str]]:
    _guard(root)
    original, source_hash = _source(root, identifier)
    pack = public_pack(original)
    clean_public(pack.model_dump(mode="json"), secrets=tuple(configured_secrets()))
    state = source_state(root, original)
    rights = state["rights_verified"]
    if type(rights) is not bool:
        raise ValueError("reference_source_rights_invalid")
    files = _payloads(pack, rights)
    snapshot = next(name.split("/")[1] for name in files if name.endswith("/pack.json"))
    value = {
        "version": VERSION,
        "source_pack_id": identifier,
        "snapshot": snapshot,
        "rights_verified": rights,
        "input_hash": content_hash(
            {
                "pack": source_hash,
                "source": state["binding"],
                "implementation": implementation_hash(),
            }
        ),
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in files.items()},
    }
    blockers = set(state["blockers"])
    if not rights:
        blockers.add("source_rights_unverified")
    if _mock(original):
        blockers.add("scripted_mock_not_public")
    return files, value, sorted(blockers)


def stage_reference(root: Path, identifier: str, out: Path) -> dict[str, Any]:
    _guard(root)
    _guard(out)
    if root.absolute().is_relative_to(out.absolute()) or any(
        out.absolute().is_relative_to(root.absolute() / name)
        for name in [
            "refs",
            "data",
            "src",
            "tests",
            "configs",
            "profiles",
            "budget",
            "docs",
            "scripts",
            ".github",
            ".agents",
            ".codex",
            ".aws",
        ]
    ):
        raise ValueError("reference_output_overlaps_private_or_source_tree")
    previous = verify_reference(out) if out.exists() else None
    payloads, value, blockers = _project(root, identifier)
    if previous:
        if previous != value or any((out / n).read_bytes() != b for n, b in payloads.items()):
            raise ValueError("reference_export_inputs_changed_use_fresh_output")
    else:
        out.mkdir(parents=True, exist_ok=False)
        for name, blob in payloads.items():
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _guard(path)
            with path.open("xb") as stream:
                stream.write(blob)
        with (out / LOCAL).open("xb") as stream:
            stream.write(canonical_json(value) + b"\n")
        verify_reference(out)
    return {
        "dry_run": True,
        "source_pack_id": identifier,
        "files": sorted(payloads),
        "publication_blockers": blockers,
        "network_writes": 0,
        "published": False,
        "model_api_calls": 0,
    }


def reference_preflight(root: Path, identifier: str, out: Path) -> dict[str, bytes]:
    value = verify_reference(out)
    payloads, current, blockers = _project(root, identifier)
    if (
        blockers
        or value != current
        or any((out / n).read_bytes() != b for n, b in payloads.items())
    ):
        raise ValueError("reference_publication_requires_current_reviewed_evidence")
    return payloads
