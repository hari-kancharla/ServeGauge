"""Private, pinned raw sources for a future owner comparison; never dispatch traffic."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
from zstandard import ZstdError

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import CheckResult, Exchange, Item, SseEvent, ValueModel
from servegauge.core.redaction import Redactor
from servegauge.core.secrets import PATTERNS, scan_paths
from servegauge.storage.analysis import Analysis, reconstruct
from servegauge.storage.raw_store import _chunks
from servegauge.transport.http import assemble_stream

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SourcePin(ValueModel):
    run_id: str = Field(min_length=1)
    files_sha256: dict[str, Digest] = Field(min_length=1)


class CaseRequest(ValueModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    run_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    check_id: str = Field(min_length=1)


class CaseSelection(CaseRequest):
    check_hash: Digest


class ComparisonRequest(ValueModel):
    cases: list[CaseRequest] = Field(min_length=2, max_length=2)


class ComparisonSelection(ValueModel):
    version: Literal["comparison-selection-v1"] = "comparison-selection-v1"
    sources: list[SourcePin] = Field(min_length=1)
    cases: list[CaseSelection] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def distinct_bound_cases(self) -> ComparisonSelection:
        runs = [source.run_id for source in self.sources]
        keys = [(c.run_id, c.endpoint_id, c.check_id) for c in self.cases]
        if (
            len(set(runs)) != len(runs)
            or set(runs) != {c.run_id for c in self.cases}
            or len({c.case_id for c in self.cases}) != 2
            or len(set(keys)) != 2
        ):
            raise ValueError("comparison_distinct_explicit_cases_required")
        return self


def _guard(path: Path) -> None:
    if ".." in path.parts or any(p.casefold() == "starter files" for p in path.parts):
        raise ValueError("comparison_path_refused")
    if any(p.is_symlink() for p in [*reversed(path.absolute().parents), path]):
        raise ValueError("comparison_symlink_refused")


def _relative(root: Path, name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or ".." in value.parts or "\\" in name:
        raise ValueError("comparison_relative_file_required")
    path = root.joinpath(*value.parts)
    _guard(path)
    return path


def _files(root: Path) -> list[Path]:
    _guard(root)
    if not root.is_dir():
        raise ValueError("comparison_source_directory_missing")
    result = []
    for path in sorted(root.iterdir()):
        _guard(path)
        if path.is_dir():
            result.extend(_files(path))
        elif path.is_file():
            result.append(path)
        else:
            raise ValueError("comparison_special_file_refused")
    return result


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("comparison_json_duplicate_keys_refused")
    return value


def _private_json(text: str | bytes) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("comparison_json_privacy_decoding_unavailable") from None
    canonical_json(value)  # Non-finite values cannot become normalized model data.
    return value


def _require_sanitized(value: Any, redactor: Redactor | None = None) -> None:
    scrubber = redactor or Redactor.from_environment()
    encoded = canonical_json(value)
    if (
        scrubber._contains_credential(value)
        or scrubber.redact(value) != value
        or any(pattern.search(encoded) for pattern in PATTERNS)
    ):
        # Refuse rather than changing a source observation or its reviewed hashes.
        raise ValueError("comparison_decoded_credential_like_source_refused")


def _private_records(path: Path, redactor: Redactor) -> Iterator[Exchange]:
    pending = b""
    try:
        for chunk in _chunks(path):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.strip():
                    value = _private_json(line)
                    _require_sanitized(value, redactor)
                    yield Exchange.model_validate(value)
        if pending:
            raise ValueError("comparison_incomplete_raw_line")
    except ZstdError:
        raise ValueError("comparison_raw_privacy_decoding_unavailable") from None


def _require_stream_sanitized(events: list[SseEvent], redactor: Redactor) -> None:
    tokens: dict[str, list[Any]] = {}
    for event in events:
        if event.is_comment or not event.data.strip() or event.data.strip() == "[DONE]":
            continue
        try:
            chunk = _private_json(event.data)
        except ValueError:
            raise ValueError("comparison_sse_privacy_decoding_unavailable") from None
        _require_sanitized(chunk, redactor)
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices", []), list):
            continue
        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict) or not isinstance(choice.get("logprobs"), dict):
                continue
            content = choice["logprobs"].get("content")
            if isinstance(content, list):
                tokens.setdefault(content_hash(choice.get("index", 0)), []).extend(content)
    for rows in tokens.values():
        _require_sanitized({"response": {"captured_logprob_tokens": rows}}, redactor)
    response, _ = assemble_stream(events)
    _require_sanitized({"response": response}, redactor)


def _envelope(path: Path) -> dict[str, Any]:
    _guard(path)
    envelope = _private_json(path.read_bytes())
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"value", "hash"}
        or not isinstance(envelope["value"], dict)
        or content_hash(envelope["value"]) != envelope["hash"]
    ):
        raise ValueError("comparison_seal_invalid")
    _require_sanitized(envelope["value"])
    return dict(envelope["value"])


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError("comparison_run_support_invalid")
    return list(value)


def _support_runs(run_id: str, sealed: dict[str, Any]) -> list[str]:
    if sealed.get("version") != "analysis-v1" or sealed.get("run_id") != run_id:
        raise ValueError("comparison_analysis_identity_invalid")
    metadata = sealed.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("run_id") != run_id:
        raise ValueError("comparison_metadata_identity_invalid")
    expected = {run_id}
    caps = metadata.get("capability_run_id")
    if caps is not None:
        expected.update(_strings([caps]))
    for field in ["replay_runs", "replay_batch_ids", "replay_by_endpoint", "agent_traces"]:
        table = metadata.get(field, {})
        if not isinstance(table, dict):
            raise ValueError("comparison_run_support_invalid")
        for value in table.values():
            if field == "replay_runs":
                expected.update(_strings([value]))
            elif field == "replay_batch_ids":
                expected.update(_strings(value))
            else:
                if not isinstance(value, dict):
                    raise ValueError("comparison_run_support_invalid")
                for nested in value.values():
                    if field == "agent_traces":
                        if not isinstance(nested, dict):
                            raise ValueError("comparison_run_support_invalid")
                        expected.update(_strings(nested.get("run_ids", [])))
                    else:
                        expected.update(_strings(nested))
    declared = _strings(sealed.get("run_ids"))
    if set(declared) != expected or len(set(declared)) != len(declared):
        raise ValueError("comparison_complete_run_support_required")
    return sorted(expected)


def pin_source(root: Path, run_id: str) -> SourcePin:
    """Describe bytes for review; a pin is not owner approval or a comparison result."""
    _guard(root)
    raw = root / "data/raw"
    sealed = _envelope(raw / safe_filename(run_id) / "analysis.json")
    paths = [p for run in _support_runs(run_id, sealed) for p in _files(raw / safe_filename(run))]
    for path in paths:
        parts = path.relative_to(raw).parts
        if not (
            (len(parts) == 2 and parts[-1] in {"capture.json", "analysis.json", "reference.json"})
            or (len(parts) == 3 and parts[-1].endswith((".jsonl", ".jsonl.zst")))
        ):
            raise ValueError("comparison_unsupported_source_file")
    return SourcePin(
        run_id=run_id,
        files_sha256={p.relative_to(raw).as_posix(): _sha(p.read_bytes()) for p in paths},
    )


@dataclass
class _Source:
    pin: SourcePin
    blobs: dict[str, bytes]
    analysis: Analysis
    records: dict[str, Exchange]


def _load(root: Path, pin: SourcePin) -> _Source:
    _guard(root)
    raw = root / "data/raw"
    # Validate the reviewed primary seal before allowing it to select other paths.
    primary = (Path(safe_filename(pin.run_id)) / "analysis.json").as_posix()
    path = _relative(raw, primary)
    if primary not in pin.files_sha256 or _sha(path.read_bytes()) != pin.files_sha256[primary]:
        raise ValueError("comparison_source_pin_changed")
    for name in pin.files_sha256:
        _relative(raw, name)
    if pin_source(root, pin.run_id) != pin:
        raise ValueError("comparison_source_pin_changed")
    blobs = {name: _relative(raw, name).read_bytes() for name in pin.files_sha256}
    if any(_sha(blob) != pin.files_sha256[name] for name, blob in blobs.items()):
        raise ValueError("comparison_source_pin_changed")
    if scan_paths([_relative(raw, name) for name in pin.files_sha256]):
        raise ValueError("comparison_credential_like_source_refused")
    sealed = _envelope(path)
    runs = _support_runs(pin.run_id, sealed)
    contexts = {run: _envelope(raw / safe_filename(run) / "capture.json") for run in runs}
    items: dict[str, dict[tuple[str, str], Item]] = {}
    for run, context in contexts.items():
        if context.get("manifest", {}).get("run_id") != run:
            raise ValueError("comparison_capture_identity_changed")
        inventory = [Item.model_validate(i) for i in context["items"]]
        items[run] = {(i.suite_id, i.item_id): i for i in inventory}
        if len(items[run]) != len(inventory):
            raise ValueError("comparison_ambiguous_item_inventory")
    records: dict[str, Exchange] = {}
    slots: set[tuple[str, str, str, str, int]] = set()
    redactor = Redactor.from_environment()
    for name in sorted(blobs):
        if not name.endswith((".jsonl", ".jsonl.zst")):
            continue
        for record in _private_records(_relative(raw, name), redactor):
            for events in [record.sse_events, *(a.sse_events for a in record.attempts)]:
                if events:
                    _require_stream_sanitized(events, redactor)
            parts = PurePosixPath(name).parts
            if parts[1] != safe_filename(record.endpoint_id) or parts[-1] not in {
                safe_filename(record.suite_id) + ".jsonl",
                safe_filename(record.suite_id) + ".jsonl.zst",
            }:
                raise ValueError("comparison_record_file_scope_changed")
            slot = (
                record.run_id,
                record.endpoint_id,
                record.suite_id,
                record.item_id,
                record.sample_idx,
            )
            if record.exchange_id in records or slot in slots:
                raise ValueError("comparison_ambiguous_duplicate_record")
            if record.run_id not in contexts or PurePosixPath(name).parts[0] != safe_filename(
                record.run_id
            ):
                raise ValueError("comparison_record_run_changed")
            item = items[record.run_id].get((record.suite_id, record.item_id))
            samples = item.expected.get("samples", 1) if item else None
            if type(samples) is not int or samples < 1 or record.sample_idx >= samples:
                raise ValueError("comparison_record_inventory_changed")
            if [attempt.n for attempt in record.attempts] != list(
                range(1, len(record.attempts) + 1)
            ):
                raise ValueError("comparison_attempt_order_unknown")
            records[record.exchange_id] = record
            slots.add(slot)
    for run, context in contexts.items():
        for item in items[run].values():
            if item.suite_id != "S0":
                continue
            samples = item.expected.get("samples", 1)
            if type(samples) is not int or samples < 1:
                raise ValueError("comparison_preflight_inventory_unknown")
            for endpoint in context["manifest"]["config"]["endpoints"]:
                if any(
                    (run, endpoint["id"], item.suite_id, item.item_id, i) not in slots
                    for i in range(samples)
                ):
                    raise ValueError("comparison_preflight_support_missing")
    if {k: content_hash(v.model_dump(mode="json")) for k, v in records.items()} != sealed[
        "record_hashes"
    ]:
        raise ValueError("comparison_complete_raw_support_required")
    for endpoint_id, tasks in sealed["metadata"].get("agent_traces", {}).items():
        for trace in tasks.values():
            identifiers = _strings(trace.get("exchange_ids", []))
            trace_runs = _strings(trace.get("run_ids", []))
            if (trace_runs and not identifiers) or any(
                identifier not in records
                or records[identifier].endpoint_id != endpoint_id
                or records[identifier].run_id not in trace_runs
                for identifier in identifiers
            ):
                raise ValueError("comparison_agent_support_missing_or_ambiguous")
    analysis = reconstruct(raw, pin.run_id)
    if any(r.endpoint_id not in {e.id for e in analysis.plan.endpoints} for r in records.values()):
        raise ValueError("comparison_record_endpoint_unknown")
    if any(_relative(raw, name).read_bytes() != blob for name, blob in blobs.items()):
        raise ValueError("comparison_source_changed_during_read")
    return _Source(pin, blobs, analysis, records)


def _case(source: _Source, case: CaseSelection) -> dict[str, Any]:
    analysis = source.analysis
    found = [
        c
        for c in analysis.checks
        if c.endpoint_id == case.endpoint_id and c.check_id == case.check_id
    ]
    if len(found) != 1:
        raise ValueError("comparison_unique_source_check_required")
    check: CheckResult = found[0]
    if (
        check.run_id != case.run_id
        or check.suite_id != case.suite_id
        or content_hash(check.model_dump(mode="json")) != case.check_hash
        or check.verdict != "FAIL"
        or check.n == 0
        or check.ci_low is None
        or check.ci_high is None
    ):
        raise ValueError("comparison_pinned_recorded_failure_required")
    sample = analysis.samples.get(case.endpoint_id, {}).get(case.check_id)
    keys = check.details.get("item_keys")
    if (
        sample is None
        or not isinstance(keys, list)
        or keys != sorted(sample.values)
        or len(keys) != check.n
    ):
        raise ValueError("comparison_original_logical_units_required")
    evidence = set(check.evidence)
    if not evidence or not evidence <= source.records.keys():
        raise ValueError("comparison_source_evidence_missing")
    target = [source.records[k] for k in evidence]
    if any(r.endpoint_id != case.endpoint_id or r.suite_id != case.suite_id for r in target):
        raise ValueError("comparison_source_evidence_scope_changed")
    metadata = analysis.metadata
    by_endpoint = metadata.get("replay_by_endpoint", {})
    replays = by_endpoint.get(case.endpoint_id, metadata.get("replay_batch_ids", {}))
    confirmations = _strings(replays.get(case.check_id, []))
    traces = metadata.get("agent_traces", {}).get(case.endpoint_id, {})
    agent_runs: set[str] = set()
    for trace in traces.values():
        if evidence.intersection(trace.get("exchange_ids", [])):
            agent_runs.update(_strings(trace.get("run_ids", [])))
    roles = {
        case.run_id: "primary",
        **{r: "confirmation" for r in confirmations},
        **{r: "agent" for r in agent_runs},
    }
    observations: dict[str, tuple[str, Exchange]] = {}
    for record in target:
        if record.run_id not in roles:
            raise ValueError("comparison_evidence_support_phase_unknown")
        observations[record.exchange_id] = (roles[record.run_id], record)
    if not any(role == "primary" for role, _ in observations.values()):
        raise ValueError("comparison_primary_support_missing")
    for run in confirmations:
        confirmation_rows = [
            r
            for r in source.records.values()
            if r.run_id == run and r.endpoint_id == case.endpoint_id
        ]
        if not confirmation_rows or any(r.suite_id != case.suite_id for r in confirmation_rows):
            raise ValueError("comparison_confirmation_support_missing")
        for record in confirmation_rows:
            observations[record.exchange_id] = ("confirmation", record)
    baseline = metadata.get("baseline_id")
    if baseline is not None:
        for role, record in list(observations.values()):
            if role != "primary":
                continue
            paired = [
                r
                for r in source.records.values()
                if (r.run_id, r.endpoint_id, r.suite_id, r.item_id, r.sample_idx)
                == (case.run_id, baseline, record.suite_id, record.item_id, record.sample_idx)
            ]
            if len(paired) != 1:
                raise ValueError("comparison_paired_support_missing")
            observations[paired[0].exchange_id] = ("paired", paired[0])
    caps = metadata.get("capability_run_id") or case.run_id
    for record in source.records.values():
        if (
            record.run_id == caps
            and record.suite_id == "S0"
            and record.endpoint_id in {case.endpoint_id, baseline}
        ):
            observations[record.exchange_id] = ("preflight", record)
    positions = {key: i for i, key in enumerate(source.records)}
    rows = [
        {
            "role": role,
            "exchange_id": record.exchange_id,
            "run_id": record.run_id,
            "endpoint_id": record.endpoint_id,
            "suite_id": record.suite_id,
            "item_id": record.item_id,
            "sample_idx": record.sample_idx,
            "record_hash": content_hash(record.model_dump(mode="json")),
            "request_hash": content_hash(record.request),
            "recorded_attempts": len(record.attempts),
            "source_position": positions[record.exchange_id],
        }
        for role, record in sorted(
            observations.values(),
            key=lambda row: positions[row[1].exchange_id],
        )
    ]
    return {
        "version": "comparison-case-source-v1",
        "case": case.model_dump(mode="json"),
        "original_check": check.model_dump(mode="json"),
        "population_n": check.n,
        "logical_unit_keys": keys,
        "observations": rows,
        "recorded_attempts": sum(r["recorded_attempts"] for r in rows),
        "source_order_scope": "raw_file_order_within_endpoint_and_run",
        "replay_adds_independent_n": False,
    }


def propose_selection(root: Path, request: ComparisonRequest) -> ComparisonSelection:
    """Derive reviewable pins from raw failures; never assert owner approval."""
    _require_sanitized(request.model_dump(mode="json"))
    sources = {run: _load(root, pin_source(root, run)) for run in {c.run_id for c in request.cases}}
    cases = []
    for case in request.cases:
        checks = [
            check
            for check in sources[case.run_id].analysis.checks
            if check.endpoint_id == case.endpoint_id and check.check_id == case.check_id
        ]
        if len(checks) != 1:
            raise ValueError("comparison_unique_source_check_required")
        selected = CaseSelection(
            **case.model_dump(mode="json"),
            check_hash=content_hash(checks[0].model_dump(mode="json")),
        )
        _case(sources[case.run_id], selected)
        cases.append(selected)
    return ComparisonSelection(sources=[sources[run].pin for run in sorted(sources)], cases=cases)


def save_selection(root: Path, selection: ComparisonSelection, out: Path) -> None:
    """Save a proposed selection privately, without replacing any existing path."""
    _guard(root)
    _guard(out)
    private = root.absolute() / ".cache/comparisons"
    if not out.absolute().is_relative_to(private) or out.absolute() == private:
        raise ValueError("comparison_private_cache_destination_required")
    selected = ComparisonSelection.model_validate(selection.model_dump(mode="json"))
    _require_sanitized(selected.model_dump(mode="json"))
    encoded = canonical_json(selected.model_dump(mode="json")) + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(encoded)


def prepare_sources(root: Path, selection: ComparisonSelection, out: Path) -> dict[str, Any]:
    """Write a new private snapshot after validating every reviewed source and case."""
    _guard(root)
    _guard(out)
    selection = ComparisonSelection.model_validate(selection.model_dump(mode="json"))
    _require_sanitized(selection.model_dump(mode="json"))
    private = root.absolute() / ".cache/comparisons"
    if not out.absolute().is_relative_to(private) or out.absolute() == private or out.exists():
        raise ValueError("comparison_new_private_cache_destination_required")
    sources = {p.run_id: _load(root, p) for p in selection.sources}
    cases = {c.case_id: _case(sources[c.run_id], c) for c in selection.cases}
    for case in cases.values():
        _require_sanitized(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".comparison-", dir=out.parent))
    try:
        payloads = {"selection.json": canonical_json(selection.model_dump(mode="json")) + b"\n"}
        payloads.update(
            {f"cases/{key}.json": canonical_json(value) + b"\n" for key, value in cases.items()}
        )
        for run, source in sources.items():
            payloads.update(
                {
                    f"sources/{safe_filename(run)}/data/raw/{name}": blob
                    for name, blob in source.blobs.items()
                }
            )
        for name, blob in payloads.items():
            path = _relative(temporary, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(blob)
            path.chmod(0o600)
        manifest = {
            "version": "comparison-sources-v1",
            "state": "prepared_only",
            "owner_comparison_executed": False,
            "setup_seconds": None,
            "private_source_material": True,
            "source_runs": sorted(sources),
            "case_ids": sorted(cases),
            "selection_hash": content_hash(selection.model_dump(mode="json")),
            "files_sha256": {name: _sha(blob) for name, blob in sorted(payloads.items())},
        }
        (temporary / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        (temporary / "manifest.json").chmod(0o600)
        verify_sources(temporary, selection_hash=content_hash(selection.model_dump(mode="json")))
        if out.exists():
            raise ValueError("comparison_destination_created_during_preparation")
        temporary.rename(out)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_sources(out: Path, *, selection_hash: str) -> dict[str, Any]:
    """Reconstruct copied evidence and verify every file without changing the snapshot."""
    _guard(out)
    manifest = _private_json((out / "manifest.json").read_bytes())
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "version",
            "state",
            "owner_comparison_executed",
            "setup_seconds",
            "private_source_material",
            "source_runs",
            "case_ids",
            "selection_hash",
            "files_sha256",
        }
        or manifest.get("selection_hash") != selection_hash
        or len(selection_hash) != 64
        or manifest.get("version") != "comparison-sources-v1"
        or manifest.get("state") != "prepared_only"
        or manifest.get("owner_comparison_executed") is not False
        or manifest.get("setup_seconds") is not None
        or manifest.get("private_source_material") is not True
        or not isinstance(manifest.get("files_sha256"), dict)
    ):
        raise ValueError("comparison_preparation_manifest_invalid")
    actual = {
        p.relative_to(out).as_posix(): _sha(p.read_bytes())
        for p in _files(out)
        if p != out / "manifest.json"
    }
    if actual != manifest["files_sha256"]:
        raise ValueError("comparison_prepared_files_changed")
    selection = ComparisonSelection.model_validate(
        _private_json((out / "selection.json").read_bytes())
    )
    _require_sanitized(selection.model_dump(mode="json"))
    if content_hash(selection.model_dump(mode="json")) != manifest.get("selection_hash"):
        raise ValueError("comparison_prepared_selection_changed")
    if sorted(c.case_id for c in selection.cases) != manifest.get("case_ids") or sorted(
        p.run_id for p in selection.sources
    ) != manifest.get("source_runs"):
        raise ValueError("comparison_prepared_identity_changed")
    sources = {
        p.run_id: _load(out / "sources" / safe_filename(p.run_id), p) for p in selection.sources
    }
    expected_names = {"selection.json"}
    for source in sources.values():
        expected_names.update(
            f"sources/{safe_filename(source.pin.run_id)}/data/raw/{name}" for name in source.blobs
        )
    for case in selection.cases:
        name = f"cases/{case.case_id}.json"
        expected_names.add(name)
        if _private_json((out / name).read_bytes()) != _case(sources[case.run_id], case):
            raise ValueError("comparison_prepared_case_changed")
    if expected_names != set(actual):
        raise ValueError("comparison_unexpected_prepared_file")
    return dict(manifest)
