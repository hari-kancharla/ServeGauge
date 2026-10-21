"""Read sealed observatory history without dispatch or operating admission."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import RollingTransportPolicy, load_rolling_policy
from servegauge.observatory.schedule import ObservatoryConfig, cadence_window, window_id
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.tick import _verified_job
from servegauge.observatory.transport import TransportBatch, transport_batch
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.transport_history import TransportHistoryJournal
from servegauge.suites.s1_protocol import protocol_items

if TYPE_CHECKING:
    from servegauge.observatory.admission import VerifiedAdmission


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _job_plan(job: dict[str, Any], config: ObservatoryConfig, identifier: str) -> RunPlan:
    model = next((m for m in config.models if m.id == job["model_id"]), None)
    if model is None:
        raise ValueError("history_model_binding_mismatch")
    plan = RunPlan.model_validate(job["plan"])
    seed = int(content_hash(["observatory-job-v1", config.seed, identifier, model.id])[:15], 16)
    if (
        job["run_id"] != "monitor-" + content_hash([identifier, model.id])
        or plan.plan_id != identifier
        or plan.seed != seed
        or plan.profile != model.profile
        or {e.id for e in plan.endpoints} != {e.id for e in model.endpoints}
        or any(
            (e.kind, e.base_url, e.model, e.defaults, e.claimed_profile)
            != (p.kind, p.base_url, p.model, p.defaults, p.claimed_profile)
            for e, p in zip(
                sorted(plan.endpoints, key=lambda e: e.id),
                sorted(model.endpoints, key=lambda e: e.id),
                strict=True,
            )
        )
    ):
        raise ValueError("history_job_binding_mismatch")
    return plan


def history_inputs(
    root: Path,
    observatory_id: str,
    *,
    policy: RollingTransportPolicy,
    admission: VerifiedAdmission | None = None,
) -> tuple[str, list[TemporalFrame], dict[str, Any], dict[str, Any]]:
    """Read complete or explicitly unavailable windows; never collect missing evidence."""
    source = implementation_hash()
    raw = root / "data/raw"
    parent = raw / "observatory"
    if any(p.is_symlink() for p in (root / "data", raw, parent)):
        raise ValueError("history_raw_symlink")
    frames: list[TemporalFrame] = []
    inputs: dict[str, Any] = {}
    streams: dict[str, Any] = {}
    for directory in sorted(parent.iterdir()) if parent.exists() else []:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("history_window_directory_invalid")
        journal = MonitorJournal.from_directory(raw, directory)
        spec = journal.spec()
        if spec.get("observatory_id") != observatory_id:
            continue
        config = ObservatoryConfig.model_validate(spec["config"])
        window = cadence_window(spec["cadence"], datetime.fromisoformat(spec["start"]))
        identifier = window_id(config.id, window)
        binding = spec["binding"]
        if (
            spec.get("version") != "observatory-window-v1"
            or config.id != observatory_id
            or journal.directory != MonitorJournal(raw, identifier).directory
            or spec.get("window_id") != identifier
            or window.start.isoformat() != spec["start"]
            or window.end.isoformat() != spec["end"]
            or binding.get("config_hash") != content_hash(spec["config"])
            or not _digest(binding.get("source_hash"))
            or not _digest(binding.get("measurement_hash"))
            or len({j["run_id"] for j in spec["jobs"]}) != len(spec["jobs"])
            or {j["model_id"] for j in spec["jobs"]} != {m.id for m in config.models}
        ):
            raise ValueError("history_window_binding_mismatch")
        events = journal.events()
        completed_positions = [
            i for i, e in enumerate(events) if e.get("type") == "window_completed"
        ]
        if completed_positions and completed_positions != [len(events) - 1]:
            raise ValueError("history_completion_is_not_terminal")
        complete = bool(events and events[-1].get("type") == "window_completed")
        resumed = any(e.get("type") == "window_resumed" for e in events)
        summary = events[-1]["result"] if complete else {}
        if complete and (
            summary.get("window_id") != identifier
            or summary.get("resumed") is not resumed
            or len(summary.get("jobs", [])) != len(spec["jobs"])
            or {j["run_id"] for j in summary["jobs"]} != {j["run_id"] for j in spec["jobs"]}
        ):
            raise ValueError("history_completion_binding_mismatch")
        inputs[identifier] = {
            "spec_hash": content_hash(spec),
            "state_hash": content_hash(events),
            "event_hashes": [content_hash(e) for e in events],
            "complete": complete,
        }
        for job in spec["jobs"]:
            plan = _job_plan(job, config, identifier)
            model = next(m for m in config.models if m.id == job["model_id"])
            analysis = None
            if complete:
                result = next(r for r in summary["jobs"] if r["run_id"] == job["run_id"])
                _verified_job(root, result)
                analysis = reconstruct(raw, job["run_id"])
                actual = analysis.plan.model_copy(deep=True)
                for endpoint in actual.endpoints:
                    endpoint.metadata.pop("observed_capabilities", None)
                if actual != plan:
                    raise ValueError("history_capture_plan_changed")
            for endpoint in plan.endpoints:
                stream = "transport-" + content_hash(
                    [observatory_id, model.id, endpoint.id, window.cadence]
                )
                streams[stream] = {
                    "model_id": model.id,
                    "endpoint_id": endpoint.id,
                    "cadence": window.cadence,
                }
                if analysis is None:
                    batch = TransportBatch(
                        False,
                        "window_incomplete",
                        job["run_id"],
                        endpoint.id,
                        0,
                        policy.batch_sizes[plan.level] if plan.level in {"core", "full"} else None,
                    )
                else:
                    batch = transport_batch(
                        analysis,
                        endpoint.id,
                        producer_source_hash=binding["source_hash"],
                        resumed=resumed,
                        faults=model.faults or FaultConfig(),
                        policy=policy,
                        admission=admission,
                        producer_binding=binding,
                    )
                admitted = admission is not None and admission.matches_binding(binding)
                if admission is not None and plan.level in {"core", "full"}:
                    # Even unavailable scheduled rows consume known design/seed scope.
                    canonical = [
                        i
                        for i in protocol_items(plan.level, plan.seed)
                        if i.expected.get("check_id") == "P01"
                    ]
                    batch = replace(
                        batch,
                        role_seed=int(content_hash([plan.seed, endpoint.id])[:15], 16),
                        request_design_hash=content_hash(
                            [i.model_dump(mode="json") for i in canonical]
                        ),
                    )
                frames.append(
                    TemporalFrame(
                        stream,
                        identifier,
                        window.cadence,
                        window.start,
                        window.end,
                        binding["source_hash"],
                        batch,
                        admission.measurement_source_hash if admitted and admission else None,
                        admission.runtime_hash if admitted and admission else None,
                    )
                )
    if implementation_hash() != source:
        raise ValueError("history_source_changed_during_inspection")
    return source, frames, inputs, streams


def history_projection(
    root: Path, observatory_id: str, *, policy: RollingTransportPolicy | None = None
) -> dict[str, Any]:
    """Candidate compatibility API: current-source only, always unqualified."""
    policy = policy or load_rolling_policy()
    source, frames, inputs, streams = history_inputs(root, observatory_id, policy=policy)
    steps = replay_candidates(frames, policy, expected_source_hash=source)
    value = {
        "version": "transport-history-v1",
        "observatory_id": observatory_id,
        "source_hash": source,
        "policy_hash": content_hash(policy.model_dump(mode="json")),
        "qualified": False,
        "verdict": "INCONCLUSIVE",
        "inputs": inputs,
        "streams": streams,
        "steps": [asdict(step) for step in steps],
    }
    if implementation_hash() != source:
        raise ValueError("history_source_changed_during_inspection")
    return {**value, "projection_id": "transport-history-" + content_hash(value)}


def inspect_history(root: Path, observatory_id: str, *, persist: bool = True) -> dict[str, Any]:
    value = history_projection(root, observatory_id)
    recorded = (
        TransportHistoryJournal(root / "data/raw", observatory_id).record(value)
        if persist
        else False
    )
    return {
        "status": "recorded" if recorded else "unchanged" if persist else "read_only",
        "projection": value,
    }
