"""Source-qualified local transport changes; endpoint quality remains inconclusive."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.observatory.admission import VerifiedAdmission, load_admission
from servegauge.observatory.history import history_inputs
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import RollingTransportPolicy, load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame, replay_candidates


@dataclass(frozen=True, init=False)
class OperatingEvaluation:
    """Immutable result capability created only by the admitted replay path."""

    _payload: bytes

    def value(self) -> dict[str, Any]:
        return dict(json.loads(self._payload))


def evaluate_frames(
    observatory_id: str,
    frames: list[TemporalFrame],
    inputs: dict[str, Any],
    streams: dict[str, Any],
    admission: VerifiedAdmission,
    policy: RollingTransportPolicy,
) -> OperatingEvaluation:
    """Pure sealed-input projection; callers obtain frames through history_inputs."""
    source = implementation_hash()
    if source != admission.current_source_hash:
        raise ValueError("operating_consumer_source_changed")
    policy_hash = content_hash(policy.model_dump(mode="json"))
    if policy_hash != admission.policy_hash:
        raise ValueError("operating_policy_not_admitted")
    if {frame.window_id for frame in frames} != set(inputs):
        raise ValueError("operating_input_window_missing")
    for item in inputs.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"spec_hash", "state_hash", "event_hashes", "complete"}
            or type(item["complete"]) is not bool
            or not isinstance(item["event_hashes"], list)
            or any(
                not isinstance(d, str)
                or len(d) != 64
                or any(c not in "0123456789abcdef" for c in d)
                for d in [item["spec_hash"], item["state_hash"], *item["event_hashes"]]
            )
        ):
            raise ValueError("operating_input_binding_invalid")
    reviewed = []
    for frame in frames:
        if (
            not admission.admits_source(frame.producer_source_hash)
            or frame.producer_runtime_hash != admission.runtime_hash
            or frame.measurement_source_hash != admission.measurement_source_hash
        ):
            frame = replace(
                frame,
                batch=replace(frame.batch, ready=False, reason="operating_provenance_not_admitted"),
            )
        reviewed.append(frame)
    steps = replay_candidates(
        reviewed, policy, expected_source_hash=admission.measurement_source_hash
    )
    by_key = {(f.stream_id, f.window_id): f for f in reviewed}
    events = []
    for step in steps:
        candidate = step.event
        if candidate is None:
            continue
        contrast = asdict(candidate)
        contrast.pop("qualified")
        contrast.pop("reference_tier")
        provenance = {
            w: by_key[(step.stream_id, w)].producer_source_hash for w in candidate.input_window_ids
        }
        event_id = "transport-event-" + content_hash(
            [
                "transport-operating-event-v1",
                observatory_id,
                step.stream_id,
                step.window_id,
                policy_hash,
            ]
        )
        events.append(
            {
                "version": "transport-operating-event-v1",
                "event_id": event_id,
                "observatory_id": observatory_id,
                "stream_id": step.stream_id,
                "ending_window_id": step.window_id,
                "policy_hash": policy_hash,
                "qualified": True,
                "reference_tier": "MOCK",
                "quality_verdict": "INCONCLUSIVE",
                "kind": "transport_change",
                "interval_scope": "fixed_declared_cadence_family",
                "family_start": step.horizon_start,
                "family_end": step.horizon_end,
                "family_alpha": policy.family_alpha,
                "max_false_events_30_days": policy.max_false_events_30_days,
                "measurement_source_hash": admission.measurement_source_hash,
                "generator_hash": admission.generator_hash,
                "runtime_hash": admission.runtime_hash,
                "producer_sources": provenance,
                "contrast": contrast,
                "qualification": {
                    "id": admission.id,
                    "receipt": admission.receipt_path,
                    "sha256": admission.receipt_hash,
                },
            }
        )
    value = {
        "version": "transport-operating-evaluation-v1",
        "observatory_id": observatory_id,
        "source_hash": source,
        "policy_hash": policy_hash,
        "qualification": {
            "id": admission.id,
            "receipt": admission.receipt_path,
            "sha256": admission.receipt_hash,
        },
        "inputs": inputs,
        "window_starts": {f.window_id: f.start.astimezone(UTC).isoformat() for f in reviewed},
        "streams": streams,
        "steps": [asdict(s) for s in steps],
        "events": events,
        "quality_verdict": "INCONCLUSIVE",
    }
    value["evaluation_id"] = "transport-evaluation-" + content_hash(value)
    if implementation_hash() != source:
        raise ValueError("operating_consumer_source_changed")
    result = object.__new__(OperatingEvaluation)
    object.__setattr__(result, "_payload", canonical_json(value))
    return result


def evaluate_history(
    root: Path, observatory_id: str, *, proof_root: Path = Path("."), persist: bool = True
) -> dict[str, Any]:
    """Evaluate existing sealed observations only; never dispatch or contact anyone."""
    if any(p.lower() == "starter files" for p in [*root.parts, *proof_root.parts]):
        raise ValueError("operating_path_refused")
    state = load_admission(proof_root)
    if state.admission is None:
        return {
            "status": "unqualified",
            "reason": state.reason,
            "events": [],
            "quality_verdict": "INCONCLUSIVE",
        }
    admission = state.admission
    policy = load_rolling_policy()
    _, frames, inputs, streams = history_inputs(
        root, observatory_id, policy=policy, admission=admission
    )
    result = evaluate_frames(observatory_id, frames, inputs, streams, admission, policy)
    # Import at the side-effect boundary; the pure event schema has no storage dependency.
    from servegauge.storage.transport_events import TransportEventJournal

    recorded = (
        TransportEventJournal(root / "data/raw", observatory_id).record(result)
        if persist
        else False
    )
    value = result.value()
    return {
        "status": "recorded" if recorded else "unchanged" if persist else "read_only",
        "evaluation": value,
        "events": value["events"],
        "quality_verdict": "INCONCLUSIVE",
    }
