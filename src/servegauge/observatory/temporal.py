"""Deterministic continuous candidate replay; operating admission stays separate."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, TypedDict

from servegauge.core.hashing import content_hash
from servegauge.observatory.rolling import RollingTransportPolicy, first_count_cut
from servegauge.observatory.schedule import Cadence, cadence_window
from servegauge.observatory.transport import TransportBatch


@dataclass(frozen=True)
class TemporalFrame:
    stream_id: str
    window_id: str
    cadence: Cadence
    start: datetime
    end: datetime
    producer_source_hash: str | None
    batch: TransportBatch
    measurement_source_hash: str | None = None
    producer_runtime_hash: str | None = None


@dataclass(frozen=True)
class CandidateChange:
    event_id: str
    stream_id: str
    input_window_ids: tuple[str, ...]
    input_hashes: tuple[str, ...]
    observation_start: str
    observation_end: str
    left_n: int
    right_n: int
    left_successes: int
    right_successes: int
    difference: float
    bound: float
    ci_low: float
    ci_high: float
    baseline_after_window_id: str
    policy_hash: str
    reference_tier: Literal["MOCK"] = field(default="MOCK", init=False)
    qualified: Literal[False] = field(default=False, init=False)


@dataclass(frozen=True)
class CandidateStep:
    stream_id: str
    window_id: str
    horizon_start: str
    horizon_end: str
    update_index: int
    status: Literal["BASELINE", "NO_CANDIDATE", "CANDIDATE", "INCONCLUSIVE"]
    reason: str
    event: CandidateChange | None = None
    qualified: Literal[False] = field(default=False, init=False)
    verdict: Literal["INCONCLUSIVE"] = field(default="INCONCLUSIVE", init=False)


class _StepContext(TypedDict):
    stream_id: str
    window_id: str
    horizon_start: str
    horizon_end: str
    update_index: int


@dataclass
class _State:
    frames: list[TemporalFrame] = field(default_factory=list)
    last: TemporalFrame | None = None


# Variant changes consume a new stream budget, never a free alpha reset.
Variant = tuple[str, str | None, str, str | None, int]


def _digest(value: str | None) -> str | None:
    return (
        value
        if value and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        else None
    )


def _variant(frame: TemporalFrame) -> Variant:
    return (
        frame.stream_id,
        _measurement_source(frame),
        frame.batch.endpoint_id,
        _digest(frame.batch.request_design_hash),
        frame.batch.expected_units if type(frame.batch.expected_units) is int else frame.batch.n,
    )


def _measurement_source(frame: TemporalFrame) -> str | None:
    actual = _digest(frame.producer_source_hash)
    if actual is None:
        return None
    return (
        actual if frame.measurement_source_hash is None else _digest(frame.measurement_source_hash)
    )


def _frame_hash(frame: TemporalFrame) -> str:
    value = {
        "stream": frame.stream_id,
        "window": frame.window_id,
        "cadence": frame.cadence,
        "start": frame.start.astimezone(UTC).isoformat(),
        "end": frame.end.astimezone(UTC).isoformat(),
        "producer": frame.producer_source_hash,
        "batch": asdict(frame.batch),
    }
    if frame.measurement_source_hash is not None:
        value["measurement_source"] = frame.measurement_source_hash
    if frame.producer_runtime_hash is not None:
        value["producer_runtime"] = frame.producer_runtime_hash
    return content_hash(value)


def _inputs(
    frames: Sequence[TemporalFrame], policy: RollingTransportPolicy
) -> list[tuple[TemporalFrame, int]]:
    unique: dict[tuple[str, str], TemporalFrame] = {}
    times: dict[tuple[str, datetime], str] = {}
    cadences: dict[str, Cadence] = {}
    for frame in frames:
        if any(t.tzinfo is None or t.utcoffset() is None for t in (frame.start, frame.end)):
            raise ValueError("aware_temporal_windows_required")
        if not frame.stream_id or not frame.window_id:
            raise ValueError("temporal_identity_required")
        window = cadence_window(frame.cadence, frame.start)
        if window.start != frame.start or window.end != frame.end:
            raise ValueError("canonical_temporal_window_required")
        key = (frame.stream_id, frame.window_id)
        if key in unique:
            if _frame_hash(unique[key]) != _frame_hash(frame):
                raise ValueError("conflicting_temporal_window")
            continue
        temporal_key = (frame.stream_id, window.start)
        if temporal_key in times:
            raise ValueError("colliding_temporal_windows")
        if frame.stream_id in cadences and cadences[frame.stream_id] != frame.cadence:
            raise ValueError("temporal_stream_identity_changed")
        cadences[frame.stream_id] = frame.cadence
        times[temporal_key] = frame.window_id
        unique[key] = frame
    active: deque[tuple[datetime, Variant]] = deque()
    counts: Counter[Variant] = Counter()
    result = []
    for frame in sorted(unique.values(), key=lambda f: (f.start, f.stream_id, f.window_id)):
        lower = frame.start - timedelta(days=policy.horizon_days)
        while active and active[0][0] <= lower:
            _, old = active.popleft()
            counts[old] -= 1
            if not counts[old]:
                del counts[old]
        variant = _variant(frame)
        active.append((frame.start, variant))
        counts[variant] += 1
        if len(counts) > policy.max_streams:
            raise ValueError("temporal_stream_budget_exceeded")
        if counts[variant] > policy.max_updates:
            raise ValueError("temporal_update_budget_exceeded")
        result.append((frame, counts[variant]))
    return result


def _unavailable(frame: TemporalFrame, policy: RollingTransportPolicy, expected: str) -> str | None:
    batch = frame.batch
    if _measurement_source(frame) != expected or (
        frame.producer_runtime_hash is not None and _digest(frame.producer_runtime_hash) is None
    ):
        return "source_not_qualified"
    if not batch.ready:
        return "transport_batch_unavailable"
    if (
        batch.n not in policy.batch_sizes.values()
        or type(batch.expected_units) is not int
        or type(batch.observed_units) is not int
        or batch.expected_units != batch.n
        or batch.observed_units != batch.n
        or any(type(bit) is not int or bit not in {0, 1} for bit in batch.values)
        or batch.reference_tier != "MOCK"
        or not _digest(batch.request_design_hash)
        or not _digest(batch.capture_hash)
        or not batch.run_id
        or not batch.endpoint_id
        or type(batch.role_seed) is not int
        or batch.role_seed < 0
    ):
        return "unsupported_transport_candidate"
    return None


def replay_candidates(
    frames: Sequence[TemporalFrame], policy: RollingTransportPolicy, *, expected_source_hash: str
) -> list[CandidateStep]:
    """Replay the full candidate policy. Outputs stay unqualified and INCONCLUSIVE."""
    if _digest(expected_source_hash) is None:
        raise ValueError("expected_temporal_source_required")
    inputs = _inputs(frames, policy)  # Refuse the whole conflicting/over-budget input first.
    states: dict[str, _State] = {}
    seeds: dict[str, set[int]] = defaultdict(set)
    captures: dict[str, set[str]] = defaultdict(set)
    result = []
    policy_hash = content_hash(policy.model_dump(mode="json"))
    for frame, update_index in inputs:
        state = states.setdefault(frame.stream_id, _State())
        unavailable = _unavailable(frame, policy, expected_source_hash)
        seed, capture = frame.batch.role_seed, frame.batch.capture_hash
        if type(seed) is int and seed >= 0:
            if seed in seeds[frame.stream_id]:
                unavailable = unavailable or "reused_transport_role_seed"
            seeds[frame.stream_id].add(seed)
        if _digest(capture) is not None:
            assert capture is not None
            if capture in captures[frame.stream_id]:
                unavailable = unavailable or "reused_transport_capture"
            captures[frame.stream_id].add(capture)
        common: _StepContext = {
            "stream_id": frame.stream_id,
            "window_id": frame.window_id,
            "horizon_start": (frame.start - timedelta(days=policy.horizon_days))
            .astimezone(UTC)
            .isoformat(),
            "horizon_end": frame.start.astimezone(UTC).isoformat(),
            "update_index": update_index,
        }
        previous = state.last
        state.last = frame
        if unavailable:
            state.frames = []
            result.append(CandidateStep(**common, status="INCONCLUSIVE", reason=unavailable))
            continue
        baseline_reason = "prior_window_required"
        if previous and previous.end != frame.start:
            state.frames = []
            baseline_reason = "missing_calendar_window"
        elif state.frames and _variant(state.frames[-1]) != _variant(frame):
            state.frames = []
            baseline_reason = "transport_measurement_changed"
        state.frames.append(frame)
        state.frames = state.frames[-policy.max_history_batches :]
        if len(state.frames) == 1:
            result.append(CandidateStep(**common, status="BASELINE", reason=baseline_reason))
            continue
        cut = first_count_cut([sum(f.batch.values) for f in state.frames], frame.batch.n, policy)
        if cut is None:
            result.append(CandidateStep(**common, status="NO_CANDIDATE", reason="no_boundary_cut"))
            continue
        hashes = tuple(_frame_hash(f) for f in state.frames)
        event = CandidateChange(
            event_id="candidate-" + content_hash([frame.stream_id, hashes, policy_hash]),
            stream_id=frame.stream_id,
            input_window_ids=tuple(f.window_id for f in state.frames),
            input_hashes=hashes,
            observation_start=state.frames[0].start.astimezone(UTC).isoformat(),
            observation_end=frame.end.astimezone(UTC).isoformat(),
            left_n=cut.left_n,
            right_n=cut.right_n,
            left_successes=cut.left_successes,
            right_successes=cut.right_successes,
            difference=cut.difference,
            bound=cut.bound,
            ci_low=cut.ci_low,
            ci_high=cut.ci_high,
            baseline_after_window_id=frame.window_id,
            policy_hash=policy_hash,
        )
        result.append(
            CandidateStep(
                **common, status="CANDIDATE", reason="candidate_boundary_cut", event=event
            )
        )
        state.frames = [frame]
    return result
