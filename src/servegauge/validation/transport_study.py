"""Predeclared family/readiness experiments; all inference is constructed offline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from servegauge.core.hashing import content_hash
from servegauge.observatory.bounds import floor_fraction
from servegauge.observatory.identity import producer_identity
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.transport import TransportBatch
from servegauge.stats.intervals import wilson
from servegauge.validation.common import offline_only
from servegauge.validation.transport_operating import CountCut, replay_counts

PLAN = Path("configs/monitor_operating_study_v1.json")


def _plan() -> dict[str, Any]:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    if (
        value.get("version") != "monitor-operating-study-v1"
        or type(value.get("replicates")) is not int
        or value["replicates"] < 2000
        or type(value.get("bootstrap_resamples")) is not int
        or value["bootstrap_resamples"] < 2000
        or type(value.get("held_out_replays")) is not int
        or value["held_out_replays"] < 120
        or value.get("min_power", 0) < 0.8
        or value.get("preflight_retries") != 2
        or not isinstance(value.get("namespace"), str)
        or not 1 <= value.get("family_chunk_size", 0) <= 100
        or len(value.get("family_cases", [])) != 4
        or len(value.get("readiness_cases", [])) != 3
    ):
        raise ValueError("unsupported_transport_study_plan")
    for case in [*value["family_cases"], *value["readiness_cases"]]:
        if not re.fullmatch(r"[a-z][a-z0-9-]+", case["id"]):
            raise ValueError("invalid_transport_study_case")
    return dict(value)


def _seed(plan: dict[str, Any], *role: object) -> int:
    return int(content_hash([plan["namespace"], *role])[:16], 16)


def _bootstrap(values: list[float], plan: dict[str, Any], label: str) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "resamples": 0}
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(_seed(plan, "bootstrap", label))
    means = []
    for _ in range(plan["bootstrap_resamples"]):
        means.append(float(np.mean(array[generator.integers(0, len(array), len(array))])))
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "resamples": plan["bootstrap_resamples"],
        "unit": "independent family or trajectory",
        "scope": (
            "Percentile bootstrap; a degenerate observed sample is not proof "
            "of zero operating risk."
        ),
    }


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _held_out(plan: dict[str, Any], replicates: int) -> dict[str, Any]:
    policy = load_rolling_policy()
    generator = np.random.default_rng(_seed(plan, "held-out-public-replay"))
    trajectories = min(replicates, plan["held_out_replays"])
    compared = 0
    for trial in range(trajectories):
        size = 400 if trial % 2 == 0 else 800
        probabilities = np.r_[np.full(15, 0.95), np.linspace(0.94, 0.4, 30)]
        counts = generator.binomial(size, probabilities, size=(1, len(probabilities)))
        available = generator.random(counts.shape) > 0.08
        expected = replay_counts(counts, available, size, policy)[0]
        frames = []
        for day, raw_count in enumerate(counts[0]):
            count = int(raw_count)
            start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)
            ready = bool(available[0, day])
            batch = TransportBatch(
                ready,
                "constructed held-out input",
                f"held-{trial}-{day}",
                "mock:held-out",
                size if ready else 0,
                size,
                values=(1,) * count + (0,) * (size - count) if ready else (),
                role_seed=trial * 1000 + day,
                request_design_hash=content_hash(["fixed", size]),
                capture_hash=content_hash([trial, day, count]),
                reference_tier="MOCK",
            )
            frames.append(
                TemporalFrame(
                    "held-out",
                    f"w-{day}",
                    "daily",
                    start,
                    start + timedelta(days=1),
                    "a" * 64 if day < 22 else "b" * 64,
                    batch,
                    measurement_source_hash="f" * 64,
                    producer_runtime_hash="d" * 64,
                )
            )
        actual = replay_candidates(frames, policy, expected_source_hash="f" * 64)
        events = [(day, step.event) for day, step in enumerate(actual) if step.event]
        if len(events) != len(expected):
            raise ValueError("transport_count_public_replay_disagreement")
        for (day, event), wanted in zip(events, expected, strict=True):
            assert event is not None
            if (
                day != wanted.end
                or event.input_window_ids != tuple(f"w-{i}" for i in range(wanted.start, day + 1))
                or any(
                    getattr(event, k) != getattr(wanted, k)
                    for k in (
                        "left_n",
                        "right_n",
                        "left_successes",
                        "right_successes",
                        "difference",
                        "ci_low",
                        "ci_high",
                    )
                )
            ):
                raise ValueError("transport_count_public_replay_disagreement")
            compared += 1
    full_paths = 0
    if replicates >= plan["replicates"]:
        for scenario in range(2):
            size, streams, updates = policy.batch_sizes["core"], policy.max_streams, 240
            probabilities = np.full((streams, updates), 0.5, dtype=np.float64)
            probabilities[0, :130] = 0.95
            probabilities[0, 130:150] = np.linspace(0.94, 0.75, 20)
            probabilities[0, 150:] = 0.75
            counts = generator.binomial(size, probabilities)
            available = np.ones(counts.shape, dtype=np.bool_)
            if scenario:
                available = generator.random(counts.shape) < 1 - (1 - probabilities) ** 3
            full_expected = replay_counts(counts, available, size, policy)
            frames = []
            for stream in range(streams):
                for update in range(updates):
                    count = int(counts[stream, update])
                    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=6 * update)
                    ready = bool(available[stream, update])
                    batch = TransportBatch(
                        ready,
                        "constructed full-family conformance",
                        f"full-{scenario}-{stream}-{update}",
                        "mock:held-out",
                        size if ready else 0,
                        size,
                        values=(1,) * count + (0,) * (size - count) if ready else (),
                        role_seed=scenario * 100000 + stream * 1000 + update,
                        request_design_hash=content_hash(["fixed", size]),
                        capture_hash=content_hash([scenario, stream, update, count]),
                        reference_tier="MOCK",
                    )
                    frames.append(
                        TemporalFrame(
                            f"stream-{stream:03}",
                            f"w-{update}",
                            "probe",
                            start,
                            start + timedelta(hours=6),
                            "a" * 64 if update < 120 else "b" * 64,
                            batch,
                            measurement_source_hash="f" * 64,
                            producer_runtime_hash="d" * 64,
                        )
                    )
            actual = replay_candidates(frames, policy, expected_source_hash="f" * 64)
            for stream in range(streams):
                full_events = [
                    r.event for r in actual if r.stream_id == f"stream-{stream:03}" and r.event
                ]
                if len(full_events) != len(full_expected[stream]):
                    raise ValueError("transport_full_family_replay_disagreement")
                for event, wanted in zip(full_events, full_expected[stream], strict=True):
                    if event.input_window_ids != tuple(
                        f"w-{i}" for i in range(wanted.start, wanted.end + 1)
                    ) or any(
                        getattr(event, k) != getattr(wanted, k)
                        for k in (
                            "left_n",
                            "right_n",
                            "left_successes",
                            "right_successes",
                            "difference",
                            "ci_low",
                            "ci_high",
                        )
                    ):
                        raise ValueError("transport_full_family_replay_disagreement")
                compared += len(full_events)
            full_paths += 1
    return {
        "trajectories": trajectories,
        "windows_per_trajectory": 45,
        "compared_events": compared,
        "full_family_paths": full_paths,
        "full_family_streams": policy.max_streams,
        "full_family_updates": 240,
        "both_sizes_missingness_resets_and_compatible_producer_bridge": True,
        "passed": True,
    }


def _truth_prefix(probabilities: npt.NDArray[np.float64]) -> list[Fraction]:
    prefix = [Fraction()]
    for p in probabilities:
        prefix.append(prefix[-1] + Fraction.from_float(float(p)))
    return prefix


def _covered(event: CountCut, prefix: list[Fraction]) -> bool:
    middle, end = event.start + event.left_batches, event.end + 1
    truth = (prefix[end] - prefix[middle]) / (end - middle) - (
        prefix[middle] - prefix[event.start]
    ) / event.left_batches
    return Fraction.from_float(event.ci_low) <= truth <= Fraction.from_float(event.ci_high)


def transport_study(out: Path, *, replicates: int | None = None) -> dict[str, Any]:
    plan = _plan()
    repetitions = plan["replicates"] if replicates is None else replicates
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive_integer_transport_replications_required")
    if any(p.lower() == "starter files" for p in out.parts):
        raise ValueError("fresh_owned_transport_study_output_required")
    if any(p.is_symlink() for p in [*reversed(out.absolute().parents), out]):
        raise ValueError("fresh_owned_transport_study_output_required")
    if out.exists():
        raise ValueError("fresh_transport_study_output_required")
    source = implementation_hash()
    producer = producer_identity()
    policy = load_rolling_policy()
    out.mkdir(parents=True)
    _json(out / "plan.json", plan)
    _json(out / "producer.json", producer)
    family_rows = []
    readiness_rows = []
    with offline_only():
        held_out = _held_out(plan, repetitions)
        _json(out / "held-out.json", held_out)
        for case in plan["family_cases"]:
            if (
                case["streams"] != policy.max_streams
                or case["size"] != policy.batch_sizes["core"]
                or case["evaluation_updates"] != policy.max_updates
                or case["priming_updates"] != policy.max_updates
                or case["cadence"] != "probe"
            ):
                raise ValueError("transport_family_shape_disagrees_with_policy")
            folder = out / case["id"]
            folder.mkdir()
            updates = case["priming_updates"] + case["evaluation_updates"]
            size, streams = case["size"], case["streams"]
            probabilities = np.full(updates, case["probability"], dtype=np.float64)
            q = 1 - (1 - probabilities) ** (plan["preflight_retries"] + 1)
            prefix = _truth_prefix(probabilities)
            event_counts, lengths, availability, failures = [], [], [], []
            for first in range(0, repetitions, plan["family_chunk_size"]):
                count = min(plan["family_chunk_size"], repetitions - first)
                values = np.empty((count * streams, updates), dtype=np.int64)
                ready = np.ones(values.shape, dtype=np.bool_)
                seeds = []
                for index in range(count):
                    seed = _seed(plan, "null-family", case["id"], first + index)
                    seeds.append(seed)
                    generator = np.random.default_rng(seed)
                    target = slice(index * streams, (index + 1) * streams)
                    values[target] = generator.binomial(
                        size, probabilities, size=(streams, updates)
                    )
                    if case["preflight_readiness"]:
                        ready[target] = generator.random((streams, updates)) < q
                events = replay_counts(values, ready, size, policy)
                np.savez_compressed(
                    folder / f"chunk-{first:06}.npz",
                    counts=values.astype(np.uint16),
                    available=ready,
                    probabilities=probabilities,
                    seeds=np.asarray(seeds, dtype=np.uint64),
                    first_family=first,
                )
                _json(
                    folder / f"events-{first:06}.json", [[asdict(e) for e in row] for row in events]
                )
                for index in range(count):
                    selected = [
                        e
                        for row in events[index * streams : (index + 1) * streams]
                        for e in row
                        if e.end >= case["priming_updates"]
                    ]
                    event_counts.append(float(len(selected)))
                    lengths.append(
                        float(
                            min(
                                (e.end - case["priming_updates"] + 1 for e in selected),
                                default=case["evaluation_updates"],
                            )
                        )
                    )
                    availability.append(
                        float(
                            np.mean(
                                ready[
                                    index * streams : (index + 1) * streams,
                                    case["priming_updates"] :,
                                ]
                            )
                        )
                    )
                    failures.append(any(not _covered(e, prefix) for e in selected))
            any_event = sum(n > 0 for n in event_counts)
            low, high = wilson(any_event, repetitions)
            failure_count = sum(failures)
            flo, fhi = wilson(failure_count, repetitions)
            family_rows.append(
                {
                    **case,
                    "families": repetitions,
                    "reference_tier": "MOCK",
                    "any_event_families": any_event,
                    "any_event_rate": any_event / repetitions,
                    "ci_low": low,
                    "ci_high": high,
                    "events_per_family": _bootstrap(event_counts, plan, case["id"] + "-events"),
                    "restricted_run_length_updates": _bootstrap(
                        lengths, plan, case["id"] + "-run-length"
                    ),
                    "right_censored_families": repetitions - any_event,
                    "available_evaluation_fraction": _bootstrap(
                        availability, plan, case["id"] + "-availability"
                    ),
                    "families_with_any_emitted_interval_failure": failure_count,
                    "interval_failure_ci_low": flo,
                    "interval_failure_ci_high": fhi,
                }
            )
        for case in plan["readiness_cases"]:
            folder = out / case["id"]
            folder.mkdir()
            updates, size, change = case["updates"], case["size"], case["change_index"]
            probabilities = np.full(updates, case["probability"], dtype=np.float64)
            probabilities -= case["drop"] * np.clip(
                (np.arange(updates) - change + 1) / case["ramp"], 0, 1
            )
            q = 1 - (1 - probabilities) ** (plan["preflight_retries"] + 1)
            prefix = _truth_prefix(probabilities)
            delays, availability, event_counts = [], [], []
            detected = prechange = interval_failures = 0
            for first in range(0, repetitions, plan["family_chunk_size"]):
                count = min(plan["family_chunk_size"], repetitions - first)
                values = np.empty((count, updates), dtype=np.int64)
                ready = np.empty(values.shape, dtype=np.bool_)
                seeds = []
                for index in range(count):
                    seed = _seed(plan, "readiness", case["id"], first + index)
                    seeds.append(seed)
                    generator = np.random.default_rng(seed)
                    values[index] = generator.binomial(size, probabilities)
                    ready[index] = generator.random(updates) < q
                events = replay_counts(values, ready, size, policy)
                np.savez_compressed(
                    folder / f"chunk-{first:06}.npz",
                    counts=values.astype(np.uint16),
                    available=ready,
                    probabilities=probabilities,
                    seeds=np.asarray(seeds, dtype=np.uint64),
                    first_trajectory=first,
                )
                _json(
                    folder / f"events-{first:06}.json", [[asdict(e) for e in row] for row in events]
                )
                for index, row in enumerate(events):
                    after = [e for e in row if e.end >= change]
                    if after:
                        detected += 1
                        delays.append(float(after[0].end - change))
                    prechange += any(e.end < change for e in row)
                    interval_failures += any(not _covered(e, prefix) for e in row)
                    availability.append(float(np.mean(ready[index])))
                    event_counts.append(float(len(row)))
            low, high = wilson(detected, repetitions)
            pre_low, pre_high = wilson(prechange, repetitions)
            failure_low, failure_high = wilson(interval_failures, repetitions)
            readiness_rows.append(
                {
                    **case,
                    "replicates": repetitions,
                    "reference_tier": "MOCK",
                    "detected": detected,
                    "power": detected / repetitions,
                    "ci_low": low,
                    "ci_high": high,
                    "right_censored": repetitions - detected,
                    "followup_updates": updates - change,
                    "prechange_alarm_trajectories": prechange,
                    "prechange_alarm_ci_low": pre_low,
                    "prechange_alarm_ci_high": pre_high,
                    "conditional_delay_updates": _bootstrap(delays, plan, case["id"] + "-delay"),
                    "available_fraction": _bootstrap(
                        availability, plan, case["id"] + "-availability"
                    ),
                    "events_per_trajectory": _bootstrap(event_counts, plan, case["id"] + "-events"),
                    "trajectories_with_any_emitted_interval_failure": interval_failures,
                    "interval_failure_ci_low": failure_low,
                    "interval_failure_ci_high": failure_high,
                }
            )
    if implementation_hash() != source:
        raise ValueError("transport_study_source_changed")
    allocation = min(
        Fraction(str(policy.family_alpha)), Fraction(str(policy.max_false_events_30_days))
    )
    allocation *= 1 - Fraction(2, policy.max_history_batches + 1)
    result = {
        "version": "monitor-operating-study-v1",
        "source_hash": source,
        "producer": producer,
        "policy": policy.model_dump(mode="json"),
        "policy_hash": content_hash(policy.model_dump(mode="json")),
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analytic_fixed_family_allocation_fraction": str(allocation),
        "analytic_fixed_family_allocation_upper_float": -floor_fraction(-allocation),
        "family_cases": family_rows,
        "readiness_cases": readiness_rows,
        "held_out_public_replay": held_out,
        "eligible_for_operating_review": bool(
            repetitions >= plan["replicates"]
            and held_out["trajectories"] >= plan["held_out_replays"]
            and held_out["full_family_paths"] == 2
            and all(row["ci_high"] <= policy.family_alpha for row in family_rows)
            and all(row["ci_low"] >= plan["min_power"] for row in readiness_rows)
            and producer["runtime"]["complete"]
            and producer["generator"]["complete"]
            and producer["loaded_source_hash"] == producer["source_hash"]
        ),
        "operating_qualification_accepted": False,
        "model_api_calls": 0,
        "new_spend_usd": 0,
        "scope": (
            "Constructed independent-binomial mock model. Fixed 30-day null families; "
            "readiness power followup can span longer. Not provider or runtime operating admission."
        ),
    }
    result["artifact_sha256"] = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    _json(out / "study.json", result)
    return result
