"""Finite-horizon null calibration and independent change-detection evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import chi2, hypergeom

from servegauge.core.hashing import content_hash
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.changepoint import ADWIN, CUSUM, EValueAccumulator
from servegauge.stats.intervals import wilson
from servegauge.validation.common import (
    ROOT,
    named_seed,
    offline_only,
    proportion,
    receipt,
    settings,
    write_json,
)


def permutation_p(current: int, baseline: int, n_current: int, n_baseline: int) -> float:
    """Exact one-sided pooled-label permutation p-value for two binary batches."""
    return float(hypergeom.cdf(current, n_current + n_baseline, current + baseline, n_current))


def trajectory(seed: int, mode: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if mode not in {"null", "abrupt", "gradual"}:
        raise ValueError("known_shift_mode_required")
    rng = np.random.default_rng(seed)
    steps = cfg["days"] * cfg["observations_per_day"]
    t = np.arange(steps)
    start = cfg["shift_day"] * cfg["observations_per_day"]
    effect = (
        np.zeros(steps)
        if mode == "null"
        else (t >= start).astype(float)
        if mode == "abrupt"
        else np.clip((t - start + 1) / (cfg["gradual_days"] * cfg["observations_per_day"]), 0, 1)
    )
    probability = cfg["binary_baseline"] + (cfg["binary_shift"] - cfg["binary_baseline"]) * effect
    return {
        "baseline": rng.normal(size=cfg["baseline_size"]),
        "values": rng.normal(size=steps) + effect * cfg["standardized_shift"],
        "binary_baseline": rng.binomial(1, cfg["binary_baseline"], size=cfg["baseline_size"]),
        "batches": rng.binomial(1, probability[:, None], size=(steps, cfg["binary_batch"])),
    }


def null_peaks(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float]:
    baseline = data["baseline"]
    cusum = CUSUM(
        baseline_mean=float(np.mean(baseline)),
        baseline_sd=float(np.std(baseline, ddof=1)),
        slack=cfg["cusum_slack"],
        threshold=float(np.finfo(float).max),
    )
    evidence = EValueAccumulator(
        kappa=cfg["evalue_kappa"], log_threshold=float(np.finfo(float).max)
    )
    counts = np.sum(data["batches"], axis=1)
    p = hypergeom.cdf(
        counts,
        cfg["binary_batch"] + cfg["baseline_size"],
        counts + int(np.sum(data["binary_baseline"])),
        cfg["binary_batch"],
    )
    a = b = 0.0
    for value, pvalue in zip(data["values"], p, strict=True):
        cusum.update(float(value))
        evidence.update(float(pvalue))
        a = max(a, cusum.positive, cusum.negative)
        b = max(b, evidence.log_evidence)
    return a, b


def adwin_events(data: dict[str, Any], delta: float) -> list[int]:
    detector = ADWIN(delta=delta)
    for bit in data["binary_baseline"]:
        detector.update(int(bit))
    return [i for i, bit in enumerate(data["batches"][:, 0]) if detector.update(int(bit))]


def calibrated_threshold(peaks: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    # Split one monthly observatory alert across all streams and all three methods;
    # reserve half the budget before independent evaluation. Never tune on that set.
    budget = (
        cfg["maximum_false_alarms_per_30_days"]
        * cfg["calibration_budget_fraction"]
        * (cfg["days"] / 30)
        / (cfg["streams"] * 3)
    )
    allowed = [k for k in range(len(peaks)) if wilson(k, len(peaks))[1] <= budget]
    if not allowed:
        raise ValueError("more_null_calibration_replicates_required_for_observatory_target")
    k = max(allowed)
    threshold = float(np.nextafter(sorted(peaks)[len(peaks) - k - 1], np.inf))
    return {
        "threshold": max(threshold, float(np.finfo(float).eps)),
        "calibration_tail": proportion(sum(p > threshold for p in peaks), len(peaks)),
        "per_stream_method_probability_budget": budget,
        "method": "largest calibration tail with Wilson upper bound inside declared budget",
    }


def evaluate_trajectory(
    data: dict[str, Any], cfg: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, list[int]]:
    baseline = list(map(float, data["baseline"]))
    cusum = CUSUM(
        baseline_mean=float(np.mean(baseline)),
        baseline_sd=float(np.std(baseline, ddof=1)),
        slack=cfg["cusum_slack"],
        threshold=defaults["D-A"]["threshold"],
    )
    evidence = EValueAccumulator(
        kappa=cfg["evalue_kappa"], log_threshold=defaults["D-B"]["threshold"]
    )
    binary = list(map(int, data["binary_baseline"]))
    recent_binary = binary.copy()
    events: dict[str, list[int]] = {
        "D-A": [],
        "D-B": [],
        "D-C": adwin_events(data, defaults["D-C"]["delta"]),
    }
    for index, value in enumerate(data["values"]):
        value = float(value)
        baseline.append(value)
        if cusum.update(value):
            events["D-A"].append(index)
            recent = baseline[-cfg["baseline_size"] :]
            cusum = CUSUM(
                baseline_mean=float(np.mean(recent)),
                baseline_sd=max(float(np.std(recent, ddof=1)), float(np.finfo(float).eps)),
                slack=cfg["cusum_slack"],
                threshold=defaults["D-A"]["threshold"],
            )
        current = list(map(int, data["batches"][index]))
        recent_binary.extend(current)
        p = permutation_p(
            sum(current), sum(binary[-cfg["baseline_size"] :]), len(current), cfg["baseline_size"]
        )
        # Keep one fixed baseline until an alarm; then the latest window becomes it.
        if evidence.update(p):
            events["D-B"].append(index)
            binary = recent_binary[-cfg["baseline_size"] :]
            evidence.reset()
    return events


def mean_interval(values: list[float], seed: int) -> dict[str, Any] | None:
    if not values:
        return None
    interval = clustered_bootstrap(values, seed=seed, resamples=2000)
    return {
        "n": len(values),
        "estimate": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "resamples": 2000,
        "reference_tier": "MOCK",
    }


def alert_rate(counts: list[int], cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    total = sum(counts)
    n = len(counts)
    # Working Poisson exposure bounds supplement the item bootstrap at zero events;
    # they are labelled rather than claimed as distribution-free repeated-test control.
    low = 0.0 if not total else float(chi2.ppf(0.025, 2 * total) / 2 / n)
    high = float(chi2.ppf(0.975, 2 * (total + 1)) / 2 / n)
    estimate = mean_interval(list(map(float, counts)), seed)
    assert estimate is not None
    scale = cfg["streams"] * 30 / cfg["days"]
    return {
        "n_stream_windows": n,
        "alerts": total,
        "expected_observatory_alerts_per_30_days": estimate["estimate"] * scale,
        "bootstrap_interval": [estimate["ci_low"] * scale, estimate["ci_high"] * scale],
        "poisson_working_interval": [low * scale, high * scale],
        "any_alarm_probability": proportion(sum(c > 0 for c in counts), n),
        "interval_assumption": "Poisson working interval plus stream bootstrap; "
        "not a distribution-free guarantee",
    }


def changepoint_study(
    *, replicates: int | None = None, out: Path = Path("docs/report")
) -> dict[str, Any]:
    full = settings()
    cfg = full["changepoint"]
    evaluation = cfg["evaluation_replicates"] if replicates is None else replicates
    if evaluation < 60:
        raise ValueError("at_least_60_independent_change_evaluations_required")
    output = receipt(
        "changepoint", {**cfg, "seed": full["seed"], "executed_evaluation_replicates": evaluation}
    )
    before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    peaks_a: list[float] = []
    peaks_b: list[float] = []
    adwin_counts: dict[float, list[int]] = {d: [] for d in cfg["adwin_deltas"]}
    with offline_only():
        for i in range(cfg["calibration_replicates"]):
            data = trajectory(named_seed(full["seed"], "change", "calibration", i), "null", cfg)
            a, b = null_peaks(data, cfg)
            peaks_a.append(a)
            peaks_b.append(b)
            largest = adwin_events(data, max(adwin_counts))
            for delta in adwin_counts:
                adwin_counts[delta].append(
                    len(largest)
                    if delta == max(adwin_counts)
                    else len(adwin_events(data, delta))
                    if largest
                    else 0
                )
            if (i + 1) % 500 == 0:
                print(
                    f"changepoint null calibration {i + 1}/{cfg['calibration_replicates']}",
                    flush=True,
                )
        defaults = {
            "D-A": calibrated_threshold(peaks_a, cfg),
            "D-B": calibrated_threshold(peaks_b, cfg),
        }
        budget = (
            cfg["maximum_false_alarms_per_30_days"]
            * cfg["calibration_budget_fraction"]
            * (cfg["days"] / 30)
            / (cfg["streams"] * 3)
        )
        candidates: list[dict[str, Any]] = []
        for delta, counts in adwin_counts.items():
            observed = proportion(sum(c > 0 for c in counts), len(counts))
            bound = cfg["days"] * cfg["observations_per_day"] * delta
            candidates.append(
                {"delta": delta, "any_alarm": observed, "per_update_union_bound": bound}
            )
        eligible = [
            r
            for r in candidates
            if r["any_alarm"]["ci_high"] <= budget and r["per_update_union_bound"] <= budget
        ]
        if not eligible:
            raise ValueError("ADWIN_grid_does_not_meet_declared_observatory_budget")
        defaults["D-C"] = {
            "delta": max(r["delta"] for r in eligible),
            "calibration_candidates": candidates,
            "budget": budget,
        }
        output["defaults"] = defaults
        output["null_calibration"] = {
            "CUSUM_peaks": peaks_a,
            "shared_baseline_log_e_peaks": peaks_b,
        }
        output["evaluation"] = {}
        output["rows"] = []
        for mode in ("null", "abrupt", "gradual"):
            traces: list[dict[str, Any]] = []
            for i in range(evaluation):
                seed = named_seed(full["seed"], "change", "evaluation", mode, i)
                traces.append(
                    {
                        "seed": seed,
                        "events": evaluate_trajectory(trajectory(seed, mode, cfg), cfg, defaults),
                    }
                )
            output["evaluation"][mode] = traces
            start = cfg["shift_day"] * cfg["observations_per_day"]
            horizon = cfg["days"] * cfg["observations_per_day"]
            for method in defaults:
                lists = [r["events"][method] for r in traces]
                delays = (
                    [
                        (next(t for t in times if t >= start) - start) / cfg["observations_per_day"]
                        for times in lists
                        if any(t >= start for t in times)
                    ]
                    if mode != "null"
                    else []
                )
                restricted = (
                    [
                        (
                            (next((t for t in times if t >= start), horizon) - start)
                            / cfg["observations_per_day"]
                        )
                        for times in lists
                    ]
                    if mode != "null"
                    else []
                )
                output["rows"].append(
                    {
                        "mode": mode,
                        "detector": method,
                        "false_alarm_rate": alert_rate(
                            [len(times) for times in lists],
                            cfg,
                            named_seed(full["seed"], mode, method, "rate"),
                        )
                        if mode == "null"
                        else None,
                        "detection_probability": proportion(len(delays), evaluation)
                        if mode != "null"
                        else None,
                        "conditional_delay_days": mean_interval(
                            delays, named_seed(full["seed"], mode, method, "delay")
                        ),
                        "restricted_delay_days": mean_interval(
                            restricted, named_seed(full["seed"], mode, method, "restricted")
                        ),
                        "restricted_run_length_days": mean_interval(
                            [
                                min(horizon, next(iter(times), horizon) + 1)
                                / cfg["observations_per_day"]
                                for times in lists
                            ],
                            named_seed(full["seed"], mode, method, "runlength"),
                        ),
                        "censored": evaluation - len(delays)
                        if mode != "null"
                        else sum(not times for times in lists),
                    }
                )
            print(f"changepoint independent {mode}: {evaluation} streams", flush=True)
        null = output["evaluation"]["null"]
        output["whole_observatory_null"] = alert_rate(
            [sum(len(r["events"][m]) for m in defaults) for r in null],
            cfg,
            named_seed(full["seed"], "whole-observatory"),
        )
    output["scope_notes"] = [
        "100 endpoint-signal streams by default, all three detectors included in the alert budget",
        "Gaussian baseline mean/sd are estimated per replicate, including estimation error",
        "D-B uses exact pooled-label permutation p-values sharing one baseline; "
        "no martingale or anytime-validity claim",
        "After CUSUM/D-B alarms, the latest window replaces the baseline; "
        "ADWIN uses its own adaptive-window update",
        "D-C sees one binary observation per fingerprint; D-B sees a batch. "
        "Delays are not comparisons at equal request cost",
        "Null/shift streams are explicit simulation models, not provider time series",
        "Run lengths and undetected delays are censored at the declared horizon; "
        "restricted means are not infinite-horizon ARL",
        "Evaluation and calibration seeds are disjoint; thresholds stay fixed during evaluation",
    ]
    output["spend_hash_before"] = before
    output["spend_hash_after"] = content_hash(
        (ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")
    )
    assert output["spend_hash_after"] == before
    output["accepted"] = (
        output["whole_observatory_null"]["expected_observatory_alerts_per_30_days"]
        <= cfg["maximum_false_alarms_per_30_days"]
        and output["whole_observatory_null"]["poisson_working_interval"][1]
        <= cfg["maximum_false_alarms_per_30_days"]
    )
    write_json(out / "changepoint.json", output)
    defaults_path = ROOT / "configs/changepoint_defaults.yaml"
    defaults_path.write_text(
        yaml.safe_dump(
            {
                "version": "calibrated-offline-v1",
                "scope": output["configuration"],
                "source_hash": output["source_hash"],
                "parameters": defaults,
                "qualification": "simulation defaults; recalibrate after cadence, stream, "
                "baseline or statistic changes",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_change_report(output, out)
    return output


def write_change_report(output: dict[str, Any], out: Path) -> None:
    from servegauge.validation.reporting import rate

    lines = [
        "# Offline change-detection calibration",
        "",
        "Actual model API calls: 0. Actual spending: $0. Reference tier: MOCK.",
        "",
        *["- " + s for s in output["scope_notes"]],
        "",
        "| Detector | Mode | Detection probability, 95% Wilson | Conditional delay (days) | "
        "Restricted delay (days) | Censored |",
        "|---|---|---|---|---|---|",
    ]
    for row in output["rows"]:

        def display(value: Any) -> str:
            return (
                "unmeasured"
                if value is None
                else f"{value['estimate']:.3f} [{value['ci_low']:.3f}, "
                f"{value['ci_high']:.3f}], n={value['n']}"
            )

        lines.append(
            f"| {row['detector']} | {row['mode']} | "
            f"{rate(row['detection_probability']) if row['detection_probability'] else 'null'} | "
            f"{display(row['conditional_delay_days'])} | "
            f"{display(row['restricted_delay_days'])} | {row['censored']} |"
        )
    null = output["whole_observatory_null"]
    lines.extend(
        [
            "",
            "All three detectors, scaled to the declared observatory: "
            f"{null['expected_observatory_alerts_per_30_days']:.4f} "
            "observed mean false alarms / 30 days; working Poisson 95% interval "
            f"{null['poisson_working_interval']}. Whole-stream bootstrap interval: "
            f"{null['bootstrap_interval']}. These intervals have different assumptions; "
            "neither certifies an unobserved production stream.",
            "",
            "Defaults were selected from null calibration before independent evaluation. "
            "Low detection probability or a fully censored method remains visible; "
            "no threshold is relaxed to hide it.",
            "",
            "[Full simulation, seeds and defaults](changepoint.json). Configuration: "
            "configs/changepoint_defaults.yaml. Observatory scheduling remains disabled.",
            "",
        ]
    )
    (out / "changepoint.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "changepoint.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "detector",
                "mode",
                "detections",
                "n",
                "ci_low",
                "ci_high",
                "restricted_delay_days",
                "censored",
            ]
        )
        for row in output["rows"]:
            p = row["detection_probability"]
            writer.writerow(
                [
                    row["detector"],
                    row["mode"],
                    p["successes"] if p else None,
                    p["n"] if p else output["configuration"]["executed_evaluation_replicates"],
                    p["ci_low"] if p else None,
                    p["ci_high"] if p else None,
                    row["restricted_delay_days"]["estimate"]
                    if row["restricted_delay_days"]
                    else None,
                    row["censored"],
                ]
            )
