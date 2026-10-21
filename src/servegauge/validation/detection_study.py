"""Versioned, socket-free held-out studies. Original accepted artifacts stay intact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import hypergeom

from servegauge.core.hashing import content_hash
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.stats.change_variants import BatchADWIN, RestartingEvidence
from servegauge.stats.changepoint import CUSUM
from servegauge.stats.logprob_dispersion import logprob_dispersion_test
from servegauge.stats.multiple import holm
from servegauge.stats.paired_fast import jsd_test_fast, logprob_test_fast
from servegauge.validation.changepoint_sim import (
    alert_rate,
    calibrated_threshold,
    evaluate_trajectory,
    mean_interval,
    trajectory,
)
from servegauge.validation.common import (
    ROOT,
    cached,
    named_seed,
    offline_only,
    proportion,
    receipt,
    settings,
    write_json,
)
from servegauge.verdicts.config import load_policy

OUTPUT = Path("docs/report/detection-v2")


def configuration() -> dict[str, Any]:
    value = yaml.safe_load((ROOT / "configs/detection_validation.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("versioned_detection_configuration_required")
    return value


def noise_observations(
    n: int, *, seed: int, jitter: float, noise: float
) -> tuple[list[str], list[dict[str, float]]]:
    backend = ScriptedBackend(
        seed,
        FaultConfig(logit_noise=noise),
        sampling_version="v2",
        healthy_logit_jitter=jitter,
    )
    body = {
        "model": "mock",
        "messages": [
            {
                "role": "user",
                "content": 'TW_TASK: {"type":"sample","choices":["red","blue","green","gold"]}',
            }
        ],
        "temperature": 1.0,
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 4,
    }
    words, logs = [], []
    for _ in range(n):
        choice = backend.respond(body)["choices"][0]
        words.append(choice["message"]["content"])
        logs.append(
            {r["token"]: r["logprob"] for r in choice["logprobs"]["content"][0]["top_logprobs"]}
        )
    return words, logs


def noise_trial(
    n: int, *, seed: int, jitter: float, noise: float, permutations: int
) -> dict[str, Any]:
    a, x = noise_observations(n, seed=named_seed(seed, "first"), jitter=jitter, noise=noise)
    b, y = noise_observations(n, seed=named_seed(seed, "second"), jitter=jitter, noise=0)
    tests = {
        "FP1": jsd_test_fast(a, b, seed=seed, resamples=permutations),
        "FP3-mean": logprob_test_fast(
            x,
            y,
            missing_floor=float(load_policy().entries["FP3"]["missing_logprob_floor"]),
            seed=seed,
            resamples=permutations,
        ),
        "FP3-dispersion": logprob_dispersion_test(x, y, seed=seed, resamples=permutations),
    }
    adjusted = holm([r.p_value for r in tests.values()])
    return {
        "seed": seed,
        "observation_hash": content_hash([a, x, b, y]),
        "results": {
            name: {
                "statistic": r.statistic,
                "p_value": r.p_value,
                "adjusted_p": p,
                "permutations": r.permutations,
                "exact": r.exact,
            }
            for (name, r), p in zip(tests.items(), adjusted, strict=True)
        },
    }


def noise_study(*, out: Path = OUTPUT) -> dict[str, Any]:
    full, policy = configuration(), load_policy()
    cfg = full["noise"]
    original_fault = yaml.safe_load((ROOT / "configs/faults/qsim.yaml").read_text(encoding="utf-8"))
    assert cfg["fault_noise"] == FaultConfig.model_validate(original_fault).logit_noise
    output = receipt("noise-v2", full)
    before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    directory = ROOT / ".cache/detection-v2" / content_hash([output["source_hash"], full])
    rows, trials = [], []
    with offline_only():
        for jitter in cfg["healthy_jitter"]:
            for mode, sizes, count in (
                ("F-QSIM", cfg["query_sizes"], cfg["replicates"]),
                ("null", [max(cfg["query_sizes"])], cfg["null_replicates"]),
            ):
                for n in sizes:
                    batch = []
                    for i in range(count):
                        seed = named_seed(
                            full["seed"], full["version"], "noise", mode, jitter, n, i
                        )
                        path = directory / f"noise-{seed}.json"
                        result = cached(path)
                        if result is None:
                            result = noise_trial(
                                n,
                                seed=seed,
                                jitter=jitter,
                                noise=cfg["fault_noise"] if mode == "F-QSIM" else 0,
                                permutations=cfg["permutations"],
                            )
                            write_json(path, result)
                        batch.append(result)
                        if (i + 1) % 500 == 0:
                            print(
                                f"noise-v2 {mode} jitter={jitter} n={n}: {i + 1}/{count}",
                                flush=True,
                            )
                    for method in batch[0]["results"]:
                        rows.append(
                            {
                                "mode": mode,
                                "healthy_jitter": jitter,
                                "n_per_side": n,
                                "queries": 2 * n,
                                "method": method,
                                "power": proportion(
                                    sum(
                                        t["results"][method]["adjusted_p"] <= policy.alpha
                                        for t in batch
                                    ),
                                    count,
                                ),
                            }
                        )
                    if mode == "null":
                        rows.append(
                            {
                                "mode": mode,
                                "healthy_jitter": jitter,
                                "n_per_side": n,
                                "queries": 2 * n,
                                "method": "family",
                                "power": proportion(
                                    sum(
                                        any(
                                            r["adjusted_p"] <= policy.alpha
                                            for r in t["results"].values()
                                        )
                                        for t in batch
                                    ),
                                    count,
                                ),
                            }
                        )
                    trials.append(
                        {"mode": mode, "healthy_jitter": jitter, "n_per_side": n, "trials": batch}
                    )
                    print(f"noise-v2 {mode} jitter={jitter} n={n}: complete", flush=True)
    output.update(
        rows=rows,
        trials=trials,
        spend_hash_before=before,
        spend_hash_after=content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")),
    )
    output["accepted"] = (
        before == output["spend_hash_after"]
        and all(
            r["power"]["estimate"] <= cfg["maximum_null_rate"]
            for r in rows
            if r["mode"] == "null" and r["method"] == "family"
        )
        and all(
            r["power"]["ci_low"] >= cfg["minimum_power_lower"]
            for r in rows
            if r["mode"] == "F-QSIM"
            and r["n_per_side"] == max(cfg["query_sizes"])
            and r["method"] == "FP3-dispersion"
        )
    )
    write_json(out / "noise.json", output)
    return output


# All binary permutation probabilities are computed from counts, not fault labels.
_P = np.asarray(
    hypergeom.cdf(np.arange(33)[:, None], 96, np.arange(33)[:, None] + np.arange(65), 32),
    dtype=np.float64,
)


def variant_events(
    data: dict[str, Any], cfg: dict[str, Any], defaults: dict[str, Any], *, peaks: bool = False
) -> dict[str, Any]:
    baseline = list(map(float, data["baseline"]))
    binary = list(map(int, data["binary_baseline"]))
    recent_binary = binary.copy()
    a = CUSUM(
        baseline_mean=float(np.mean(baseline)),
        baseline_sd=float(np.std(baseline, ddof=1)),
        slack=cfg["cusum_slack"],
        threshold=defaults["D-A"]["threshold"],
    )
    b = RestartingEvidence(kappa=cfg["evalue_kappa"], log_threshold=defaults["D-B"]["threshold"])
    c = BatchADWIN(delta=defaults["D-C"]["delta"])
    c.update(binary)
    events: dict[str, Any] = {"D-A": [], "D-B": [], "D-C": [], "peaks": [0.0, 0.0]}
    for i, value in enumerate(data["values"]):
        baseline.append(float(value))
        alarm_a = a.update(float(value))
        current = list(map(int, data["batches"][i]))
        recent_binary.extend(current)
        # Exact lookup only for the declared 32/64 design; generic path remains valid.
        p = (
            float(_P[sum(current), sum(binary)])
            if len(current) == 32 and len(binary) == 64
            else float(
                hypergeom.cdf(
                    sum(current),
                    len(current) + len(binary),
                    sum(current) + sum(binary),
                    len(current),
                )
            )
        )
        alarm_b = b.update(p)
        if c.update(current):
            events["D-C"].append(i)
        events["peaks"][0] = max(events["peaks"][0], a.positive, a.negative)
        events["peaks"][1] = max(events["peaks"][1], b.log_evidence)
        if alarm_a and not peaks:
            events["D-A"].append(i)
            latest = baseline[-cfg["baseline_size"] :]
            a = CUSUM(
                baseline_mean=float(np.mean(latest)),
                baseline_sd=max(float(np.std(latest, ddof=1)), float(np.finfo(float).eps)),
                slack=cfg["cusum_slack"],
                threshold=defaults["D-A"]["threshold"],
            )
        if alarm_b and not peaks:
            events["D-B"].append(i)
            binary = recent_binary[-cfg["baseline_size"] :]
            b.reset()
    return events


def delay_values(
    traces: list[dict[str, Any]], method: str, cfg: dict[str, Any]
) -> tuple[list[float], list[float]]:
    start = cfg["shift_day"] * cfg["observations_per_day"]
    horizon = cfg["days"] * cfg["observations_per_day"]
    first = [next((t for t in r["events"][method] if t >= start), horizon) for r in traces]
    restricted = [(t - start) / cfg["observations_per_day"] for t in first]
    return [delay for t, delay in zip(first, restricted, strict=True) if t < horizon], restricted


def change_study(*, out: Path = OUTPUT) -> dict[str, Any]:
    full, original = configuration(), settings()["changepoint"]
    cfg = {**original, **full["change"]}
    output = receipt("changepoint-v2", {"design": full, "trajectory": cfg})
    before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    old_defaults = yaml.safe_load(
        (ROOT / "configs/changepoint_defaults.yaml").read_text(encoding="utf-8")
    )["parameters"]
    directory = ROOT / ".cache/detection-v2" / content_hash([output["source_hash"], full, cfg])
    unbounded = {
        "D-A": {"threshold": float(np.finfo(float).max)},
        "D-B": {"threshold": float(np.finfo(float).max)},
        "D-C": {"delta": old_defaults["D-C"]["delta"]},
    }
    calibration = []
    evaluation: dict[str, Any] = {}
    with offline_only():
        for i in range(cfg["calibration_replicates"]):
            seed = named_seed(full["seed"], full["version"], "change", "calibration", i)
            path = directory / f"calibration-{seed}.json"
            result = cached(path)
            if result is None:
                result = {
                    "seed": seed,
                    **variant_events(trajectory(seed, "null", cfg), cfg, unbounded, peaks=True),
                }
                write_json(path, result)
            calibration.append(result)
            if (i + 1) % 500 == 0:
                print(
                    f"change-v2 null calibration {i + 1}/{cfg['calibration_replicates']}",
                    flush=True,
                )
        defaults = {
            "D-A": calibrated_threshold([r["peaks"][0] for r in calibration], cfg),
            "D-B": calibrated_threshold([r["peaks"][1] for r in calibration], cfg),
            "D-C": {
                "delta": unbounded["D-C"]["delta"],
                "calibration_tail": proportion(
                    sum(bool(r["D-C"]) for r in calibration), len(calibration)
                ),
                "union_bound": cfg["days"]
                * cfg["observations_per_day"]
                * unbounded["D-C"]["delta"],
            },
        }
        budget = defaults["D-A"]["per_stream_method_probability_budget"]
        if (
            defaults["D-C"]["calibration_tail"]["ci_high"] > budget
            or defaults["D-C"]["union_bound"] > budget
        ):
            raise ValueError("batch_adwin_calibration_failed_preserve_holdout")
        output["calibration"] = calibration
        output["defaults"] = defaults
        # The thresholds are serialized before any final evaluation is generated.
        write_json(directory / "frozen-thresholds.json", defaults)
        for mode in ("null", "abrupt", "gradual"):
            traces = []
            for i in range(cfg["evaluation_replicates"]):
                seed = named_seed(full["seed"], full["version"], "change", "evaluation", mode, i)
                path = directory / f"evaluation-{seed}.json"
                result = cached(path)
                if result is None:
                    data = trajectory(seed, mode, cfg)
                    new = variant_events(data, cfg, defaults)
                    result = {
                        "seed": seed,
                        "v2": {k: v for k, v in new.items() if k != "peaks"},
                        "v1": evaluate_trajectory(data, cfg, old_defaults),
                    }
                    write_json(path, result)
                traces.append(result)
                if (i + 1) % 500 == 0:
                    print(
                        f"change-v2 independent {mode} {i + 1}/{cfg['evaluation_replicates']}",
                        flush=True,
                    )
            evaluation[mode] = traces
    output["evaluation"] = evaluation
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for version in ("v1", "v2"):
        for mode, traces in evaluation.items():
            view = [{"events": r[version]} for r in traces]
            for method in defaults:
                conditional, restricted = delay_values(view, method, cfg)
                times = [r["events"][method] for r in view]
                start = cfg["shift_day"] * cfg["observations_per_day"]
                horizon = cfg["days"] * cfg["observations_per_day"]
                rows.append(
                    {
                        "version": version,
                        "mode": mode,
                        "detector": method,
                        "detection_probability": proportion(len(conditional), len(traces))
                        if mode != "null"
                        else None,
                        "conditional_delay_days": mean_interval(
                            conditional, named_seed(version, mode, method, "conditional")
                        )
                        if mode != "null"
                        else None,
                        "restricted_delay_days": mean_interval(
                            restricted, named_seed(version, mode, method, "restricted")
                        )
                        if mode != "null"
                        else None,
                        "restricted_run_length_days": mean_interval(
                            [
                                min(horizon, next(iter(t), horizon) + 1)
                                / cfg["observations_per_day"]
                                for t in times
                            ],
                            named_seed(version, mode, method, "runlength"),
                        ),
                        "prechange_alarm_probability": proportion(
                            sum(any(t < start for t in ts) for ts in times), len(traces)
                        )
                        if mode != "null"
                        else None,
                        "censored": len(traces) - len(conditional)
                        if mode != "null"
                        else sum(not t for t in times),
                        "false_alarm_rate": alert_rate(
                            [len(t) for t in times], cfg, named_seed(version, method, "rate")
                        )
                        if mode == "null"
                        else None,
                    }
                )
    for mode in ("abrupt", "gradual"):
        traces = evaluation[mode]
        for method in ("D-B", "D-C"):
            _, old = delay_values([{"events": r["v1"]} for r in traces], method, cfg)
            _, new_delay = delay_values([{"events": r["v2"]} for r in traces], method, cfg)
            comparisons.append(
                {
                    "mode": mode,
                    "detector": method,
                    "paired_v2_minus_v1_days": mean_interval(
                        [b - a for a, b in zip(old, new_delay, strict=True)],
                        named_seed(mode, method, "paired-delay"),
                    ),
                }
            )
    output["rows"], output["comparisons"] = rows, comparisons
    output["whole_observatory_null"] = alert_rate(
        [sum(len(t) for t in r["v2"].values()) for r in evaluation["null"]],
        cfg,
        named_seed(full["seed"], "v2-observatory-null"),
    )
    output["spend_hash_before"] = before
    output["spend_hash_after"] = content_hash(
        (ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")
    )
    output["accepted"] = (
        before == output["spend_hash_after"]
        and output["whole_observatory_null"]["poisson_working_interval"][1]
        <= cfg["maximum_false_alarms_per_30_days"]
        and all(
            r["detection_probability"]["estimate"] >= cfg["minimum_abrupt_power"]
            for r in rows
            if r["version"] == "v2" and r["mode"] == "abrupt" and r["detector"] in {"D-B", "D-C"}
        )
        and all(r["paired_v2_minus_v1_days"]["ci_high"] < 0 for r in comparisons)
    )
    write_json(out / "changepoint.json", output)
    write_json(
        out / "changepoint-defaults.json",
        {
            "version": full["version"],
            "source_hash": output["source_hash"],
            "configuration": cfg,
            "parameters": defaults,
            "qualification": "Offline simulation only; inactive, recalibrate before monitoring",
        },
    )
    return output
