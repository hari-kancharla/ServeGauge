"""Independent finite-n text truths and predeclared fingerprint qualification studies.

These categorical component experiments test the actual kernel/statistics, not a
universal property of language models. Captured-wire acceptance is a separate phase.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq
from scipy.stats import binom

from servegauge.core.hashing import content_hash
from servegauge.stats.common import FloatArray
from servegauge.stats.fingerprint_strata import Cell, bootstrap_cells, randomization_test
from servegauge.stats.intervals import wilson
from servegauge.validation.common import (
    ROOT,
    named_seed,
    offline_only,
    proportion,
    receipt,
    write_json,
)
from servegauge.verdicts.config import load_policy


def _probabilities(first: FloatArray, second: FloatArray) -> tuple[FloatArray, FloatArray]:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or len(a) < 2
        or any(
            not np.all(np.isfinite(p))
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1, rtol=0, atol=float(100 * np.finfo(float).eps))
            for p in (a, b)
        )
    ):
        raise ValueError("matched_finite_probability_vectors_required")
    return a, b


def kernel_features(words: list[str], *, length: int) -> FloatArray:
    """Compact exact geometry of the same normalized Unicode-position kernel."""
    if length < 1 or len(set(words)) != len(words) or not words or any(not w for w in words):
        raise ValueError("distinct_nonempty_words_and_positive_length_required")
    padded = [[ord(w[i]) if i < len(w) else -1 for i in range(length)] for w in words]
    matrix = np.asarray(padded)
    gram = np.mean(matrix[:, None, :] == matrix[None, :, :], axis=-1)
    values, vectors = np.linalg.eigh(gram)
    if np.min(values) < -100 * np.finfo(float).eps * len(words):
        raise ValueError("positive_semidefinite_hamming_kernel_required")
    return np.asarray(vectors * np.sqrt(np.maximum(values, 0)), dtype=np.float64)


def _expected_jsd(first: FloatArray, second: FloatArray, n: int) -> float:
    # JSD is a sum over categories. Each pair of category counts has independent
    # binomial marginals, so no multinomial count-vector pair enumeration is needed.
    grid = np.arange(n + 1, dtype=np.float64) / n
    a, b = grid[:, None], grid[None, :]
    middle = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        component = (
            np.where(a > 0, a * np.log2(a / middle), 0)
            + np.where(b > 0, b * np.log2(b / middle), 0)
        ) / 2
    counts = np.arange(n + 1)
    return float(
        sum(
            binom.pmf(counts, n, p) @ component @ binom.pmf(counts, n, q)
            for p, q in zip(first, second, strict=True)
        )
    )


def finite_truth(method: str, reference: FloatArray, target: FloatArray, groups: int) -> float:
    """Expected reference-centered statistic for a known iid categorical process."""
    p, q = _probabilities(reference, target)
    if type(groups) is not int or not 2 <= groups <= 1000:
        raise ValueError("bounded_independent_component_count_required")
    if method == "FP1":
        return _expected_jsd(q, p, groups) - _expected_jsd(p, p, groups)
    if method == "FP2" and len(p) == 4:
        features = kernel_features(["red", "blue", "green", "gold"], length=32)
        difference = (q - p) @ features
        # The within-sample statistic is unbiased, including at small n.
        return float(difference @ difference)
    raise ValueError("known_text_fingerprint_metric_required")


def categorical_cells(
    method: str,
    reference: FloatArray,
    target: FloatArray,
    *,
    cells: int,
    groups: int,
    seed: int,
) -> list[Cell]:
    p, q = _probabilities(reference, target)
    if cells < 1 or groups < 2:
        raise ValueError("positive_strata_and_independent_components_required")
    if method == "FP1":
        features = np.eye(len(p))
    elif method == "FP2" and len(p) == 4:
        features = kernel_features(["red", "blue", "green", "gold"], length=32)
    else:
        raise ValueError("known_text_fingerprint_metric_required")
    rng = np.random.default_rng(seed)
    return [
        Cell(
            method, *(features[rng.choice(len(p), size=groups, p=probs)] for probs in (q, p, p, p))
        )
        for _ in range(cells)
    ]


def orientations(cells: list[Cell]) -> list[list[Cell]]:
    result = []
    for index in range(3):
        rows = []
        for cell in cells:
            peers = (cell.b, cell.c, cell.d)
            rows.append(Cell(cell.method, cell.a, *(peers[(index + j) % 3] for j in range(3))))
        result.append(rows)
    return result


def interval_trial(
    cells: list[Cell],
    *,
    truth: float,
    seed: int,
    resamples: int,
    alpha: float,
    margin: float,
) -> dict[str, Any]:
    intervals, deltas = [], []
    for i, rows in enumerate(orientations(cells)):
        draws = bootstrap_cells(rows, seed=seed + i, resamples=resamples, batch_size=resamples)
        intervals.append(np.quantile(draws[:, 2], [alpha / 6, 1 - alpha / 6]))
        deltas.append(float(np.mean([c.distance(c.a, c.b) - c.distance(c.c, c.d) for c in rows])))
    low = float(min(row[0] for row in intervals))
    high = float(max(row[1] for row in intervals))
    return {
        "n": sum(len(c.a) for c in cells),
        "reference_tier": "MOCK",
        "estimate": float(np.median(deltas)),
        "ci_low": low,
        "ci_high": high,
        "truth": truth,
        "covered": low <= truth <= high,
        "inside_margin": low > -margin and high < margin,
        "margin": margin,
        "seed": seed,
        "bootstrap_resamples": resamples,
    }


def null_p_value(cells: list[Cell], *, seed: int, permutations: int) -> float:
    return max(
        randomization_test(rows, seed=seed + i, permutations=permutations).p_value
        for i, rows in enumerate(orientations(cells))
    )


def configuration() -> dict[str, Any]:
    cfg = yaml.safe_load(
        (ROOT / "configs/fingerprint_product_validation.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(cfg, dict) or cfg.get("version") != "fingerprint-product-v1":
        raise ValueError("versioned_fingerprint_product_configuration_required")
    for key, minimum in (
        ("calibration_replicates", 200),
        ("coverage_replicates", 400),
        ("null_replicates", 2000),
        ("clean_replicates", 60),
        ("fault_replicates", 30),
        ("workers", 1),
    ):
        if type(cfg.get(key)) is not int or cfg[key] < minimum:
            raise ValueError("predeclared_fingerprint_replicates_required")
    return cfg


def interval_cases(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    policy = load_policy()
    result = []
    alternative = np.array([0.01, 0.04, 0.10, 0.85])
    for method in ("FP1", "FP2"):
        groups = cfg["design"][method.lower() + "_samples"]
        cells = cfg["design"][method.lower() + "_cells"]
        margin = policy.margin(method)
        for index, probabilities in enumerate(cfg["probability_grid"]):
            p = np.asarray(probabilities, dtype=np.float64)
            cases = [("null", 0.0, False)]
            for factor in cfg["boundary_margin_factors"]:
                wanted = factor * margin
                if finite_truth(method, p, alternative, groups) <= wanted:
                    raise ValueError("predeclared_boundary_unreachable_with_witness_alternative")
                weight = brentq(
                    _truth_gap,
                    0,
                    1,
                    args=(method, p, alternative, groups, wanted),
                    xtol=1e-14,
                )
                cases.append((f"boundary-{factor}", weight, factor >= 1))
            for label, weight, outside in cases:
                q = (1 - weight) * p + weight * alternative
                result.append(
                    {
                        "id": f"{method}-p{index}-{label}",
                        "method": method,
                        "reference": p.tolist(),
                        "target": q.tolist(),
                        "groups": groups,
                        "cells": cells,
                        "truth": finite_truth(method, p, q, groups),
                        "margin": margin,
                        "outside_equivalence": outside,
                        "null": label == "null",
                    }
                )
    return result


def _truth_gap(
    weight: float, method: str, p: FloatArray, alternative: FloatArray, groups: int, wanted: float
) -> float:
    return finite_truth(method, p, (1 - weight) * p + weight * alternative, groups) - wanted


def _trial(job: tuple[dict[str, Any], int, float, int]) -> dict[str, Any]:
    case, seed, alpha, resamples = job
    cells = categorical_cells(
        case["method"],
        np.asarray(case["reference"]),
        np.asarray(case["target"]),
        cells=case["cells"],
        groups=case["groups"],
        seed=named_seed(seed, "components"),
    )
    return interval_trial(
        cells,
        truth=case["truth"],
        seed=named_seed(seed, "bootstrap"),
        resamples=resamples,
        alpha=alpha,
        margin=case["margin"],
    )


def _required_scale(row: dict[str, Any]) -> float:
    center, truth = row["estimate"], row["truth"]
    radius = center - row["ci_low"] if truth < center else row["ci_high"] - center
    if radius <= 0:
        raise ValueError("nondegenerate_interval_radius_required_for_qualification")
    return float(max(1.0, abs(truth - center) / radius))


def _widen(row: dict[str, Any], scale: float) -> dict[str, Any]:
    center = row["estimate"]
    low, high = (
        center - scale * (center - row["ci_low"]),
        center + scale * (row["ci_high"] - center),
    )
    return {
        **row,
        "raw_interval": [row["ci_low"], row["ci_high"]],
        "ci_low": low,
        "ci_high": high,
        "covered": low <= row["truth"] <= high,
        "inside_margin": low > -row["margin"] and high < row["margin"],
    }


def _saved_rows(
    path: Path,
    header: dict[str, Any],
    case: dict[str, Any],
    phase: str,
    count: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["trials"]
    if (
        any(data.get(k) != header[k] for k in ("source_hash", "configuration_hash"))
        or data.get("case") != case
    ):
        raise ValueError("saved_fingerprint_case_requires_original_source_and_configuration")
    if (
        len(rows) != count
        or data.get("trials_hash") != content_hash(rows)
        or any(
            row["seed"] != named_seed(named_seed(cfg["seed"], phase, case["id"], i), "bootstrap")
            or row["n"] != case["cells"] * case["groups"]
            or row["truth"] != case["truth"]
            or not all(np.isfinite(row[k]) for k in ("estimate", "ci_low", "ci_high"))
            or row["ci_low"] > row["ci_high"]
            for i, row in enumerate(rows)
        )
    ):
        raise ValueError("complete_hashed_seeded_fingerprint_case_required")
    return list(rows)


def interval_study(out: Path) -> dict[str, Any]:
    """Calibrate widening on one bank, evaluate unchanged on a disjoint bank."""
    cfg, policy = configuration(), load_policy()
    result = receipt("fingerprint-product-intervals", cfg)
    path = out / "intervals.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(previous)
        expected_hash = payload.pop("result_hash", None)
        if expected_hash != content_hash(payload):
            raise ValueError("saved_interval_result_hash_mismatch")
        if (
            previous["source_hash"] != result["source_hash"]
            or previous["configuration_hash"] != result["configuration_hash"]
        ):
            raise ValueError("use_new_output_directory_for_changed_fingerprint_qualification")
        return dict(previous)
    cases = interval_cases(cfg)
    calibration, evaluation = [], []
    scales: dict[str, float] = {}
    with offline_only(), ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for case in cases:
            case_path = out / "calibration" / (case["id"] + ".json")
            jobs = [
                (
                    case,
                    named_seed(cfg["seed"], "interval-calibration", case["id"], i),
                    policy.alpha,
                    policy.bootstrap_resamples,
                )
                for i in range(cfg["calibration_replicates"])
            ]
            rows = _saved_rows(
                case_path, result, case, "interval-calibration", cfg["calibration_replicates"], cfg
            )
            if rows is None:
                rows = list(pool.map(_trial, jobs))
            ratios = sorted(_required_scale(row) for row in rows)
            allowed = [i for i in range(len(rows)) if wilson(i, len(rows))[1] <= policy.alpha]
            if not allowed:
                raise ValueError("more_interval_calibration_replicates_required")
            scale = ratios[len(rows) - max(allowed) - 1]
            scales[case["method"]] = max(scales.get(case["method"], 1), scale)
            calibration.append({"case": case, "scale": scale, "trials": rows})
            write_json(
                case_path,
                {**result, "case": case, "trials": rows, "trials_hash": content_hash(rows)},
            )
            print(
                f"calibration saved: {case['id']}; n={case['cells'] * case['groups']}", flush=True
            )
        for case in cases:
            case_path = out / "evaluation" / (case["id"] + ".json")
            jobs = [
                (
                    case,
                    named_seed(cfg["seed"], "interval-evaluation", case["id"], i),
                    policy.alpha,
                    policy.bootstrap_resamples,
                )
                for i in range(cfg["coverage_replicates"])
            ]
            rows = _saved_rows(
                case_path, result, case, "interval-evaluation", cfg["coverage_replicates"], cfg
            )
            if rows is None:
                rows = [_widen(row, scales[case["method"]]) for row in pool.map(_trial, jobs)]
            n = len(rows)
            covered = proportion(sum(r["ci_low"] <= case["truth"] <= r["ci_high"] for r in rows), n)
            equivalent = proportion(
                sum(r["ci_low"] > -case["margin"] and r["ci_high"] < case["margin"] for r in rows),
                n,
            )
            accepted = covered["ci_low"] >= cfg["minimum_coverage_lower"]
            if case["outside_equivalence"]:
                accepted &= equivalent["ci_high"] <= cfg["maximum_false_equivalence_upper"]
            if case["null"]:
                accepted &= equivalent["estimate"] >= cfg["minimum_clean_pass_rate"]
            evaluation.append(
                {
                    "case": case,
                    "coverage": covered,
                    "inside_margin": equivalent,
                    "accepted": accepted,
                    "trials": rows,
                }
            )
            write_json(
                case_path,
                {
                    **result,
                    **evaluation[-1],
                    "scale": scales[case["method"]],
                    "trials_hash": content_hash(rows),
                },
            )
            print(f"evaluation saved: {case['id']}; accepted={bool(accepted)}", flush=True)
    result.update(
        cases=cases,
        scales=scales,
        calibration=calibration,
        evaluation=evaluation,
        accepted=all(r["accepted"] for r in evaluation),
        scope="Known iid categorical component grid only; not universal language-model "
        "interval coverage or provider qualification",
    )
    result["result_hash"] = content_hash(result)
    write_json(path, result)
    return result
