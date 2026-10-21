"""Candidate-size null families on declared iid component processes.

This study tests conditional randomization/family error. It does not qualify
bootstrap equivalence, captured-wire power or hidden provider dependence.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from servegauge.core.hashing import content_hash
from servegauge.stats.fingerprint_strata import Cell
from servegauge.stats.multiple import holm
from servegauge.validation.common import named_seed, offline_only, proportion, receipt, write_json
from servegauge.validation.fingerprint_product import categorical_cells, configuration, null_p_value
from servegauge.verdicts.config import DecisionPolicy, load_policy

METRICS = ("FP1", "FP2", "FP3-mean", "FP3-dispersion")
PROCESS = "heterogeneous-categorical-and-gaussian-logit-iid-v1"


def null_components(cfg: dict[str, Any], seed: int) -> dict[str, list[Cell]]:
    """Vary categorical strata; preserve shared FP3 mean/second-moment data."""
    design = cfg["design"]
    for key, maximum in (
        ("fp1_cells", 40),
        ("fp2_cells", 25),
        ("fp1_samples", 100000),
        ("fp2_samples", 100000),
        ("fp3_samples", 100000),
    ):
        if type(design[key]) is not int or not 2 <= design[key] <= maximum:
            raise ValueError("bounded_null_component_design_required")
    grid = cfg["probability_grid"]
    if not isinstance(grid, list) or not grid:
        raise ValueError("predeclared_null_probability_grid_required")
    result: dict[str, list[Cell]] = {}
    for method in ("FP1", "FP2"):
        result[method] = []
        for index in range(design[method.lower() + "_cells"]):
            p = np.asarray(grid[index % len(grid)], dtype=np.float64)
            result[method].extend(
                categorical_cells(
                    method,
                    p,
                    p,
                    cells=1,
                    groups=design[method.lower() + "_samples"],
                    seed=named_seed(seed, method, index),
                )
            )
    jitter = cfg["healthy_logit_jitter"]
    if not isinstance(jitter, (int, float)) or not np.isfinite(jitter) or jitter < 0:
        raise ValueError("finite_nonnegative_null_logit_jitter_required")
    rng = np.random.default_rng(named_seed(seed, "FP3"))
    logits = rng.normal([0.1, -0.3, 0.8, 0.35], jitter, size=(4, design["fp3_samples"], 4))
    shifted = logits - logits.max(axis=2, keepdims=True)
    logs = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    anchors = [side[:, :2] for side in logs]
    result["FP3-mean"] = [Cell("FP3-mean", *anchors)]
    result["FP3-dispersion"] = [
        Cell("FP3-dispersion", *(np.concatenate((side, side**2), axis=1) for side in anchors))
    ]
    return result


def family_result(p_values: dict[str, float], *, alpha: float) -> dict[str, Any]:
    if set(p_values) != set(METRICS) or not 0 < alpha < 1:
        raise ValueError("complete_null_method_family_required")
    if any(not np.isfinite(p_values[m]) or not 0 <= p_values[m] <= 1 for m in METRICS):
        raise ValueError("finite_null_p_values_required")
    checks = {
        "FP1": p_values["FP1"],
        "FP2": p_values["FP2"],
        "FP3": min(1.0, 2 * min(p_values["FP3-mean"], p_values["FP3-dispersion"])),
    }
    adjusted = dict(zip(checks, holm(list(checks.values())), strict=True))
    return {
        "check_p_values": checks,
        "adjusted_p_values": adjusted,
        "family_rejected": any(p <= alpha for p in adjusted.values()),
    }


def null_trial(cfg: dict[str, Any], policy: DecisionPolicy, seed: int) -> dict[str, Any]:
    cells = null_components(cfg, seed)
    p_values = {
        m: null_p_value(
            cells[m],
            seed=named_seed(seed, "randomization", m),
            permutations=policy.permutation_resamples,
        )
        for m in METRICS
    }
    return {
        "seed": seed,
        "n": {m: sum(len(c.a) for c in cells[m]) for m in METRICS},
        "requested_permutations": policy.permutation_resamples,
        "p_values": p_values,
        **family_result(p_values, alpha=policy.alpha),
    }


def _checked_trial(
    row: dict[str, Any], cfg: dict[str, Any], policy: DecisionPolicy, seed: int
) -> dict[str, Any]:
    design = cfg["design"]
    expected_n = {
        "FP1": design["fp1_cells"] * design["fp1_samples"],
        "FP2": design["fp2_cells"] * design["fp2_samples"],
        "FP3-mean": design["fp3_samples"],
        "FP3-dispersion": design["fp3_samples"],
    }
    if (
        type(row.get("seed")) is not int
        or row["seed"] != seed
        or row.get("n") != expected_n
        or row.get("requested_permutations") != policy.permutation_resamples
    ):
        raise ValueError("matched_seeded_null_trial_required")
    derived = family_result(row["p_values"], alpha=policy.alpha)
    if any(row.get(key) != value for key, value in derived.items()):
        raise ValueError("consistent_null_family_inference_required")
    return row


def _cached_chunk(
    path: Path,
    header: dict[str, Any],
    seeds: list[int],
    cfg: dict[str, Any],
    policy: DecisionPolicy,
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(data.get(key) != header[key] for key in ("source_hash", "configuration_hash")):
        raise ValueError("null_resume_requires_original_source_and_configuration")
    rows = data["trials"]
    if len(rows) != len(seeds) or data.get("trials_hash") != content_hash(rows):
        raise ValueError("complete_hashed_null_chunk_required")
    return [_checked_trial(row, cfg, policy, seed) for row, seed in zip(rows, seeds, strict=True)]


def _summary(
    rows: list[dict[str, Any]], cfg: dict[str, Any], policy: DecisionPolicy
) -> dict[str, Any]:
    metrics = {
        m: proportion(sum(r["p_values"][m] <= policy.alpha for r in rows), len(rows))
        for m in METRICS
    }
    family = proportion(sum(r["family_rejected"] for r in rows), len(rows))
    return {
        "method_type_I": metrics,
        "holm_family": family,
        "accepted": all(v["estimate"] <= cfg["maximum_type_I"] for v in metrics.values())
        and family["estimate"] <= cfg["maximum_family_rate"],
    }


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    write_json(temporary, value)
    temporary.replace(path)


def null_study(out: Path) -> dict[str, Any]:
    cfg, policy = configuration(), load_policy()
    header = receipt(
        "fingerprint-product-null",
        {"design_configuration": cfg, "policy": policy.model_dump(mode="json"), "process": PROCESS},
    )
    path = out / "null.json"
    seeds = [named_seed(cfg["seed"], PROCESS, i) for i in range(cfg["null_replicates"])]
    if len(set(seeds)) != len(seeds):
        raise ValueError("distinct_null_trial_seeds_required")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(saved)
        if payload.pop("result_hash", None) != content_hash(payload):
            raise ValueError("saved_null_result_hash_mismatch")
        if any(saved.get(key) != header[key] for key in ("source_hash", "configuration_hash")):
            raise ValueError("null_result_requires_original_source_and_configuration")
        rows = saved["trials"]
        if len(rows) != len(seeds):
            raise ValueError("complete_null_result_required")
        for row, seed in zip(rows, seeds, strict=True):
            _checked_trial(row, cfg, policy, seed)
        if any(saved.get(k) != v for k, v in _summary(rows, cfg, policy).items()):
            raise ValueError("consistent_null_summary_required")
        return dict(saved)
    rows = []
    with offline_only(), ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for start in range(0, len(seeds), 25):
            chunk_seeds = seeds[start : start + 25]
            chunk_path = out / "chunks" / f"{start:06d}-{start + len(chunk_seeds):06d}.json"
            chunk = _cached_chunk(chunk_path, header, chunk_seeds, cfg, policy)
            if chunk is None:
                chunk = list(pool.map(lambda seed: null_trial(cfg, policy, seed), chunk_seeds))
                for row, seed in zip(chunk, chunk_seeds, strict=True):
                    _checked_trial(row, cfg, policy, seed)
                _save(chunk_path, {**header, "trials": chunk, "trials_hash": content_hash(chunk)})
            rows.extend(chunk)
            print(f"null families saved: {len(rows)}/{len(seeds)}", flush=True)
    result = {
        **header,
        **_summary(rows, cfg, policy),
        "trials": rows,
        "scope": "Known heterogeneous iid categorical strata and fixed-anchor Gaussian-logit "
        "components only; not provider dependence, interval coverage or captured-wire power",
    }
    result["result_hash"] = content_hash(result)
    _save(path, result)
    return result
