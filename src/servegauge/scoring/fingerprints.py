"""Matched-cell fingerprint comparisons with explicit same/different controls.

These functions operate on observed completions. They cannot attest model identity.
FP4 text embedding integration remains optional; its statistic is studied separately.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from servegauge.core.models import CheckResult
from servegauge.refs.calibrate import calibrate
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.jsd import jsd_bits
from servegauge.stats.mmd import hamming_mmd
from servegauge.verdicts.config import DecisionPolicy
from servegauge.verdicts.engine import Measurement, decide_family

type Cells = list[list[str]]


def distances(first: Cells, second: Cells, *, method: str, length: int) -> list[float]:
    if (
        not first
        or len(first) != len(second)
        or any(len(a) != len(b) or len(a) < 2 for a, b in zip(first, second, strict=True))
    ):
        raise ValueError("matched_prompt_cells_and_sample_counts_required")
    result = []
    for a, b in zip(first, second, strict=True):
        if method == "FP1":
            vocabulary = sorted(set(a) | set(b))
            ca, cb = Counter(a), Counter(b)
            result.append(jsd_bits([ca[x] for x in vocabulary], [cb[x] for x in vocabulary]))
        elif method == "FP2":
            result.append(hamming_mmd(a, b, length=length) / length)
        else:
            raise ValueError("unsupported_fingerprint_method")
    return result


def fingerprint_checks(
    target: Cells,
    references: list[Cells],
    controls: list[Cells],
    *,
    seed: int,
    length: int,
    policy: DecisionPolicy,
    extra: list[Measurement] | None = None,
    evidence: list[str] | None = None,
) -> list[CheckResult]:
    if len(references) < 3 or len(controls) < 2:
        raise ValueError("three_clean_references_and_two_different_model_controls_required")
    measurements = list(extra or [])
    bands: dict[str, dict[str, Any]] = {}
    for method in ("FP1", "FP2"):
        observed = distances(target, references[0], method=method, length=length)
        null = np.mean(
            [
                distances(references[i], references[j], method=method, length=length)
                for i, j in ((0, 1), (0, 2), (1, 2))
            ],
            axis=0,
        ).tolist()
        calibration = calibrate(null, seed=seed, resamples=policy.bootstrap_resamples)
        same = clustered_bootstrap(null, seed=seed, resamples=policy.bootstrap_resamples)
        different = [
            clustered_bootstrap(
                distances(control, references[0], method=method, length=length),
                seed=seed,
                resamples=policy.bootstrap_resamples,
            )
            for control in controls
        ]
        bands[method] = {
            "same_model_band": [same.low, same.high],
            "different_model_bands": [[d.low, d.high] for d in different],
            "different_model_lower_boundary": min(d.low for d in different),
            "n_cells": len(target),
            "samples_per_cell": len(target[0]),
            "fixed_length": length,
            "calibration": calibration,
        }
        measurements.append(
            Measurement(
                check_id=method,
                suite_id="S6",
                endpoint_id="mock:target",
                run_id="validation-" + str(seed),
                check_class="comparative",
                outcomes=[-v for v in observed],
                reference=[-v for v in null],
                reference_tier="MOCK",
                binary=False,
                two_sided=True,
                noise_band=float(str(calibration["noise_band"])),
                seed=seed,
                evidence=evidence or [],
                details={
                    "metric": "mean JSD bits"
                    if method == "FP1"
                    else "mean normalized unbiased Hamming MMD",
                    "independent_unit": "prompt cell; all repeated completions clustered",
                    "negative_distance_orientation": True,
                    "fingerprint_bands": bands[method],
                    "limitations": "matched synthetic controls; no identity or quantization proof",
                },
            )
        )
    results = decide_family(measurements, policy)
    for result in results:
        if result.check_id not in bands:
            continue
        band = bands[result.check_id]
        distance = -float(result.estimate or 0)
        result.details["fingerprint_position"] = (
            "different-model-band"
            if distance >= band["different_model_lower_boundary"]
            else "outside-same-model-band"
            if distance > band["same_model_band"][1]
            else "same-model-band"
        )
    return results
