"""Fixed named seed streams; null operating characteristics, not detector validation claims."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy.stats import norm

from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.energy import energy_test
from servegauge.stats.intervals import paired_score_z, wilson
from servegauge.stats.jsd import jsd_test
from servegauge.stats.logprob_track import logprob_test
from servegauge.stats.mmd import hamming_test
from servegauge.stats.multiple import benjamini_hochberg, holm
from servegauge.stats.tests import midp_mcnemar, paired_permutation, wilcoxon_test


def _rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _rate(count: int, n: int, assumptions: str, alpha: float) -> dict[str, Any]:
    return {
        "count": count,
        "n": n,
        "estimate": count / n,
        "interval": list(wilson(count, n)),
        "reference_tier": "MOCK",
        "reference_estimate": alpha,
        "assumptions": assumptions,
    }


def null_simulations(*, replicates: int, seed: int, permutations: int = 99) -> dict[str, Any]:
    if replicates < 1 or permutations < 1:
        raise ValueError("positive_simulation_sizes_required")
    alpha = 0.05
    results: dict[str, Any] = {}
    definitions = {
        "wilson": ("asymptotic", "independent Binomial(100,0.5); two-sided 95% coverage"),
        "tango": ("asymptotic", "100 independent paired items; +/- discordance each 0.1"),
        "mcnemar_midp": ("conservative", "100 paired items; equal discordance probabilities 0.1"),
        "wilcoxon": (
            "asymptotic",
            "60 independent continuous symmetric standard-normal differences",
        ),
        "paired_permutation": (
            "conservative",
            "12 independent symmetric standard-normal paired differences",
        ),
        "jsd": ("conservative", "one fixed cell; 8 iid categorical draws per side, 4 categories"),
        "mmd_hamming": (
            "conservative",
            "8 iid length-6 strings per side, uniform 4-character alphabet",
        ),
        "energy": ("conservative", "8 iid two-dimensional standard-normal vectors per side"),
        "logprob": (
            "conservative",
            "8 iid negative logprob vectors per side, random missing tokens; floor -30",
        ),
    }
    z = float(norm.ppf(1 - alpha / 2))
    for name, (kind, assumption) in definitions.items():
        rng = _rng(seed, name)
        rejected = 0
        for _ in range(replicates):
            permutation_seed = int(rng.integers(0, 2**63 - 1))
            if name == "wilson":
                low, high = wilson(int(rng.binomial(100, 0.5)), 100)
                reject = not low <= 0.5 <= high
            elif name in {"tango", "mcnemar_midp"}:
                positive, negative, _ = rng.multinomial(100, [0.1, 0.1, 0.8])
                reject = (
                    abs(paired_score_z(int(positive), int(negative), 100, 0)) > z
                    if name == "tango"
                    else midp_mcnemar(int(positive), int(negative)) <= alpha
                )
            elif name == "wilcoxon":
                reject = wilcoxon_test(rng.normal(size=60), method="asymptotic").p_value <= alpha
            elif name == "paired_permutation":
                reject = (
                    paired_permutation(
                        rng.normal(size=12), seed=permutation_seed, resamples=permutations
                    ).p_value
                    <= alpha
                )
            elif name == "jsd":
                first = [[str(value) for value in rng.choice(4, size=8, p=[0.4, 0.3, 0.2, 0.1])]]
                second = [[str(value) for value in rng.choice(4, size=8, p=[0.4, 0.3, 0.2, 0.1])]]
                reject = (
                    jsd_test(first, second, seed=permutation_seed, resamples=permutations).p_value
                    <= alpha
                )
            elif name == "mmd_hamming":
                strings = ["".join(row) for row in rng.choice(list("abcd"), size=(16, 6))]
                reject = (
                    hamming_test(
                        strings[:8],
                        strings[8:],
                        length=6,
                        seed=permutation_seed,
                        resamples=permutations,
                    ).p_value
                    <= alpha
                )
            elif name == "energy":
                reject = (
                    energy_test(
                        rng.normal(size=(8, 2)),
                        rng.normal(size=(8, 2)),
                        seed=permutation_seed,
                        resamples=permutations,
                    ).p_value
                    <= alpha
                )
            else:
                rows = [
                    {
                        token: -2 - abs(float(rng.normal()))
                        for token in ("a", "b")
                        if rng.random() > 0.1
                    }
                    for _ in range(16)
                ]
                # A token-free draw has no observable logprob distribution; regenerate once
                # as a declared token rather than silently reporting a p-value for empty data.
                for row in rows:
                    if not row:
                        row["a"] = -2 - abs(float(rng.normal()))
                reject = (
                    logprob_test(
                        rows[:8],
                        rows[8:],
                        missing_floor=-30,
                        seed=permutation_seed,
                        resamples=permutations,
                    ).p_value
                    <= alpha
                )
            rejected += int(reject)
        results[name] = {**_rate(rejected, replicates, assumption, alpha), "class": kind}
    # Holm permits arbitrary dependence. This family models the shared-item setting;
    # a separate Bonferroni union-bound known-answer test checks independent-family control.
    rng = _rng(seed, "holm")
    family_size, correlation = 10, 0.5
    rejected = 0
    for _ in range(replicates):
        shared = float(rng.normal())
        scores = math.sqrt(correlation) * shared + math.sqrt(1 - correlation) * rng.normal(
            size=family_size
        )
        p_values = [float(norm.sf(value)) for value in scores]
        rejected += int(any(p <= alpha for p in holm(p_values)))
    holm_report = _rate(
        rejected,
        replicates,
        "10 complete-null one-sided Gaussian p-values; equicorrelation 0.5",
        alpha,
    )
    rng = _rng(seed, "bh")
    false_discovery_proportions = []
    for _ in range(replicates):
        p_values = [*rng.random(8).tolist(), *rng.uniform(0, 1e-6, size=2).tolist()]
        adjusted = benjamini_hochberg(p_values)
        false = sum(p <= alpha for p in adjusted[:8])
        discoveries = sum(p <= alpha for p in adjusted)
        false_discovery_proportions.append(false / max(discoveries, 1))
    fdr = clustered_bootstrap(
        false_discovery_proportions,
        seed=int.from_bytes(hashlib.sha256(f"{seed}:bh-ci".encode()).digest()[:8], "big"),
    )
    bh_report = {
        "n": replicates,
        "estimate": fdr.estimate,
        "interval": [fdr.low, fdr.high],
        "reference_tier": "MOCK",
        "reference_estimate": alpha,
        "assumptions": (
            "10 independent tests: 8 uniform nulls and 2 strong alternatives; "
            "FDP bootstrapped by replicate"
        ),
    }
    return {
        "seed": seed,
        "replicates": replicates,
        "permutations_per_randomized_test": permutations,
        "tests": results,
        "holm": holm_report,
        "bh": bh_report,
        "limitations": (
            "Finite synthetic scenarios check implementation; they do not validate "
            "real endpoints or observatory-wide alarms. Reference estimate is nominal alpha."
        ),
    }
