"""Label-blind contrasts using the unchanged raw fingerprint component statistics."""

from __future__ import annotations

import re
from typing import Any, cast

from servegauge.embeddings.local import LocalEncoder
from servegauge.scoring.common import Samples
from servegauge.scoring.fingerprint_comparison import collection_cells
from servegauge.stats.fingerprint_energy import EnergyCell, compare_energy_cells
from servegauge.stats.fingerprint_strata import compare_cells
from servegauge.verdicts.config import DecisionPolicy

METHODS = ("FP1", "FP2", "FP3-mean", "FP3-dispersion", "FP4")
COUNTS = (
    "expected_requests",
    "requests",
    "captures",
    "missing_responses",
    "missing_logprobs",
    "censored_responses",
    "invalid_responses",
    "duplicate_identities",
)


def score_quartet(
    samples: list[dict[str, Samples]],
    *,
    policy: DecisionPolicy,
    seed: int,
    encoder: LocalEncoder | None = None,
) -> dict[str, dict[str, Any]]:
    """One score per method; intervention labels are deliberately not accepted.

    Caller reconstructs and binds raw runs before calling this pure function. Four
    exclusive captures provide target/control minus control/control distance. This
    experimental contrast is not qualified reference evidence or a serving FAIL.
    """
    if len(samples) != 4:
        raise ValueError("four_exclusive_study_captures_required")
    output = {}
    for method in METHODS:
        base = "FP3" if method.startswith("FP3-") else method
        present = [side.get(base) for side in samples]
        counts = [
            {name: side.details.get(name) for name in COUNTS}
            if side
            else {name: None for name in COUNTS}
            for side in present
        ]
        row: dict[str, Any] = {
            "method": method,
            "reference_tier": "BASELINE",
            "verdict": "INCONCLUSIVE",
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_components": 0,
            "capture_counts": counts,
            "reason": "method_not_collected",
            "interval_scope": "empirical_conditional_component_bootstrap; not_provider_qualified",
        }
        if any(side is None for side in present):
            output[method] = row
            continue
        try:
            matched = cast(list[Samples], present)
            cells = collection_cells(matched, method=method, policy=policy, encoder=encoder)
            result = (
                compare_energy_cells(
                    cast(list[EnergyCell], cells),
                    seed=seed,
                    permutations=policy.permutation_resamples,
                    resamples=policy.bootstrap_resamples,
                    alpha=policy.alpha,
                )
                if method == "FP4"
                else compare_cells(
                    cells,
                    seed=seed,
                    permutations=policy.permutation_resamples,
                    resamples=policy.bootstrap_resamples,
                    alpha=policy.alpha,
                )
            )
            minimum = matched[0].details["design"]["minimum_independent_groups"]
            row.update(
                estimate=result.delta,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                n_components=result.n,
                groups_per_cell=[len(c.a) for c in cells],
                target_distance=result.estimate,
                control_distance=result.reference_estimate,
                reason="descriptive_unqualified_study_contrast"
                if all(len(c.a) >= minimum for c in cells)
                else "underpowered_descriptive_study_contrast",
            )
        except ValueError as error:
            reason = error.args[0] if error.args else "unavailable_study_metric"
            row["reason"] = (
                reason
                if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]+", reason)
                else "unavailable_study_metric"
            )
        output[method] = row
    return output
