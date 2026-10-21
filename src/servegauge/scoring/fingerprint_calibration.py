"""Conservative signal cutoffs fitted on a separate healthy calibration bank."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Literal

import numpy as np
from pydantic import Field

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.scoring.fingerprint_evidence import FingerprintEvidence
from servegauge.stats.intervals import wilson


class SignalCalibration(ValueModel):
    version: Literal["robust-signal-calibration-v1"] = "robust-signal-calibration-v1"
    source_hash: str
    scope: str
    probe_configuration_hash: str
    reference_tier: Literal["MOCK"] = "MOCK"
    observations_per_session: int = Field(ge=1)
    permutations: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    maximum_family_rate: float = Field(gt=0, lt=1)
    by_groups: dict[str, dict[str, Any]]


def fit_signal_calibration(
    banks: list[dict[str, Any]], cfg: dict[str, Any], *, source_hash: str
) -> SignalCalibration:
    parameters = {}
    for groups in cfg["effective_group_sizes"]:
        samples = {}
        cuts = []
        for bank in banks:
            trials = next(
                b["trials"]
                for b in bank["batches"]
                if b["phase"] == "calibration" and b["groups"] == groups
            )
            if len(trials) != cfg["calibration_replicates"] or not all(
                t["eligible"] for t in trials
            ):
                raise ValueError("complete_independent_calibration_bank_required")
            values = sorted(min(r["adjusted_p"] for r in t["results"].values()) for t in trials)
            allowed = [
                k
                for k in range(len(values))
                if wilson(k, len(values))[1] <= cfg["maximum_family_false_positive_rate"]
            ]
            if not allowed:
                raise ValueError("more_calibration_replicates_required")
            cuts.append(float(np.nextafter(values[max(allowed)], -np.inf)))
            samples[bank["profile"]] = values
        cutoff = min(cfg["alpha"], *cuts)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("positive_calibrated_cutoff_unavailable")
        controls = {}
        for profile, values in samples.items():
            count = sum(v <= cutoff for v in values)
            controls[profile] = {
                "count": count,
                "n": len(values),
                "estimate": count / len(values),
                "interval": list(wilson(count, len(values))),
            }
        parameters[str(groups)] = {"cutoff": cutoff, "controls": controls}
    return SignalCalibration(
        source_hash=source_hash,
        scope=cfg["signal_scope"],
        probe_configuration_hash=content_hash(
            ["four-colors-fixed-anchors-v3", cfg["anchors"], "fixed-probe-configuration"]
        ),
        observations_per_session=cfg["observations_per_session"],
        permutations=cfg["permutations"],
        alpha=cfg["alpha"],
        maximum_family_rate=cfg["maximum_family_false_positive_rate"],
        by_groups=parameters,
    )


def match_signal_calibration(
    calibration: SignalCalibration,
    evidence: FingerprintEvidence,
    *,
    groups: int,
    alpha: float,
    permutations: int,
    current_source_hash: str | None,
) -> tuple[str | None, float | None]:
    if current_source_hash != calibration.source_hash:
        return "calibration_source_mismatch", None
    if (
        evidence.calibration_scope != calibration.scope
        or evidence.reference_tier != calibration.reference_tier
    ):
        return "calibration_scope_mismatch", None
    if (
        content_hash([evidence.probe_id, evidence.anchor_tokens, evidence.anchor_source])
        != calibration.probe_configuration_hash
    ):
        return "calibration_probe_mismatch", None
    if calibration.alpha != alpha or calibration.permutations != permutations:
        return "calibration_statistical_policy_mismatch", None
    if str(groups) not in calibration.by_groups:
        return "calibration_group_count_mismatch", None
    for rows in (evidence.target, evidence.reference):
        counts = Counter(r.session_id for r in rows)
        if any(n != calibration.observations_per_session for n in counts.values()):
            return "calibration_session_size_mismatch", None
    cutoff = float(calibration.by_groups[str(groups)]["cutoff"])
    if not math.isfinite(cutoff) or not 0 < cutoff <= alpha:
        return "invalid_calibration_cutoff", None
    return None, cutoff
