"""Captured scripted stimuli and blind summaries for held-out S6 power studies.

Fault truth only chooses the stimulus. Comparison and outcome summaries consume
captured observations and ordinary check results, never the injected fault label.
"""

from __future__ import annotations

from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import CheckResult, Item
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun
from servegauge.scoring.fingerprint_collection import fingerprint_samples
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import DecisionPolicy


def capture_run(
    items: list[Item],
    policy: DecisionPolicy,
    *,
    seeds: list[int],
    source_id: str,
    healthy_jitter: float,
    fault: FaultConfig | None = None,
    control: str | None = None,
    reference: bool = False,
) -> tuple[ReferenceRun, list[dict[str, Any]]]:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or (reference and (fault is not None or control is not None))
    ):
        raise ValueError("distinct_clean_reference_seeds_required")
    observed, endpoints, declarations, receipts = {}, [], [], []
    for seed in seeds:
        ep = endpoint("mock:wire-" + content_hash([seed, "endpoint"])[:20], seed, clean=reference)
        ep.metadata["mock_declaration"].update(
            backend="other:" + control if control else "scripted",
            sampling_version="v2",
            healthy_logit_jitter=healthy_jitter,
        )
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
        backend = ScriptedBackend(
            seed,
            fault or FaultConfig(),
            model_control=control,
            sampling_version="v2",
            healthy_logit_jitter=healthy_jitter,
        )
        records = captures(items, backend, ep, source_id)
        observed[ep.id] = fingerprint_samples(items, records, ep, policy)
        endpoints.append(
            ReferenceEndpoint(
                id=ep.id, profile=ep.claimed_profile, kind="mock", seed=seed, clean=reference
            )
        )
        declarations.append(ep.model_dump(mode="json"))
        receipts.append(
            {
                "seed": seed,
                "endpoint_id": ep.id,
                "run_id": source_id,
                "captures": len(records),
                "record_hash": content_hash([r.model_dump(mode="json") for r in records]),
                "observation_hashes": {
                    m: s.details["observation_hash"] for m, s in observed[ep.id].items()
                },
            }
        )
    run = ReferenceRun(
        run_id=source_id,
        profile="gpt-oss-120b",
        configuration={"policy_hash": content_hash(policy.model_dump(mode="json"))},
        item_set_hashes={"S6": content_hash([i.model_dump(mode="json") for i in items])},
        endpoints=endpoints,
        checks={},
        fingerprints=observed,
        fingerprint_policy=policy,
        manifest={"endpoint_metadata": declarations},
    )
    return run, receipts


def outcome_signals(checks: list[CheckResult], policy: DecisionPolicy) -> dict[str, bool]:
    """Prospective text equivalence and observed fault signatures, without truth labels."""
    names = [c.check_id for c in checks]
    if len(set(names)) != len(names):
        raise ValueError("unique_fingerprint_checks_required")
    empty = {
        "text_equivalence_candidate": False,
        "swap_signature": False,
        "mixture_signature": False,
        "noise_signature": False,
        "any_comparative_fail": any(
            c.check_class == "comparative" and c.verdict == "FAIL" for c in checks
        ),
        "logprob_signal": any(
            c.check_id == "FP3" and c.verdict == "INFO" and c.details.get("signal_detected") is True
            for c in checks
        ),
    }
    if set(names) != {"FP1", "FP2", "FP3"}:
        return empty
    by_id = {c.check_id: c for c in checks}
    text = [by_id[m] for m in ("FP1", "FP2")]
    logs = by_id["FP3"]
    metrics = {c.check_id: c.details.get("fingerprint_metrics", {}) for c in checks}
    if any(c.check_id not in metrics[c.check_id] for c in text) or set(metrics["FP3"]) != {
        "FP3-mean",
        "FP3-dispersion",
    }:
        return empty

    def between(value: dict[str, Any]) -> bool:
        return any(
            row.get("position") == "between-bands" for row in value.get("control_positions", [])
        )

    def inside(c: CheckResult) -> bool:
        noise = c.details.get("noise_band")
        margin = policy.margin(c.check_id)
        return bool(
            c.reference_tier == "MOCK"
            and c.n >= 2
            and c.verdict in {"PASS", "INCONCLUSIVE"}
            and isinstance(noise, (int, float))
            and 0 <= noise <= margin
            and c.p_value is not None
            and c.delta_ci_low is not None
            and c.delta_ci_high is not None
            and c.delta_ci_low > -margin
            and c.delta_ci_high < margin
        )

    log_signal = logs.verdict == "INFO" and logs.details.get("signal_detected") is True
    dispersion = metrics["FP3"]["FP3-dispersion"]
    p = dispersion.get("p_value")
    # Conservative attribution to this subtest, even if the other FP3 subtest drove
    # the combined Holm signal. No uncorrected subtest is called a noise detection.
    dispersion_detected = (
        isinstance(p, (int, float))
        and 0 <= p <= 1
        and min(1, len(checks) * len(metrics["FP3"]) * p) <= policy.alpha
    )
    lower = dispersion.get("delta_ci_low")
    return {
        "text_equivalence_candidate": all(inside(c) for c in text),
        "swap_signature": all(
            c.verdict == "FAIL"
            and metrics[c.check_id][c.check_id].get("position") == "different-model-band-or-beyond"
            for c in text
        ),
        "mixture_signature": by_id["FP1"].verdict == "FAIL"
        and between(metrics["FP1"]["FP1"])
        and log_signal
        and any(between(v) for v in metrics["FP3"].values()),
        "noise_signature": bool(
            log_signal
            and dispersion_detected
            and isinstance(lower, (int, float))
            and lower > 0
            and between(dispersion)
        ),
        "any_comparative_fail": any(c.verdict == "FAIL" for c in text),
        "logprob_signal": log_signal,
    }
