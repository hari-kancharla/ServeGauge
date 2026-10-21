"""Offline session/caching controls, independent calibration and held-out evaluation."""

from __future__ import annotations

import copy
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from servegauge.core.hashing import content_hash
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.fingerprint_analysis import METHODS, analyze_evidence, group_evidence
from servegauge.scoring.fingerprint_calibration import fit_signal_calibration
from servegauge.scoring.fingerprint_evidence import FingerprintEvidence, prepare_evidence
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.fingerprint_blocks import block_comparisons
from servegauge.stats.logprob_dispersion import logprob_dispersion_test
from servegauge.stats.multiple import holm
from servegauge.stats.paired_fast import jsd_test_fast, logprob_test_fast
from servegauge.validation.common import (
    ROOT,
    cached,
    named_seed,
    offline_only,
    proportion,
    receipt,
    write_json,
)
from servegauge.verdicts.config import load_policy

OUT = Path("docs/report/robustness-v3-1")
GUARDS = (
    "missing_all",
    "missing_one",
    "missing_anchor",
    "changing_topk",
    "unknown_dependence",
    "mismatched_sampling",
    "mismatched_cache",
    "one_cache_component",
    "cache_hit_missing",
    "invalid_probability",
    "missing_swap",
)
BODY = {
    "model": "mock",
    "messages": [{"role": "user", "content": 'TW_TASK: {"choices":["red","blue","green","gold"]}'}],
    "temperature": 1.0,
    "logprobs": True,
    "top_logprobs": 4,
    "max_tokens": 8,
}


def configuration() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/robustness_validation.yaml").read_text(encoding="utf-8")
    )
    policy = load_policy()
    cfg.update(
        alpha=policy.alpha,
        logprob_mass_tolerance=float(policy.entries["P06"]["prob_sum_tolerance"]),
    )
    assert cfg["permutations"] == policy.permutation_resamples
    original = FaultConfig.model_validate(
        yaml.safe_load((ROOT / "configs/faults/qsim.yaml").read_text(encoding="utf-8"))
    )
    assert cfg["fault_logit_noise"] == original.logit_noise
    return cfg


class SessionBackend(ScriptedBackend):
    """Versioned four-choice Gaussian-logit process, stationary within each session."""

    def __init__(self, seed: int, *, jitter: float, rho: float, noise: float, swap: bool) -> None:
        super().__init__(
            seed,
            FaultConfig(logit_noise=noise, route_to_other_model=float(swap)),
            sampling_version="v2",
        )
        self.jitter, self.rho = jitter, rho
        self.latent = [self.rng.gauss(0, 1) for _ in range(4)]

    def _sample(
        self, choices: list[str], body: dict[str, Any], other: bool, rng: random.Random
    ) -> str:
        if len(choices) != 4 or float(body.get("temperature", 1)) != 1:
            raise ValueError("fixed_four_choice_validation_model_required")
        self.latent = [
            self.rho * z + math.sqrt(1 - self.rho**2) * rng.gauss(0, 1) for z in self.latent
        ]
        logits = [-0.65 * i for i in range(4)]
        if other:
            logits.reverse()
        logits = [
            v
            + self.jitter * z
            + (rng.gauss(0, self.faults.logit_noise) if self.faults.logit_noise else 0)
            for v, z in zip(logits, self.latent, strict=True)
        ]
        weights = [math.exp(v - max(logits)) for v in logits]
        total = sum(weights)
        self.sample_distribution = {
            token: w / total for token, w in zip(choices, weights, strict=True)
        }
        return rng.choices(choices, weights, k=1)[0]


def generate_evidence(
    seed: int, profile: str, *, groups: int, noise: float, cfg: dict[str, Any], swap: bool = False
) -> FingerprintEvidence:
    if profile not in cfg["profiles"] or groups < 1:
        raise ValueError("known_profile_and_positive_group_count_required")
    sides: dict[str, Any] = {}
    for side in ("target", "reference"):
        rows = []
        for group in range(groups):
            backend = SessionBackend(
                named_seed(seed, side, group),
                jitter=cfg["healthy_logit_jitter"],
                rho=cfg["ar1_rho"] if profile == "correlated" else 0,
                noise=noise if side == "target" else 0,
                swap=swap and side == "target",
            )
            burst = cfg["cache_burst_length"] if profile in {"cache_bursts", "linked_cache"} else 1
            generated: list[dict[str, Any]] = []
            for draw in range(cfg["observations_per_session"]):
                generated.append(
                    backend.respond(BODY) if draw % burst == 0 else copy.deepcopy(generated[-1])
                )
            copies = 2 if profile == "linked_cache" else 1
            for duplicate in range(copies):
                for draw, response in enumerate(generated):
                    rows.append(
                        {
                            "session_id": f"s-{group * copies + duplicate}",
                            "draw": draw,
                            "response": copy.deepcopy(response),
                            "cache_group": f"{side}-{group}-{draw // burst}" if burst > 1 else None,
                        }
                    )
        sides[side] = rows
    return FingerprintEvidence(
        probe_id="four-colors-fixed-anchors-v3",
        reference_tier="MOCK",
        independent_sessions=True,
        matched_sampling_design=True,
        calibration_scope=cfg["signal_scope"],
        anchor_tokens=cfg["anchors"],
        anchor_source="fixed-probe-configuration",
        seed=seed,
        **sides,
    )


def primary_trial(
    seed: int, profile: str, groups: int, noise: float, cfg: dict[str, Any], *, legacy: bool
) -> dict[str, Any]:
    evidence = generate_evidence(seed, profile, groups=groups, noise=noise, cfg=cfg)
    prepared = prepare_evidence(
        evidence,
        minimum_anchor_tokens=cfg["minimum_anchor_tokens"],
        tolerance=cfg["logprob_mass_tolerance"],
    )
    grouping = group_evidence(evidence, prepared)
    methods = block_comparisons(
        prepared,
        grouping.pairs,
        anchors=evidence.anchor_tokens,
        seed=seed,
        resamples=cfg["permutations"],
    )
    adjusted = holm([methods[name].test.p_value if name in methods else 1 for name in METHODS])
    result: dict[str, Any] = {
        "seed": seed,
        "eligible": len(methods) == 3 and grouping.reason is None and len(grouping.pairs) == groups,
        "effective_groups": len(grouping.pairs),
        "requests_both_sides": len(evidence.target) + len(evidence.reference),
        "response_lineages_both_sides": sum(grouping.response_lineages.values()),
        "parsed_observation_hash": content_hash(
            [
                [
                    (s.session_id, s.draw, s.cache_group, r.text, r.logprobs, r.reason)
                    for s, r in zip(rows, parsed, strict=True)
                ]
                for rows, parsed in (
                    (evidence.target, prepared.first),
                    (evidence.reference, prepared.second),
                )
            ]
        ),
        "results": {
            name: {
                "statistic": methods[name].test.statistic,
                "p_value": methods[name].test.p_value,
                "adjusted_p": p,
                "n": groups,
            }
            for name, p in zip(METHODS, adjusted, strict=True)
            if name in methods
        },
    }
    if legacy:
        old = {
            "FP1": jsd_test_fast(
                [str(r.text) for r in prepared.first],
                [str(r.text) for r in prepared.second],
                seed=seed,
                resamples=cfg["permutations"],
            ),
            "FP3-mean": logprob_test_fast(
                prepared.first_logs,
                prepared.second_logs,
                missing_floor=float(load_policy().entries["FP3"]["missing_logprob_floor"]),
                seed=seed,
                resamples=cfg["permutations"],
            ),
            "FP3-dispersion": logprob_dispersion_test(
                prepared.first_logs, prepared.second_logs, seed=seed, resamples=cfg["permutations"]
            ),
        }
        adjusted_old = holm([r.p_value for r in old.values()])
        result["rowwise_comparator"] = {
            name: {"p_value": r.p_value, "adjusted_p": p, "n": r.n}
            for (name, r), p in zip(old.items(), adjusted_old, strict=True)
        }
    return result


def guarded_evidence(seed: int, guard: str, cfg: dict[str, Any]) -> FingerprintEvidence:
    evidence = generate_evidence(
        seed,
        "cache_bursts" if guard == "cache_hit_missing" else "iid",
        groups=cfg["evaluation_groups"],
        noise=0,
        cfg=cfg,
        swap=guard == "missing_swap",
    )
    if guard in {"missing_all", "missing_swap"}:
        for sample in evidence.target + evidence.reference:
            assert sample.response is not None
            sample.response["choices"][0]["logprobs"] = None
    elif guard in {"missing_one", "cache_hit_missing"}:
        assert evidence.target[1].response is not None
        evidence.target[1].response["choices"][0]["logprobs"] = None
    elif guard == "missing_anchor":
        assert evidence.target[0].response is not None
        row = evidence.target[0].response["choices"][0]["logprobs"]["content"][0]
        row["top_logprobs"] = [r for r in row["top_logprobs"] if r["token"] != cfg["anchors"][0]]
    elif guard == "changing_topk":
        for i, sample in enumerate(evidence.target + evidence.reference):
            assert sample.response is not None
            row = sample.response["choices"][0]["logprobs"]["content"][0]
            top = row["top_logprobs"]
            required = max(j + 1 for j, r in enumerate(top) if r["token"] in cfg["anchors"])
            row["top_logprobs"] = top[: required if i % 2 else len(top)]
    elif guard == "unknown_dependence":
        evidence.independent_sessions = False
    elif guard == "mismatched_sampling":
        evidence.matched_sampling_design = False
    elif guard == "mismatched_cache":
        evidence.target[0].cache_group = evidence.target[1].cache_group = "different-design"
        evidence.target[1].response = copy.deepcopy(evidence.target[0].response)
    elif guard == "one_cache_component":
        for side in (evidence.target, evidence.reference):
            for sample in side:
                sample.cache_group = "one-response"
                sample.response = copy.deepcopy(side[0].response)
    elif guard == "invalid_probability":
        assert evidence.target[0].response is not None
        evidence.target[0].response["choices"][0]["logprobs"]["content"][0]["logprob"] = True
    else:
        raise ValueError("known_guard_required")
    return evidence


def _profile_study(args: tuple[dict[str, Any], str, str, str]) -> dict[str, Any]:
    cfg, profile, directory, stage = args
    batches = []
    with offline_only():
        phases = (
            [
                ("calibration", g, cfg["calibration_replicates"])
                for g in cfg["effective_group_sizes"]
            ]
            if stage == "calibration"
            else [
                ("null", cfg["evaluation_groups"], cfg["null_replicates"]),
                *[("fault", g, cfg["fault_replicates"]) for g in cfg["effective_group_sizes"]],
            ]
        )
        for phase, groups, count in phases:
            trials = []
            for i in range(count):
                seed = named_seed(cfg["seed"], cfg["version"], profile, phase, groups, i)
                path = Path(directory) / f"trial-{seed}.json"
                trial = cached(path)
                if trial is None:
                    trial = primary_trial(
                        seed,
                        profile,
                        groups,
                        cfg["fault_logit_noise"] if phase == "fault" else 0,
                        cfg,
                        legacy=phase == "null",
                    )
                    write_json(path, trial)
                trials.append(trial)
                if (i + 1) % 250 == 0:
                    print(
                        f"robust-v3.1 {profile} {phase} groups={groups}: {i + 1}/{count}",
                        flush=True,
                    )
            batches.append({"phase": phase, "groups": groups, "trials": trials})
            print(f"robust-v3.1 {profile} {phase} groups={groups}: complete", flush=True)
    return {"profile": profile, "batches": batches}


def robustness_study(*, out: Path = OUT) -> dict[str, Any]:
    cfg = configuration()
    output = receipt("robustness-v3.1", cfg)
    before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    directory = ROOT / ".cache/robust-v3" / content_hash([cfg, output["source_hash"]])
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        banks = list(
            pool.map(
                _profile_study, [(cfg, p, str(directory), "calibration") for p in cfg["profiles"]]
            )
        )
    signal_policy = fit_signal_calibration(banks, cfg, source_hash=output["source_hash"])
    # Seal the calibration before any fresh held-out evaluation is generated.
    write_json(out / "signal-calibration.json", signal_policy.model_dump(mode="json"))
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        evaluations = list(
            pool.map(
                _profile_study, [(cfg, p, str(directory), "evaluation") for p in cfg["profiles"]]
            )
        )
    profiles = [
        {"profile": b["profile"], "batches": b["batches"] + e["batches"]}
        for b, e in zip(banks, evaluations, strict=True)
    ]
    output["signal_calibration"] = signal_policy.model_dump(mode="json")
    rows, calibration = [], []
    for profile in profiles:
        for batch in profile["batches"]:
            trials = batch["trials"]
            if batch["phase"] == "calibration":
                for method in METHODS:
                    values = [t["results"][method]["statistic"] for t in trials]
                    band = clustered_bootstrap(
                        values,
                        seed=named_seed(cfg["seed"], profile["profile"], method, "band"),
                        statistic=lambda x: float(np.quantile(x, 1 - cfg["alpha"])),
                        resamples=2000,
                    )
                    calibration.append(
                        {
                            "profile": profile["profile"],
                            "method": method,
                            "groups": batch["groups"],
                            "n": len(values),
                            "noise_band": band.estimate,
                            "ci_low": band.low,
                            "ci_high": band.high,
                            "reference_tier": "MOCK",
                        }
                    )
                continue
            cutoff = float(signal_policy.by_groups[str(batch["groups"])]["cutoff"])
            for method in (*METHODS, "family"):
                count = sum(
                    any(r["adjusted_p"] <= cutoff for r in t["results"].values())
                    if method == "family"
                    else t["results"][method]["adjusted_p"] <= cutoff
                    for t in trials
                )
                rows.append(
                    {
                        "profile": profile["profile"],
                        "signal_cutoff": cutoff,
                        "phase": batch["phase"],
                        "groups": batch["groups"],
                        "method": method,
                        "rate": proportion(count, len(trials)),
                        "all_eligible": all(t["eligible"] for t in trials),
                        "requests_both_sides": trials[0]["requests_both_sides"],
                        "response_lineages_both_sides": trials[0]["response_lineages_both_sides"],
                    }
                )
            if batch["phase"] == "null":
                count = sum(
                    any(r["adjusted_p"] <= cutoff for r in t["rowwise_comparator"].values())
                    for t in trials
                )
                rows.append(
                    {
                        "profile": profile["profile"],
                        "signal_cutoff": cutoff,
                        "phase": "null",
                        "groups": batch["groups"],
                        "method": "unqualified-rowwise-family",
                        "rate": proportion(count, len(trials)),
                    }
                )
    guards: list[dict[str, Any]] = []
    policy = load_policy()
    with offline_only():
        for guard in GUARDS:
            results = []
            for i in range(cfg["guard_replicates"]):
                seed = named_seed(cfg["seed"], cfg["version"], "guard", guard, i)
                path = directory / f"guard-{seed}.json"
                result = cached(path)
                if result is None:
                    evidence = guarded_evidence(seed, guard, cfg)
                    result = {
                        "seed": seed,
                        "analysis": analyze_evidence(
                            evidence,
                            policy=policy,
                            minimum_groups=cfg["minimum_independent_groups"],
                            minimum_anchor_tokens=cfg["minimum_anchor_tokens"],
                            signal_calibration=signal_policy,
                            current_source_hash=output["source_hash"],
                        ),
                    }
                    write_json(path, result)
                results.append(result)
            guards.append({"guard": guard, "results": results})
            print(f"robust-v3.1 guard {guard}: {len(results)} complete", flush=True)
    output.update(
        profiles=profiles,
        rows=rows,
        calibration=calibration,
        guards=guards,
        spend_hash_before=before,
        spend_hash_after=content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")),
    )
    output["accepted"] = (
        before == output["spend_hash_after"]
        and all(r.get("all_eligible", True) for r in rows)
        and all(
            r["rate"]["estimate"] <= cfg["maximum_family_false_positive_rate"]
            for r in rows
            if r["phase"] == "null" and r["method"] == "family"
        )
        and all(
            r["rate"]["ci_low"] >= cfg["minimum_fault_power_lower"]
            for r in rows
            if r["phase"] == "fault"
            and r["groups"] == max(cfg["effective_group_sizes"])
            and r["method"] == "FP3-dispersion"
        )
        and all(
            r["analysis"]["status"] == ("INFO" if g["guard"] == "changing_topk" else "INCONCLUSIVE")
            and all(c["verdict"] not in {"PASS", "FAIL"} for c in r["analysis"]["checks"])
            for g in guards
            for r in g["results"]
        )
    )
    write_json(out / "robustness.json", output)
    return output
