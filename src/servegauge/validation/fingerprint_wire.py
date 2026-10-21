"""Held-out captured-scripted S6 power, conditional on independent reference packs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from servegauge.core.hashing import content_hash
from servegauge.core.models import CheckResult, Item, ReferencePack
from servegauge.mock.faults import FaultConfig
from servegauge.refs.pack import ReferenceRun, build_pack
from servegauge.scoring.fingerprint_results import fingerprint_results
from servegauge.stats.multiple import holm
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.common import (
    ROOT,
    named_seed,
    offline_only,
    proportion,
    receipt,
    write_json,
)
from servegauge.validation.fingerprint_product import configuration
from servegauge.validation.fingerprint_wire_capture import capture_run, outcome_signals
from servegauge.verdicts.config import DecisionPolicy, load_policy
from servegauge.verdicts.engine import decide_family

CASES = ("clean", "swap", "mix", "qsim")
SIGNALS = {
    "clean": "text_equivalence_candidate",
    "swap": "swap_signature",
    "mix": "mixture_signature",
    "qsim": "noise_signature",
}


def wire_configuration() -> dict[str, Any]:
    cfg = yaml.safe_load(
        (ROOT / "configs/fingerprint_wire_validation.yaml").read_text(encoding="utf-8")
    )
    base = configuration()
    if (
        not isinstance(cfg, dict)
        or cfg.get("version") != "fingerprint-wire-v1"
        or type(cfg.get("reference_packs")) is not int
        or cfg["reference_packs"] < 3
        or type(cfg.get("workers")) is not int
        or cfg["workers"] < 1
        or set(cfg.get("replicates", {})) != set(CASES)
        or any(
            type(cfg["replicates"][c]) is not int
            or cfg["replicates"][c]
            < base["clean_replicates" if c == "clean" else "fault_replicates"]
            for c in CASES
        )
        or not base["minimum_clean_pass_rate"] <= cfg.get("minimum_clean_equivalence_lower", 0) < 1
        or not base["minimum_fault_power_lower"] <= cfg.get("minimum_fault_power_lower", 0) < 1
    ):
        raise ValueError("predeclared_wire_design_without_weakened_gates_required")
    return cfg


def _local_file(name: str) -> Path:
    path = Path(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(p.lower() == "starter files" for p in path.parts)
    ):
        raise ValueError("local_qualification_evidence_path_required")
    return ROOT / path


def prerequisites(
    base: dict[str, Any], cfg: dict[str, Any], policy: DecisionPolicy
) -> dict[str, Any]:
    try:
        matched_runtime = all(
            version(package) == expected for package, expected in cfg["runtime"].items()
        )
    except PackageNotFoundError:
        matched_runtime = False
    if not matched_runtime:
        raise ValueError("qualified_numerical_runtime_required")
    evidence = {}
    for name, spec in cfg["evidence"].items():
        data = _local_file(spec["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("intact_qualification_prerequisites_required")
        evidence[name] = json.loads(data)
    for name in ("interval_result", "null_result"):
        payload = dict(evidence[name])
        if (
            payload.pop("result_hash", None) != content_hash(payload)
            or payload.get("accepted") is not True
        ):
            raise ValueError("completed_accepted_qualification_prerequisites_required")
    if evidence["interval_result"]["configuration"] != base or evidence["null_result"][
        "configuration"
    ] != {
        "design_configuration": base,
        "policy": policy.model_dump(mode="json"),
        "process": "heterogeneous-categorical-and-gaussian-logit-iid-v1",
    }:
        raise ValueError("matched_design_and_policy_qualification_required")
    interval = evidence["interval_acceptance"]
    if (
        interval.get("accepted") is not True
        or evidence["null_acceptance"].get("accepted") is not True
    ):
        raise ValueError("independently_accepted_prerequisites_required")
    bindings = {**interval["study_code_sha256"], **cfg["null_source_files"]}
    if any(
        hashlib.sha256(_local_file(p).read_bytes()).hexdigest() != digest
        for p, digest in bindings.items()
    ):
        raise ValueError("unchanged_qualified_statistical_primitives_required")
    return {
        "interval_accepted": True,
        "null_accepted": True,
        "primitive_files_verified": len(interval["study_code_sha256"]),
        "evidence_hashes": {name: spec["sha256"] for name, spec in cfg["evidence"].items()},
        "code_bindings": bindings,
    }


def wire_items(base: dict[str, Any], cfg: dict[str, Any]) -> list[Item]:
    return fingerprint_items("core", named_seed(cfg["seed"], "items"), base["design"])


def _run_id(seed: int) -> str:
    return "wire-" + content_hash([seed, "run"])[:24]


def prepare_pack(
    items: list[Item], base: dict[str, Any], cfg: dict[str, Any], policy: DecisionPolicy, index: int
) -> dict[str, Any]:
    source, receipts = capture_run(
        items,
        policy,
        seeds=[named_seed(cfg["seed"], "reference", index, i) for i in range(3)],
        source_id=_run_id(named_seed(cfg["seed"], "reference-run", index)),
        healthy_jitter=base["healthy_logit_jitter"],
        reference=True,
    )
    controls = []
    for name in ("control-a", "control-b"):
        seed = named_seed(cfg["seed"], "control", index, name)
        control, captured = capture_run(
            items,
            policy,
            seeds=[seed],
            source_id=_run_id(seed),
            healthy_jitter=base["healthy_logit_jitter"],
            control=name,
        )
        controls.append(control)
        receipts.extend(captured)
    pack = build_pack(
        source,
        "MOCK",
        seed=named_seed(cfg["seed"], "pack", index),
        resamples=policy.bootstrap_resamples,
        controls=controls,
    )
    value = pack.model_dump(mode="json")
    return {
        "pack_index": index,
        "pack": value,
        "pack_hash": content_hash(value),
        "sources": [r.model_dump(mode="json") for r in [source, *controls]],
        "capture_receipts": receipts,
    }


def _fault(case: str, base: dict[str, Any]) -> FaultConfig:
    if case == "clean":
        return FaultConfig()
    if case == "swap":
        return FaultConfig(route_to_other_model=1)
    if case == "mix":
        return FaultConfig(route_to_other_model=base["route_mix"])
    if case == "qsim":
        return FaultConfig(logit_noise=base["fault_logit_noise"])
    raise ValueError("predeclared_wire_case_required")


def _evidence_n(base: dict[str, Any]) -> dict[str, int]:
    d = base["design"]
    return {
        "FP1": d["fp1_cells"] * d["fp1_samples"],
        "FP2": d["fp2_cells"] * d["fp2_samples"],
        "FP3": d["fp3_samples"],
    }


def measure_target(
    items: list[Item],
    base: dict[str, Any],
    cfg: dict[str, Any],
    policy: DecisionPolicy,
    prepared: dict[str, Any],
    pack_index: int,
    case: str,
    index: int,
) -> dict[str, Any]:
    seed = named_seed(cfg["seed"], "target", pack_index, case, index)
    source, captured = capture_run(
        items,
        policy,
        seeds=[seed],
        source_id=_run_id(seed),
        healthy_jitter=base["healthy_logit_jitter"],
        fault=_fault(case, base),
    )
    endpoint_id = source.endpoints[0].id
    pack = ReferencePack.model_validate(prepared["pack"])
    before = fingerprint_results(
        source.fingerprints[endpoint_id],
        endpoint_id=endpoint_id,
        run_id=source.run_id,
        pack=pack,
        policy=policy,
        seed=named_seed(seed, "inference"),
    )
    checks = decide_family([], policy, external=before)
    compact = [
        {
            "check": c.model_dump(mode="json", exclude={"evidence"}),
            "evidence_count": len(c.evidence),
            "evidence_hash": content_hash(c.evidence),
        }
        for c in checks
    ]
    return {
        "pack_index": pack_index,
        "case": case,
        "index": index,
        "seed": seed,
        "pack_hash": prepared["pack_hash"],
        "captures": captured[0]["captures"],
        "capture_hash": captured[0]["record_hash"],
        "observations": captured[0]["observation_hashes"],
        "checks": compact,
        "signals": outcome_signals(checks, policy),
    }


def checked_trial(
    row: dict[str, Any],
    base: dict[str, Any],
    cfg: dict[str, Any],
    policy: DecisionPolicy,
    prepared: dict[str, Any],
    pack_index: int,
    case: str,
    index: int,
) -> dict[str, Any]:
    n = _evidence_n(base)
    seed = named_seed(cfg["seed"], "target", pack_index, case, index)
    if (
        row.get("pack_index") != pack_index
        or row.get("case") != case
        or row.get("index") != index
        or row.get("seed") != seed
        or row.get("pack_hash") != prepared["pack_hash"]
        or row.get("captures") != sum(n.values())
        or len(row.get("capture_hash", "")) != 64
    ):
        raise ValueError("matched_seeded_wire_trial_required")
    if set(row.get("observations", {})) != set(n) or any(
        not isinstance(v, str) or len(v) != 64 for v in row["observations"].values()
    ):
        raise ValueError("complete_wire_observation_hashes_required")
    checks = []
    for value in row["checks"]:
        check = CheckResult.model_validate(value["check"])
        if (
            value["evidence_count"] != n.get(check.check_id)
            or len(value["evidence_hash"]) != 64
            or check.reference_tier not in {None, "MOCK"}
            or check.run_id != _run_id(seed)
            or check.endpoint_id != "mock:wire-" + content_hash([seed, "endpoint"])[:20]
        ):
            raise ValueError("complete_wire_evidence_counts_required")
        if check.details.get("fingerprint_metrics") and check.n != n[check.check_id]:
            raise ValueError("matched_wire_component_n_required")
        checks.append(check)
    if (
        {c.check_id for c in checks} != set(n)
        or len(checks) != len(n)
        or row["signals"] != outcome_signals(checks, policy)
    ):
        raise ValueError("consistent_blind_wire_signals_required")
    expected = holm([1.0 if c.p_value is None else c.p_value for c in checks])
    if any(
        c.p_value is not None and c.adjusted_p != p for c, p in zip(checks, expected, strict=True)
    ):
        raise ValueError("consistent_wire_family_correction_required")
    return row


def _save(path: Path, header: dict[str, Any], value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    write_json(
        temporary,
        {
            "source_hash": header["source_hash"],
            "configuration_hash": header["configuration_hash"],
            "value": value,
            "value_hash": content_hash(value),
        },
    )
    temporary.replace(path)


def _read(path: Path, header: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(data.get(k) != header[k] for k in ("source_hash", "configuration_hash")) or data.get(
        "value_hash"
    ) != content_hash(data["value"]):
        raise ValueError("intact_wire_checkpoint_on_original_source_required")
    return dict(data["value"])


def _calibration(
    out: Path,
    header: dict[str, Any],
    items: list[Item],
    base: dict[str, Any],
    cfg: dict[str, Any],
    policy: DecisionPolicy,
    index: int,
) -> dict[str, Any]:
    path = out / "calibration" / f"pack-{index:02d}.json"
    value = _read(path, header)
    if value is None:
        value = prepare_pack(items, base, cfg, policy, index)
        _save(path, header, value)
    else:
        sources = [ReferenceRun.model_validate(v) for v in value["sources"]]
        rebuilt = build_pack(
            sources[0],
            "MOCK",
            seed=named_seed(cfg["seed"], "pack", index),
            resamples=policy.bootstrap_resamples,
            controls=sources[1:],
        )
        if rebuilt.model_dump(mode="json") != value["pack"]:
            raise ValueError("raw_reference_rebuild_must_match_saved_wire_pack")
    if value["pack_index"] != index or value["pack_hash"] != content_hash(value["pack"]):
        raise ValueError("matched_immutable_wire_pack_required")
    return value


def _summaries(
    rows: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    cfg: dict[str, Any],
    policy: DecisionPolicy,
) -> list[dict[str, Any]]:
    result = []
    for index, prepared in enumerate(packs):
        metrics = prepared["pack"]["fingerprints"]["views"]["*"]["metrics"]
        qualified = all(v["bands_separated"] for v in metrics.values()) and all(
            metrics[m]["noise_band"] <= policy.margin(m) for m in ("FP1", "FP2")
        )
        cases = {}
        for case in CASES:
            selected = [r for r in rows if r["pack_index"] == index and r["case"] == case]
            if len(selected) != cfg["replicates"][case]:
                raise ValueError("complete_per_pack_wire_trial_bank_required")
            rate = proportion(sum(r["signals"][SIGNALS[case]] for r in selected), len(selected))
            minimum = cfg[
                "minimum_clean_equivalence_lower"
                if case == "clean"
                else "minimum_fault_power_lower"
            ]
            counts = {
                m: dict(
                    Counter(
                        v["check"]["verdict"]
                        for r in selected
                        for v in r["checks"]
                        if v["check"]["check_id"] == m
                    )
                )
                for m in ("FP1", "FP2", "FP3")
            }
            cases[case] = {
                "success": rate,
                "criterion": SIGNALS[case],
                "accepted": rate["ci_low"] >= minimum,
                "actual_verdict_counts": counts,
                "signature_rates": {
                    signal: proportion(sum(r["signals"][signal] for r in selected), len(selected))
                    for signal in selected[0]["signals"]
                },
            }
        result.append(
            {
                "pack_index": index,
                "pack_hash": prepared["pack_hash"],
                "calibration_eligible": qualified,
                "cases": cases,
                "accepted": qualified and all(c["accepted"] for c in cases.values()),
            }
        )
    return result


def wire_study(out: Path) -> dict[str, Any]:
    base, cfg, policy = configuration(), wire_configuration(), load_policy()
    proof = prerequisites(base, cfg, policy)
    header = receipt(
        "fingerprint-wire-power",
        {
            "design": base,
            "wire": cfg,
            "policy": policy.model_dump(mode="json"),
            "prerequisites": proof,
        },
    )
    items = wire_items(base, cfg)
    jobs = [
        (p, c, i)
        for p in range(cfg["reference_packs"])
        for c in CASES
        for i in range(cfg["replicates"][c])
    ]
    target_seeds = [named_seed(cfg["seed"], "target", *job) for job in jobs]
    reference_seeds = [
        named_seed(cfg["seed"], "reference", p, i)
        for p in range(cfg["reference_packs"])
        for i in range(3)
    ]
    control_seeds = [
        named_seed(cfg["seed"], "control", p, c)
        for p in range(cfg["reference_packs"])
        for c in ("control-a", "control-b")
    ]
    if len(set([*target_seeds, *reference_seeds, *control_seeds])) != len(target_seeds) + len(
        reference_seeds
    ) + len(control_seeds):
        raise ValueError("disjoint_wire_roles_and_target_seeds_required")
    path = out / "wire.json"
    saved = _read(path, header)
    if saved is not None:
        packs = [
            _calibration(out, header, items, base, cfg, policy, i)
            for i in range(cfg["reference_packs"])
        ]
        if len(saved["trials"]) != len(jobs):
            raise ValueError("complete_wire_trial_bank_required")
        for row, (p, c, i) in zip(saved["trials"], jobs, strict=True):
            checked_trial(row, base, cfg, policy, packs[p], p, c, i)
        summaries = _summaries(saved["trials"], packs, cfg, policy)
        if saved["per_pack"] != summaries or saved["accepted"] != all(
            s["accepted"] for s in summaries
        ):
            raise ValueError("consistent_wire_summary_required")
        return saved
    rows = []
    with offline_only():
        packs = []
        for index in range(cfg["reference_packs"]):
            packs.append(_calibration(out, header, items, base, cfg, policy, index))
            print(
                f"captured wire calibration saved: {index + 1}/{cfg['reference_packs']}", flush=True
            )

        def trial(job: tuple[int, str, int]) -> dict[str, Any]:
            p, c, i = job
            path = out / "trials" / f"pack-{p:02d}-{c}-{i:04d}.json"
            row = _read(path, header)
            if row is None:
                row = measure_target(items, base, cfg, policy, packs[p], p, c, i)
                checked_trial(row, base, cfg, policy, packs[p], p, c, i)
                _save(path, header, row)
            return checked_trial(row, base, cfg, policy, packs[p], p, c, i)

        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            for row in pool.map(trial, jobs):
                rows.append(row)
                print(f"captured wire targets saved: {len(rows)}/{len(jobs)}", flush=True)
    summaries = _summaries(rows, packs, cfg, policy)
    result = {
        **header,
        "per_pack": summaries,
        "trials": rows,
        "accepted": all(s["accepted"] for s in summaries),
        "captured_responses": sum(r["captures"] for r in rows)
        + sum(r["captures"] for p in packs for r in p["capture_receipts"]),
        "scope": "Captured scripted mock responses, conditional on each of three reference packs; "
        "prospective text equivalence only. Not provider, physical quantization "
        "or production power.",
    }
    _save(path, header, result)
    return result
