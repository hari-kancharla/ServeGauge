"""Controlled fault study: inject outside scorers, decide from captured wire only."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import yaml

from servegauge.core.hashing import content_hash
from servegauge.core.models import Item
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.refs.pack import build_pack
from servegauge.runner.evaluate import collect, evaluate, reference_source
from servegauge.runner.plan import ReferenceSpec, RunPlan
from servegauge.scoring.agents import agent_samples
from servegauge.scoring.common import Samples
from servegauge.scoring.operational import cache_samples, operational_samples
from servegauge.suites.s1_protocol import probe, protocol_items
from servegauge.suites.s2_tools import tool_items
from servegauge.suites.s4_longctx import context_items
from servegauge.suites.s5_longout import reasoning_items
from servegauge.suites.s7_accounting import accounting_items
from servegauge.validation.capture import captures, endpoint
from servegauge.validation.common import (
    ROOT,
    cached,
    named_seed,
    offline_only,
    proportion,
    receipt,
    settings,
    source_hash,
    write_json,
)
from servegauge.verdicts.config import load_policy

# The labels below belong to the independent acceptance oracle, never the scorer.
CASES = {
    "F-TEMP": ("ignore_temperature", "P03", "PARAM_IGNORED"),
    "F-TOPP": ("ignore_top_p", "P04.top_p", "PARAM_IGNORED"),
    "F-MAXTOK": ("ignore_max_tokens", "P01", "PARAM_IGNORED"),
    "F-STOP": ("ignore_stop", "P02", "PARAM_IGNORED"),
    "F-EFFORT": ("ignore_reasoning_effort", "P07", "PARAM_IGNORED"),
    "F-TOOLDROP": ("tool_drop", "T06", "PARSER_DROP"),
    "F-LEAK": ("leak_markers", "T05", "PARSER_LEAK"),
    "F-NOINDEX": ("noindex", "T04.parallel.stream", "PARSER_DROP"),
    "F-ARGJSON": ("corrupt_arguments", "T02", "GUIDED_DECODING_ABSENT"),
    "F-TEMPLATE": ("template", "T07.passback", "TEMPLATE_MISMATCH"),
    "F-HEADTRUNC": ("headtrunc", "L04", "CONTEXT_TRUNCATION"),
    "F-OUTTRUNC": ("outtrunc", "R03.8000", "LONG_OUTPUT_FAULT"),
    "F-USAGE": ("usage", "B01", "ACCOUNTING_MISMATCH"),
    "F-CACHE": ("cache", "P03.cache", "CACHE_ANOMALY"),
    "F-ERR": ("errors", "S8.success", "INFRA_UNRELIABLE"),
    "F-SWAP": ("swap", "FP1", "DIFFERENT_MODEL_LIKE"),
}


def clean_pass_rate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Irrelevant checks have no PASS gate; unexpected missing support still counts."""
    if results[0]["check_class"] != "comparative" or all(
        r["verdict"] == "NOT_APPLICABLE" for r in results
    ):
        return None
    return proportion(sum(r["verdict"] == "PASS" for r in results), len(results))


def isolated_replicate(check: str, faults: FaultConfig, seed: int) -> dict[str, Any]:
    with offline_only():
        return run_replicate(check, faults, seed)


@lru_cache(maxsize=32)
def inventory(check: str) -> tuple[Item, ...]:
    seed = settings()["seed"]
    if check in {"P01", "P02", "P03", "P04.top_p", "P07"}:
        return tuple(i for i in protocol_items("core", seed) if i.expected["check_id"] == check)
    if check.startswith(("T04", "T05", "T06", "T02")):
        return tuple(i for i in tool_items("core", seed) if "agent_goal" not in i.expected)
    if check.startswith("T07"):
        return tuple(
            i for i in tool_items("core", seed) if i.expected["check_id"] == "T07.passback"
        )
    if check == "B01":
        return tuple(accounting_items("core", seed))
    if check == "L04":
        return tuple(
            i
            for i in context_items(
                "core", seed, [endpoint("mock:design", seed, clean=True)], {"lengths": [32000]}
            )
            if i.expected["check_id"] == "context"
        )
    if check.startswith("R03"):
        return tuple(
            i for i in reasoning_items("core", seed) if i.expected["check_id"].startswith("R03")
        )
    if check == "FP1":
        from servegauge.validation.fingerprints import fingerprint_items

        return tuple(fingerprint_items())
    if check in {"P03.cache", "S8.success"}:
        return tuple(
            probe(
                check,
                index,
                "repeats",
                f"Choose a color, one word. Independent prompt {seed}-{index}.",
                {"type": "sample", "choices": ["red", "blue", "green", "gold", "orange", "violet"]},
                params={"max_tokens": 128, "temperature": 1.0},
                samples=20 if check == "P03.cache" else 1,
            )
            for index in range(400 if check == "P03.cache" else 2000)
        )
    raise ValueError("unknown_validation_inventory")


def run_replicate(check: str, faults: FaultConfig, seed: int) -> dict[str, Any]:
    items = list(inventory(check))
    run_id = "validation-" + str(seed)
    policy, profile = load_policy(), load_profile("gpt-oss-120b")
    ends = [
        endpoint("mock:target", seed, clean=False),
        *(
            endpoint(f"mock:reference-{i}", named_seed(seed, "reference", i), clean=True)
            for i in range(3)
        ),
    ]
    suites = sorted({i.suite_id for i in items})
    plan = RunPlan(
        plan_id="offline-validation",
        profile=profile.id,
        level="core",
        seed=settings()["seed"],
        suites=suites,
        endpoints=ends,
        max_usd=0,
        reference=ReferenceSpec(tier="MOCK"),
    )
    samples: dict[str, dict[str, Samples]] = {}
    records = []
    target_backend = ScriptedBackend(seed, faults.model_copy(update={"seed": seed}))
    by_endpoint = {}
    for index, target in enumerate(ends):
        backend = (
            target_backend
            if index == 0
            else ScriptedBackend(int(target.metadata["mock_declaration"]["seed"]), FaultConfig())
        )
        rows = captures(items, backend, target, run_id)
        records.extend(rows)
        by_endpoint[target.id] = rows
        if check == "FP1":
            continue
        scores = collect(
            plan.model_copy(update={"endpoints": [target]}), items, rows, profile, policy
        )[target.id]
        if check.startswith("T07"):
            scores.update(agent_samples(items, rows, target.id))
        if check == "P03.cache":
            scores[check] = cache_samples(items, rows, target.id)
        if check == "S8.success":
            scores[check] = operational_samples(rows, target.id)
        samples[target.id] = scores
    if check == "FP1":
        from servegauge.validation.fingerprints import fingerprint_checks

        checks = fingerprint_checks(items, by_endpoint, seed)
    else:
        source = reference_source(
            plan,
            run_id,
            items,
            samples,
            policy,
            {
                "item_set_hashes": {
                    "validation": content_hash([i.model_dump(mode="json") for i in items])
                }
            },
        )
        pack = build_pack(source, "MOCK", seed=seed) if source.checks else None
        confirmed: dict[tuple[str, str], bool] = {}
        for name, sample in samples[ends[0].id].items():
            if (
                sample.check_class != "invariant"
                or sample.status
                or sum(sample.values.values()) < policy.k_fail
                or not sample.replay_candidates
            ):
                continue
            probes = [i for i in items if i.item_id in sample.replay_candidates[0]]
            for start in range(0, settings()["replay_limit"], settings()["replay_batch"]):
                repeated = []
                for j in range(start, start + settings()["replay_batch"]):
                    for original in probes:
                        item = original.model_copy(deep=True)
                        item.item_id += f"-confirmation-{j}"
                        item.expected["trial"] = (
                            str(item.expected.get("trial")) + f"-confirmation-{j}"
                        )
                        repeated.append(item)
                replayed = captures(
                    repeated, target_backend, ends[0], run_id + f"-confirmation-{name}-{start}"
                )
                records.extend(replayed)
                found = collect(
                    plan.model_copy(update={"endpoints": [ends[0]]}),
                    repeated,
                    replayed,
                    profile,
                    policy,
                )[ends[0].id][name]
                if found.status is None and any(found.values.values()):
                    confirmed[(ends[0].id, name)] = True
                    break
        checks = evaluate(
            plan,
            run_id,
            items,
            samples,
            policy,
            target_ids=[ends[0].id],
            reference_pack=pack,
            replay_confirmed=confirmed,
        )
    if check == "S8.success":
        for result in checks:
            if result.check_id == check:
                result.details["operational_alarm"] = result.verdict == "FAIL"
                result.verdict = "INFO"
                result.check_class = "report_only"
                result.reason = "operational diagnostic; calibrated comparison retained in details"
    found_diagnoses = diagnoses(checks)
    retained = {
        "replay_confirmed",
        "failure_signatures",
        "fingerprint_position",
        "fingerprint_bands",
        "operational_alarm",
        "declared_margin",
        "noise_band",
        "one_sided_bounds",
        "reference_interval",
        "bootstrap_resamples",
        "worst_case_holm_alpha",
        "metric",
        "independent_unit",
    }
    return {
        "seed": seed,
        "requests": len(records),
        "records_hash": content_hash([r.model_dump(mode="json") for r in records]),
        "checks": [
            {
                **r.model_dump(mode="json"),
                "evidence": [content_hash(r.evidence)],
                "details": {k: v for k, v in r.details.items() if k in retained},
            }
            for r in checks
        ],
        "diagnoses": [
            {**d.model_dump(mode="json"), "evidence": [content_hash(d.evidence)]}
            for d in found_diagnoses
        ],
    }


def fault_matrix(
    *, replicates: int | None = None, out: Path = Path("docs/report")
) -> dict[str, Any]:
    cfg = settings()
    n_fault = cfg["fault_replicates"] if replicates is None else replicates
    if n_fault < 30:
        raise ValueError("fault_validation_requires_at_least_30_replicates")
    n_clean = max(cfg["clean_replicates"], n_fault)
    output = receipt(
        "faults",
        {**cfg, "executed_fault_replicates": n_fault, "executed_clean_replicates": n_clean},
    )
    output["cases"] = {}
    identity = content_hash([source_hash(), output["configuration_hash"]])[:24]
    clean_cache: dict[tuple[str, int], dict[str, Any]] = {}
    spend_before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    with (
        offline_only(),
        ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=get_context("spawn")) as workers,
    ):
        for fault_id, (filename, check, code) in CASES.items():
            data = yaml.safe_load(
                (ROOT / f"configs/faults/{filename}.yaml").read_text(encoding="utf-8")
            )
            faults = FaultConfig.model_validate(data)
            rows: dict[str, list[dict[str, Any]]] = {"clean": [], "fault": []}
            for mode, n in (("clean", n_clean), ("fault", n_fault)):
                design = "S2" if check.startswith(("T04", "T05", "T06", "T02")) else check
                jobs = []
                for index in range(n):
                    # Different clean and fault seeds; fault labels cannot select a seed.
                    seed = named_seed(cfg["seed"], design, mode, index)
                    path = (
                        ROOT
                        / ".cache/validation"
                        / identity
                        / design
                        / mode
                        / f"{filename if mode == 'fault' else 'clean'}-{index}.json"
                    )
                    row = cached(path)
                    future = (
                        None
                        if row is not None
                        else workers.submit(
                            isolated_replicate,
                            check,
                            FaultConfig() if mode == "clean" else faults,
                            seed,
                        )
                    )
                    jobs.append((seed, path, row, future))
                for index, (seed, path, row, future) in enumerate(jobs):
                    if row is None:
                        assert future is not None
                        row = future.result()
                        write_json(path, row)
                    rows[mode].append(row)
                    if mode == "clean":
                        clean_cache[(check, seed)] = row
                    if (index + 1) % 10 == 0:
                        print(f"{fault_id} {mode} {index + 1}/{n}", flush=True)
            clean_by_check: dict[str, list[dict[str, Any]]] = {}
            for row in rows["clean"]:
                for result in row["checks"]:
                    clean_by_check.setdefault(result["check_id"], []).append(result)
            clean_rates: dict[str, Any] = {
                name: {
                    "false_positive_rate": proportion(
                        sum(
                            r["verdict"] == "FAIL" or r["details"].get("operational_alarm", False)
                            for r in results
                        ),
                        len(results),
                    ),
                    "comparative_pass_rate": clean_pass_rate(results),
                }
                for name, results in clean_by_check.items()
            }
            detected, diagnosed = 0, 0
            for row in rows["fault"]:
                detection = any(
                    r["check_id"] == check
                    and (r["verdict"] == "FAIL" or r["details"].get("operational_alarm", False))
                    for r in row["checks"]
                )
                detected += detection
                diagnosed += detection and any(d["code"] == code for d in row["diagnoses"])
            acceptance = (
                detected / n_fault >= cfg["minimum_power"]
                and diagnosed / n_fault >= cfg["minimum_power"]
                and all(
                    r["false_positive_rate"]["estimate"] <= cfg["maximum_false_positive_rate"]
                    and (
                        r["comparative_pass_rate"] is None
                        or r["comparative_pass_rate"]["estimate"]
                        >= cfg["minimum_clean_comparative_pass_rate"]
                    )
                    for r in clean_rates.values()
                )
            )
            output["cases"][fault_id] = {
                "fault_configuration": faults.model_dump(mode="json"),
                "expected_check": check,
                "expected_code": code,
                "power": proportion(detected, n_fault),
                "correct_diagnosis_power": proportion(diagnosed, n_fault),
                "diagnosis_sets": dict(
                    Counter(
                        "+".join(sorted(d["code"] for d in row["diagnoses"])) or "none"
                        for row in rows["fault"]
                    )
                ),
                "fault_power_by_check": {
                    name: proportion(
                        sum(
                            r["check_id"] == name
                            and (
                                r["verdict"] == "FAIL"
                                or r["details"].get("operational_alarm", False)
                            )
                            for row in rows["fault"]
                            for r in row["checks"]
                        ),
                        n_fault,
                    )
                    for name in clean_by_check
                },
                "clean": clean_rates,
                "accepted": acceptance,
                "replicates": rows,
            }
            output["accepted"] = len(output["cases"]) == len(CASES) and all(
                c["accepted"] for c in output["cases"].values()
            )
            write_json(out / "faults.json", output)
            print(
                f"{fault_id}: detected {detected}/{n_fault}; "
                f"correct diagnosis {diagnosed}/{n_fault}; acceptance={acceptance}",
                flush=True,
            )
    output["spend_hash_before"] = spend_before
    output["spend_hash_after"] = content_hash(
        (ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")
    )
    if output["spend_hash_after"] != spend_before:
        raise RuntimeError("offline_validation_must_not_change_spending")
    write_json(out / "faults.json", output)
    from servegauge.validation.reporting import fault_report

    fault_report(output, out)
    return output
