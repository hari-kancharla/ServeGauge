"""Independent clean/F-ERR replicates for captured whole-procedure health alarms."""

from __future__ import annotations

import copy
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError

from servegauge.core.completion import completion_error
from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, Exchange, Item
from servegauge.diagnose.rules import diagnoses
from servegauge.items.agent_tasks import execute_tool
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.refs.pack import build_pack
from servegauge.runner.evaluate import collect, evaluate, reference_source
from servegauge.runner.plan import ReferenceSpec, RunPlan
from servegauge.scoring.common import calls, choice, message
from servegauge.scoring.operational_health import health_samples
from servegauge.scoring.toolcall_scoring import parsed_call
from servegauge.suites.registry import prepare_item
from servegauge.suites.s8_operations import HEALTH_CHECKS, operational_items
from servegauge.validation.capture import capture, endpoint
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


def procedure_captures(
    items: list[Item], backend: ScriptedBackend, target: EndpointSpec, run_id: str
) -> list[Exchange]:
    """Mock-only framing with full attempts and the runner's bounded task semantics.

    Timings are unmeasured. This helper is used for health outcomes, not latency.
    """
    declared_seed = target.metadata.get("mock_declaration", {}).get("seed")
    if target.kind != "mock" or type(declared_seed) is not int or backend.seed != declared_seed:
        raise ValueError("health_lab_backend_seed_mismatch")
    output = []
    for item in items:
        draws = int(item.expected.get("samples", 1))
        agent = "agent_goal" in item.expected
        if agent and draws != 1:
            raise ValueError("one_initial_capture_per_agent_required")
        for sample in range(draws):
            history = copy.deepcopy(item.payload["messages"])
            state = copy.deepcopy(item.expected.get("agent_initial", {}))
            for turn in range(int(item.expected.get("max_turns", 1))):
                child = item.model_copy(deep=True)
                child.payload["messages"] = copy.deepcopy(history)
                identifier = run_id
                if turn:
                    child.item_id += f"-turn-{turn}"
                    identifier += "-agent-" + content_hash(target.id)[:8] + f"-{turn}"
                decision = prepare_item(child, target)
                if decision.body is None:
                    raise ValueError("health_lab_procedure_unexpectedly_unsupported")
                record = capture(child, decision.body, backend, target, identifier, sample)
                record.attempts[0].response = copy.deepcopy(record.response)
                record.attempts[0].metadata = copy.deepcopy(record.metadata)
                record.attempts[0].sse_events = copy.deepcopy(record.sse_events)
                if record.status != 200:
                    record.error = f"http_status_{record.status}"
                elif completion_error(record.response):
                    record.error = record.attempts[0].error = "provider_completion_error"
                output.append(record)
                if not agent or record.error:
                    break
                try:
                    if choice(record).get("finish_reason") == "length":
                        break
                    returned = calls(record)
                    if not returned or len(returned) > item.expected["max_turns"]:
                        break
                    assistant = copy.deepcopy(message(record))
                    if not item.expected["pass_reasoning"]:
                        for key in ("reasoning", "reasoning_details", "reasoning_content"):
                            assistant.pop(key, None)
                    history.append(assistant)
                    for call in returned:
                        name, arguments = parsed_call(call)
                        if not isinstance(call.get("id"), str) or not call["id"]:
                            raise ValueError("missing_tool_call_id")
                        state = execute_tool(state, name, arguments)
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    {"ok": True, "state": state[name]}, ensure_ascii=False
                                ),
                            }
                        )
                    if state == item.expected["agent_goal"]:
                        break
                except (ValueError, KeyError, TypeError, ValidationError):
                    break
    return output


def run_health_replicate(seed: int, faults: FaultConfig) -> dict[str, Any]:
    items = operational_items("core", settings()["seed"])
    run_id = f"health-validation-{seed}"
    ends = [
        endpoint("mock:target", seed, clean=False),
        *(
            endpoint(f"mock:reference-{i}", named_seed(seed, "health-reference", i), clean=True)
            for i in range(settings()["reference_instances"])
        ),
    ]
    plan = RunPlan(
        plan_id="operational-health-validation",
        profile="gpt-oss-120b",
        level="core",
        seed=settings()["seed"],
        reference=ReferenceSpec(tier="MOCK"),
        endpoints=ends,
        suites=["S8"],
        max_usd=0,
        retries=0,
    )
    records = []
    backend_seeds = {}
    for index, target in enumerate(ends):
        actual_seed = int(target.metadata["mock_declaration"]["seed"])
        assigned = (faults if index == 0 else FaultConfig()).model_copy(
            update={"seed": actual_seed}
        )
        backend = ScriptedBackend(actual_seed, assigned)
        backend_seeds[target.id] = backend.seed
        records.extend(procedure_captures(items, backend, target, run_id))
    policy = load_policy()
    samples = collect(plan, items, records, load_profile(plan.profile), policy)
    for e in ends:
        samples[e.id].update(health_samples(plan, items, records, e.id, run_id))
    source = reference_source(
        plan,
        run_id,
        items,
        samples,
        policy,
        {"item_set_hashes": {"S8": content_hash([i.model_dump(mode="json") for i in items])}},
    )
    pack = build_pack(source, "MOCK", seed=plan.seed)
    results = evaluate(
        plan,
        run_id,
        items,
        samples,
        policy,
        target_ids=[ends[0].id],
        reference_pack=pack,
        operational_records=records,
    )
    return {
        "seed": seed,
        "backend_seeds": backend_seeds,
        "requests": len(records),
        "pack_id": pack.pack_id,
        "records_hash": content_hash([r.model_dump(mode="json") for r in records]),
        "checks": [
            {
                **{
                    k: c.model_dump(mode="json")[k]
                    for k in (
                        "check_id",
                        "n",
                        "estimate",
                        "ci_low",
                        "ci_high",
                        "reference_tier",
                        "reference_estimate",
                        "delta",
                        "delta_ci_low",
                        "delta_ci_high",
                        "p_value",
                        "adjusted_p",
                        "margin",
                        "verdict",
                    )
                },
                "comparison_verdict": c.details["comparison_verdict"],
                "operational_alarm": c.details["operational_alarm"],
                "evidence_count": len(c.evidence),
                "evidence_hash": content_hash(c.evidence),
            }
            for c in results
            if c.check_id in HEALTH_CHECKS
        ],
        "diagnoses": [
            {
                "code": d.code,
                "failure_class": d.failure_class,
                "confidence": d.confidence,
                "check_ids": d.check_ids,
                "evidence_count": len(d.evidence),
            }
            for d in diagnoses(results)
        ],
    }


def _isolated(seed: int, faults: FaultConfig) -> dict[str, Any]:
    with offline_only():
        return run_health_replicate(seed, faults)


def health_validation(out: Path) -> dict[str, Any]:
    cfg = settings()
    if (out / "health.json").exists():
        raise ValueError("health_study_output_exists_choose_fresh_directory")
    if (
        cfg["clean_replicates"] < 60
        or cfg["fault_replicates"] < 30
        or cfg["reference_instances"] < 3
    ):
        raise ValueError("health_validation_requires_SPEC_minimum_replicates_and_references")
    faults = FaultConfig.model_validate(
        yaml.safe_load((ROOT / "configs/faults/errors.yaml").read_text())
    )
    design = operational_items("core", cfg["seed"])
    configuration = {
        **{
            k: cfg[k]
            for k in (
                "seed",
                "clean_replicates",
                "fault_replicates",
                "reference_instances",
                "minimum_power",
                "maximum_false_positive_rate",
                "minimum_clean_comparative_pass_rate",
            )
        },
        "operations_contract": design[0].expected["operations_contract"],
        "health_contract": design[0].expected["operational_health_contract"],
        "policy": load_policy().model_dump(mode="json"),
        "fault": faults.model_dump(mode="json"),
        "seed_override_rule": "independent per-endpoint trial seeds override the template seed",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }
    result = receipt("operational-health", configuration)
    identity = content_hash([source_hash(), configuration])[:24]
    before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {"clean": [], "fault": []}
    seeds = [
        named_seed(cfg["seed"], "operational-health-v1", mode, index)
        for mode in rows
        for index in range(cfg[mode + "_replicates"])
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("independent_health_replicate_seeds_required")
    with (
        offline_only(),
        ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=get_context("spawn")) as pool,
    ):
        for mode in rows:
            jobs = []
            for index in range(cfg[mode + "_replicates"]):
                seed = named_seed(cfg["seed"], "operational-health-v1", mode, index)
                path = (
                    ROOT
                    / ".cache/validation/operational-health"
                    / identity
                    / mode
                    / f"{index:04}.json"
                )
                row = cached(path)
                future = (
                    None
                    if row is not None
                    else pool.submit(_isolated, seed, faults if mode == "fault" else FaultConfig())
                )
                jobs.append((path, row, future))
            for index, (path, row, future) in enumerate(jobs):
                if row is None:
                    assert future is not None
                    row = future.result()
                    write_json(path, row)
                rows[mode].append(row)
                if (index + 1) % 10 == 0:
                    print(f"operational health {mode}: {index + 1}/{len(jobs)}", flush=True)
    clean = {}
    for check in HEALTH_CHECKS:
        cards = [next(c for c in r["checks"] if c["check_id"] == check) for r in rows["clean"]]
        clean[check] = {
            "false_positive_rate": proportion(
                sum(c["operational_alarm"] for c in cards), len(cards)
            ),
            "comparison_pass_rate": proportion(
                sum(c["comparison_verdict"] == "PASS" for c in cards), len(cards)
            ),
        }
    detected = [any(c["operational_alarm"] for c in r["checks"]) for r in rows["fault"]]
    diagnosed = [
        hit
        and any(
            d["code"] == "INFRA_UNRELIABLE" and d["failure_class"] == "ROUTING_OR_INFRA"
            for d in row["diagnoses"]
        )
        for hit, row in zip(detected, rows["fault"], strict=True)
    ]
    runtime_seeds = [
        seed for group in rows.values() for row in group for seed in row["backend_seeds"].values()
    ]
    seed_audit = {
        "backend_instances": len(runtime_seeds),
        "distinct_backend_seeds": len(set(runtime_seeds)),
        "all_target_seeds_match_trials": all(
            row["backend_seeds"]["mock:target"] == row["seed"]
            for group in rows.values()
            for row in group
        ),
    }
    result.update(
        clean=clean,
        power=proportion(sum(detected), len(detected)),
        correct_diagnosis_power=proportion(sum(diagnosed), len(diagnosed)),
        replicates=rows,
        runtime_seed_audit=seed_audit,
        captured_requests=sum(r["requests"] for group in rows.values() for r in group),
        budget_hash_before=before,
        budget_hash_after=content_hash(
            (ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")
        ),
        qualifications=[
            "independent scripted roots; three clean captured MOCK references",
            "default core procedures, single unchanged F-ERR injection, retries disabled",
            "in-memory timings unmeasured; reserved ASGI/pack/replay acceptance is separate",
            "provider/R3/temporal/cache/compound-fault qualification remains open",
        ],
    )
    result["accepted"] = (
        result["power"]["estimate"] >= cfg["minimum_power"]
        and result["correct_diagnosis_power"]["estimate"] >= cfg["minimum_power"]
        and all(
            c["false_positive_rate"]["estimate"] <= cfg["maximum_false_positive_rate"]
            and c["comparison_pass_rate"]["estimate"] >= cfg["minimum_clean_comparative_pass_rate"]
            for c in clean.values()
        )
        and result["budget_hash_before"] == result["budget_hash_after"]
        and seed_audit["backend_instances"] == seed_audit["distinct_backend_seeds"]
        and seed_audit["all_target_seeds_match_trials"]
    )
    write_json(out / "health.json", result)
    lines = [
        "# Operational health validation",
        "",
        f"Accepted: {result['accepted']}.",
        "",
        "| Outcome | Count | Estimate (Wilson CI) | Reference |",
        "|---|---:|---:|---|",
    ]
    rates = {
        "F-ERR detection": result["power"],
        "F-ERR correct diagnosis": result["correct_diagnosis_power"],
    }
    for check, values in clean.items():
        rates.update(
            {
                check + " clean false alarm": values["false_positive_rate"],
                check + " clean comparison PASS": values["comparison_pass_rate"],
            }
        )
    for name, rate in rates.items():
        lines.append(
            f"| {name} | {rate['successes']}/{rate['n']} | "
            f"{rate['estimate']:.4f} [{rate['ci_low']:.4f}, {rate['ci_high']:.4f}] | MOCK |"
        )
    lines.extend(
        [
            "",
            "Paired check intervals/p-values, seeds and capture hashes: health.json.",
            "",
            *("- " + s for s in result["qualifications"]),
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines))
    return result
