"""Requirements check: capability preflight, paired primary traffic and budgeted real replay."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from servegauge.core.errors import SafeError
from servegauge.core.ids import safe_filename
from servegauge.core.models import CheckResult, Item, ReferencePack
from servegauge.core.redaction import Redactor
from servegauge.diagnose.drift import DRIFT_CONTRACT, annotate_context
from servegauge.diagnose.replay_expectations import mark_reproduced
from servegauge.profiles import load_profile
from servegauge.runner.agents import run_agents
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.evaluate import collect, evaluate, exit_code, reference_source
from servegauge.runner.plan import Requirements, RunPlan
from servegauge.runner.scheduler import _write_json, run_plan
from servegauge.scoring.common import score_capabilities
from servegauge.scoring.routing import ROUTING_CHECK, ROUTING_CONTRACT, routing_samples
from servegauge.scoring.tools import history_format_sample
from servegauge.storage.analysis import seal_analysis
from servegauge.storage.check_report import write_check_report
from servegauge.storage.raw_store import RawStore
from servegauge.suites.config import suite_config
from servegauge.suites.registry import item_checks
from servegauge.suites.s0_capabilities import capability_items
from servegauge.suites.s1_cache import CACHE_CHECKS, cache_items
from servegauge.suites.s1_protocol import protocol_items
from servegauge.suites.s2_tools import tool_items
from servegauge.suites.s3_structured import structured_items
from servegauge.suites.s4_longctx import context_items
from servegauge.suites.s5_longout import reasoning_items
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.suites.s7_accounting import accounting_items
from servegauge.suites.s8_operations import HEALTH_CHECKS, OPERATIONAL_CHECKS, operational_items
from servegauge.verdicts.classes import check_class
from servegauge.verdicts.config import load_policy

if TYPE_CHECKING:
    from servegauge.runner.captured import CapturedCheck


@dataclass(frozen=True)
class CheckRun:
    run_id: str
    code: int
    checks: list[CheckResult]
    estimate: dict[str, Any]
    report_dir: Path | None


def selected_check(check: str, selected: set[str]) -> bool:
    return any(
        token == "all"
        or (token == "S1" and check.startswith("P"))
        or (token == "S2" and check.startswith("T"))
        or (token == "S3" and check.startswith("J"))
        or (token == "S4" and check.startswith("L"))
        or (token == "S5" and (check.startswith("R") or check == "P07"))
        or (token == "S6" and check.startswith("FP"))
        or (token == "S7" and check.startswith("B"))
        or (token == "S8" and check.startswith("S8."))
        or check == token
        or check.startswith(token + ".")
        for token in selected
    )


def normalize_tests(tests: str | None) -> set[str]:
    aliases = {
        "protocol": "S1",
        "cache": "P03.cache",
        "tool-calling": "S2",
        "json": "S3",
        "context": "S4",
        "billing": "S7",
        "reasoning": "S5",
        "fingerprint": "S6",
        "operations": "S8",
        "all": "all",
    }
    tokens = {aliases.get(value.strip(), value.strip()) for value in (tests or "all").split(",")}
    allowed = {
        "all",
        "S0",
        "S8",
        *OPERATIONAL_CHECKS,
        *HEALTH_CHECKS,
        ROUTING_CHECK,
        *CACHE_CHECKS,
        "S1",
        "S2",
        "S3",
        "S4",
        "S7",
        "B01",
        "B02",
        "B03",
        "B03.prompt",
        "B03.completion",
        "B04",
        "B05",
        "S5",
        "S6",
        "FP1",
        "FP2",
        "FP3",
        "R01",
        "R03",
        "R04",
        "R05",
        "L01",
        "L02",
        "L03",
        "L04",
        "L05",
        "L06",
        "L06.fresh",
        "L06.warm",
        "J01",
        "J02",
        "J02.validity",
        "J02.semantic",
        "J03",
        "J04",
        "J05",
        *(f"T{i:02}" for i in range(1, 9)),
        "T07.passback",
        "T07.no_passback",
        *(
            f"T04.{probe}.{mode}"
            for probe in (
                "single",
                "parallel",
                "irrelevant",
                "nested",
                "unicode",
                "empty",
                "forced",
                "after_tool",
                "reasoning",
            )
            for mode in ("json", "stream")
        ),
        *(f"P{i:02}" for i in range(1, 15)),
        "P04.top_p",
        "P04.top_k",
        "P06.structure",
        "P06.calibration",
        "P09.none",
        "P09.required",
        "P09.named",
    }
    if not tokens or any(
        token not in allowed
        and not re.fullmatch(
            r"(?:L0[1-5]|R03)\.[1-9]\d{2,6}|R04\.(?:repeated_ngram|compression_ratio|longest_repeat)\.[1-9]\d{2,6}",
            token,
        )
        for token in tokens
    ):
        raise ValueError("unknown_or_unimplemented_test_selection")
    if "P08" in tokens:
        raise ValueError("P08_is_not_defined_in_SPEC")
    return tokens


def expand_requirements(requirements: Requirements, *, level: str = "core") -> Requirements:
    from servegauge.runner.plan import Requirement

    available = {
        *OPERATIONAL_CHECKS,
        *HEALTH_CHECKS,
        ROUTING_CHECK,
        *CACHE_CHECKS,
        "P01",
        "P02",
        "P03",
        "P04.top_p",
        "P04.top_k",
        "P05",
        "P06.structure",
        "P06.calibration",
        "P07",
        "P09.none",
        "P09.required",
        "P09.named",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
        "T01",
        "T02",
        "T03",
        "T05",
        "T06",
        "T07.passback",
        "T07.no_passback",
        "T08",
        "J01",
        "J02.validity",
        "J02.semantic",
        "J03",
        "J04",
        "J05",
        "B01",
        "B02",
        "B03.prompt",
        "B03.completion",
        "B04",
        "B05",
        "R01",
        "R05",
        "L06",
        "L06.fresh",
        "L06.warm",
        *(
            f"L{index:02}.{length}"
            for length in suite_config()["long_context"][level]["lengths"]
            for index in (1, 2, 3, 4, 5)
        ),
        *(f"R03.{length}" for length in suite_config()["reasoning"][level]["lengths"]),
        *(
            f"T04.{probe}.{mode}"
            for probe in (
                "single",
                "parallel",
                "irrelevant",
                "nested",
                "unicode",
                "empty",
                "forced",
                "after_tool",
                "reasoning",
            )
            for mode in ("json", "stream")
        ),
    }
    expanded = []
    for requirement in requirements.requires:
        minimum = requirement.params.get("min_effective_context")
        if requirement.params:
            if (
                set(requirement.params) != {"min_effective_context"}
                or not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or minimum <= 0
                or any(token not in {"L04", "L05", "S4", "context"} for token in requirement.tests)
            ):
                raise ValueError("known_positive_context_requirement_required")
            lengths = suite_config()["long_context"][level]["lengths"]
            qualifying = [length for length in lengths if length >= minimum]
            if not qualifying:
                raise ValueError("context_requirement_not_covered_at_selected_level")
            length = min(qualifying)
            expanded.append(
                Requirement(
                    id=requirement.id,
                    tests=[f"L04.{length}", f"L05.{length}"],
                    margin=requirement.margin,
                    params=requirement.params,
                )
            )
            continue
        checks = set()
        for token in requirement.tests:
            matches = {
                check for check in available if selected_check(check, normalize_tests(token))
            }
            if not matches:
                raise ValueError("requirement_check_not_measured")
            checks.update(matches)
        if requirement.margin is not None and any(c.startswith("S8.") for c in checks):
            raise ValueError("operational_metrics_are_report_only_not_SLA_gates")
        expanded.append(
            Requirement(
                id=requirement.id,
                tests=sorted(checks),
                margin=requirement.margin,
                params=requirement.params,
            )
        )
    return requirements.model_copy(update={"requires": expanded})


def _selected_items(items: list[Item], selected: set[str]) -> list[Item]:
    return [
        item
        for item in items
        if any(selected_check(check, selected) for check in item_checks(item))
    ]


async def _mock_metadata(plan: RunPlan, transports: dict[str, httpx.AsyncBaseTransport]) -> None:
    for endpoint in plan.endpoints:
        if endpoint.kind != "mock":
            continue
        try:
            async with httpx.AsyncClient(
                transport=transports.get(endpoint.id),
                timeout=10,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = await client.get(endpoint.base_url.rstrip("/") + "/models")
                if response.status_code != 200:
                    raise ValueError("metadata unavailable")
                rows = response.json()["data"]
                row = next(r for r in rows if r["id"] == endpoint.model)
                fields = {
                    key: row[key]
                    for key in (
                        "context_length",
                        "max_prompt_tokens",
                        "max_completion_tokens",
                        "supported_parameters",
                        "supports_tool_choice",
                        "mock_declaration",
                        "declared_json_schema",
                        "strict_schema_guarantee",
                    )
                    if key in row
                }
                if (
                    not isinstance(fields.get("context_length"), int)
                    or fields["context_length"] < 1
                ):
                    raise ValueError("invalid context declaration")
                endpoint.metadata.update(Redactor.from_environment().redact(fields))
                endpoint.metadata["metadata_source"] = endpoint.base_url.rstrip("/") + "/models"
        except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration):
            raise SafeError("mock_metadata_unavailable") from None


async def run_check(
    plan: RunPlan,
    *,
    tests: str | None = None,
    requirements: Requirements | None = None,
    baseline_id: str | None = None,
    reference_pack: ReferencePack | None = None,
    root: Path = Path("."),
    approved: bool = False,
    dry_run: bool = False,
    budget_scope: str | None = None,
    replay_of: dict[str, Any] | None = None,
    inventory: list[Item] | None = None,
    include_history_check: bool | None = None,
    capability_inventory: list[Item] | None = None,
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    on_estimate: Callable[[dict[str, Any]], None] | None = None,
    out: Path | None = None,
    captured: CapturedCheck | None = None,
) -> CheckRun:
    if approved and dry_run:
        raise ValueError("choose_yes_or_dry_run")
    if inventory is not None and (
        (not inventory and capability_inventory is None)
        or len({i.item_id for i in inventory}) != len(inventory)
        or (any(i.suite_id == "S0" for i in inventory) and capability_inventory != [])
    ):
        raise ValueError("unique_nonempty_primary_replay_inventory_required")
    if capability_inventory is not None and (
        (not capability_inventory and not inventory)
        or len({i.item_id for i in capability_inventory}) != len(capability_inventory)
        or any(i.suite_id != "S0" for i in capability_inventory)
    ):
        raise ValueError("unique_nonempty_capability_replay_inventory_required")
    if requirements:
        requirements = expand_requirements(requirements, level=plan.level)
        if tests is None:
            tests = ",".join(
                check for requirement in requirements.requires for check in requirement.tests
            )
    selected = normalize_tests(tests)
    if captured is not None:
        captured.validate_inputs(
            plan,
            selected,
            inventory,
            capability_inventory,
            baseline_id,
            requirements=requirements,
            reference_pack=reference_pack,
        )
    if requirements and any(
        not selected_check(check, selected) for req in requirements.requires for check in req.tests
    ):
        raise ValueError("required_check_excluded_by_tests_selection")
    profile, policy = load_profile(plan.profile), load_policy()
    if requirements:
        for requirement in requirements.requires:
            for check in requirement.tests:
                if check_class(check) == "comparative":
                    policy.margin(check, requirement.margin)
    config = plan.model_copy(deep=True)
    supplied = transports or {}
    if approved and not dry_run and captured is None:
        await _mock_metadata(config, supplied)
    primary = config.endpoints[0]
    if baseline_id is not None and baseline_id not in {e.id for e in config.endpoints[1:]}:
        raise ValueError("separate_baseline_endpoint_required")
    caps = (
        [i.model_copy(deep=True) for i in capability_inventory]
        if capability_inventory is not None
        else capability_items()
    )
    probes = (
        [i.model_copy(deep=True) for i in inventory]
        if inventory is not None
        else _selected_items(
            [
                *(
                    operational_items(config.level, config.seed, config.overrides.get("S8"))
                    if any(t in {"all", "S8"} or t.startswith("S8.") for t in selected)
                    else []
                ),
                *(
                    protocol_items(
                        config.level,
                        config.seed,
                        endpoint=primary,
                        profile=profile,
                        overrides=config.overrides.get("S1"),
                    )
                    if any(t in {"all", "S1", "S5"} or t.startswith("P") for t in selected)
                    else []
                ),
                *(
                    cache_items(config.level, config.seed, config.overrides.get("CACHE"))
                    if any(t in {"all", "S1", *CACHE_CHECKS} for t in selected)
                    else []
                ),
                *(
                    tool_items(config.level, config.seed, config.overrides.get("S2"))
                    if any(t in {"all", "S2"} or t.startswith("T") for t in selected)
                    else []
                ),
                *(
                    structured_items(config.level, config.seed, config.overrides.get("S3"))
                    if any(t in {"all", "S3"} or t.startswith("J") for t in selected)
                    else []
                ),
                *(
                    context_items(
                        config.level, config.seed, config.endpoints, config.overrides.get("S4")
                    )
                    if any(t in {"all", "S4"} or t.startswith("L") for t in selected)
                    else []
                ),
                *(
                    accounting_items(config.level, config.seed, config.overrides.get("S7"))
                    if any(t in {"all", "S7"} or t.startswith("B") for t in selected)
                    else []
                ),
                *(
                    reasoning_items(config.level, config.seed, config.overrides.get("S5"))
                    if any(t in {"all", "S5"} or t.startswith("R") for t in selected)
                    else []
                ),
                *(
                    fingerprint_items(config.level, config.seed, config.overrides.get("S6"))
                    if any(t in {"all", "S6"} or t.startswith("FP") for t in selected)
                    else []
                ),
            ],
            selected,
        )
    )
    if {i.item_id for i in caps} & {i.item_id for i in probes}:
        raise ValueError("unique_primary_and_capability_inventory_required")
    available = {
        "T08",
        ROUTING_CHECK,
        *(check for item in [*caps, *probes] for check in item_checks(item)),
    }
    if any(not any(selected_check(check, {token}) for check in available) for token in selected):
        raise ValueError("selected_check_has_no_matching_inventory_at_this_level")
    estimated = (
        captured.estimate(config, [*caps, *probes])
        if captured is not None
        else estimate_plan(config, items_override=[*caps, *probes])
    )
    estimated["replay_reservation_note"] = (
        "Replays share the primary budget; only observed failing invariant trials are resent."
    )
    if on_estimate:
        on_estimate(estimated)
    if dry_run:
        return CheckRun("dry-run", 0, [], estimated, None)
    if not approved:
        raise BudgetExceeded("explicit_yes_required")
    identifier = "check-" + uuid.uuid4().hex
    caps_id = identifier + "-capabilities" if caps else None
    store = RawStore(root / "data/raw")
    capability_records = []
    capability_results: list[CheckResult] = []
    if caps_id is not None:
        await run_plan(
            config.model_copy(update={"suites": ["S0"]}),
            root=root,
            run_id=caps_id,
            approved=approved,
            transports=supplied,
            items_override=caps,
            budget_scope=budget_scope or identifier,
            captured=captured.preflight() if captured is not None else None,
        )
        capability_records = list(store.exchanges(run_id=caps_id))
        for endpoint in config.endpoints:
            results, observed = score_capabilities(caps, capability_records, endpoint, caps_id)
            endpoint.metadata["observed_capabilities"] = observed
            if endpoint.id == primary.id:
                capability_results.extend(results)
    primary_plan = config.model_copy(
        update={"suites": sorted({item.suite_id for item in probes}) or ["S0"]}
    )
    await run_plan(
        primary_plan,
        root=root,
        run_id=identifier,
        approved=approved,
        transports=supplied,
        items_override=probes,
        budget_scope=budget_scope or identifier,
        captured=captured.primary() if captured is not None else None,
    )
    records = list(store.exchanges(run_id=identifier))
    initial_capabilities = [i for i in probes if i.suite_id == "S0"]
    if caps_id is None and initial_capabilities:
        for endpoint in config.endpoints:
            results, _ = score_capabilities(initial_capabilities, records, endpoint, identifier)
            if endpoint.id == primary.id:
                capability_results.extend(results)
    samples = collect(primary_plan, probes, records, profile, policy)
    agent_samples, agent_traces = await run_agents(
        primary_plan,
        probes,
        records,
        root=root,
        run_id=identifier,
        approved=approved,
        transports=supplied,
        budget_scope=budget_scope or identifier,
        captured=captured,
    )
    for endpoint_id, values in agent_samples.items():
        samples[endpoint_id].update(values)
    history_included = (
        selected_check("T08", selected) if include_history_check is None else include_history_check
    )
    if history_included:
        for endpoint_id in samples:
            samples[endpoint_id].setdefault("T08", history_format_sample(profile))
    routing_inventory = [*caps, *probes]
    for endpoint in primary_plan.endpoints:
        routing = routing_samples(
            routing_inventory,
            [*capability_records, *records],
            endpoint,
            run_id=identifier,
            capability_run_id=caps_id,
            contract=ROUTING_CONTRACT,
        )
        assert routing is not None
        samples[endpoint.id][ROUTING_CHECK] = routing
    confirmed: dict[tuple[str, str], bool] = {}
    replay_runs: dict[str, str] = {}
    replay_outcomes: dict[str, Any] = {}
    for check, sample in samples[primary.id].items():
        if (
            (check != ROUTING_CHECK and not selected_check(check, selected))
            or sample.check_class != "invariant"
            or sample.status is not None
        ):
            continue
        if sum(sample.values.values()) < policy.k_fail or not sample.replay_candidates:
            continue
        candidates = sample.replay_candidates[: suite_config()["protocol"]["max_replay_trials"]]
        wanted = {item_id for trial in candidates for item_id in trial}
        resend = [
            item
            for item in (routing_inventory if check == ROUTING_CHECK else probes)
            if item.item_id in wanted
        ]
        replay_plan = primary_plan.model_copy(update={"endpoints": [primary]})
        tool_replay = check.startswith("T")
        replay_cfg = suite_config()["tool_calling"]
        batch = int(replay_cfg["replay_batch"]) if tool_replay else 1
        limit = int(replay_cfg["replay_limit"]) if tool_replay else 1
        batch_ids: list[str] = []
        outcomes: list[dict[str, Any]] = []
        for start in range(0, limit, batch):
            replay_id = identifier + "-replay-" + safe_filename(check) + f"-{start}"
            repeated = []
            for index in range(start, min(start + batch, limit)):
                for item in resend:
                    clone = item.model_copy(deep=True)
                    if tool_replay:
                        clone.item_id += f"-confirmation-{index}"
                        clone.expected["trial"] = str(clone.expected.get("trial")) + f"-{index}"
                    repeated.append(clone)
            await run_plan(
                replay_plan,
                root=root,
                run_id=replay_id,
                approved=approved,
                transports=supplied,
                items_override=repeated,
                budget_scope=budget_scope or identifier,
                captured=captured.confirmation(check, len(batch_ids))
                if captured is not None
                else None,
            )
            replay_records = list(store.exchanges(run_id=replay_id))
            if check == ROUTING_CHECK:
                replay_sample = routing_samples(
                    repeated,
                    replay_records,
                    primary,
                    run_id=replay_id,
                    contract=ROUTING_CONTRACT,
                )
                assert replay_sample is not None
            else:
                replay_sample = collect(replay_plan, repeated, replay_records, profile, policy)[
                    primary.id
                ][check]
            reproduced = replay_sample.status is None and any(replay_sample.values.values())
            confirmed[(primary.id, check)] = reproduced
            sample.evidence.extend(replay_sample.evidence)
            batch_ids.append(replay_id)
            outcomes.append(replay_sample.model_dump(mode="json"))
            if reproduced:
                break
        sample.details.update(
            replay_run_id=batch_ids[0], replay_batch_ids=batch_ids, replay_outcomes=outcomes
        )
        replay_runs[check] = batch_ids[0]
        replay_outcomes[check] = outcomes
    operational_run_ids = {
        identifier,
        *(
            rid
            for tasks in agent_traces.values()
            for trace in tasks.values()
            for rid in trace.get("run_ids", [])
        ),
    }
    operational_records = [
        r for rid in sorted(operational_run_ids) for r in store.exchanges(run_id=rid)
    ]
    results = evaluate(
        primary_plan,
        identifier,
        probes,
        samples,
        policy,
        target_ids=[primary.id],
        baseline_id=baseline_id,
        reference_pack=reference_pack,
        requirements=requirements,
        replay_confirmed=confirmed,
        operational_records=operational_records,
    )
    annotate_context(results, primary_plan.endpoints, policy, contract=DRIFT_CONTRACT)
    wanted_checks = {r.check_id for r in results if selected_check(r.check_id, selected)}
    mark_reproduced(results, replay_of)
    captured_summary = captured.finish(results) if captured is not None else None
    if requirements:
        wanted_checks = {test for req in requirements.requires for test in req.tests}
    required = {"S0.answers", *wanted_checks}
    all_results = [*capability_results, *results]
    code = exit_code(all_results, required)
    directory = root / "data/runs" / safe_filename(identifier)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = reference_source(
        primary_plan, identifier, probes, samples, policy, manifest, baseline_id
    )
    _write_json(
        directory / "outcomes.json",
        Redactor.from_environment().redact(source.model_dump(mode="json")),
    )
    evaluation = {
        "run_id": identifier,
        "capability_run_id": caps_id,
        "baseline_id": baseline_id,
        "reference_pack_id": reference_pack.pack_id if reference_pack else None,
        "replay_runs": replay_runs,
        "replay_outcomes": replay_outcomes,
        "agent_traces": agent_traces,
        "policy": policy.model_dump(mode="json"),
        "suite_config": suite_config(),
        "profile": profile.model_dump(mode="json"),
        "requirements": requirements.model_dump(mode="json") if requirements else None,
        "tests": tests,
        "required_checks": sorted(required),
        "exit_code": code,
        "synthetic_mock": all(e.kind == "mock" for e in config.endpoints),
        "target_ids": [primary.id],
        "include_T08": history_included,
        "replay_of": replay_of,
        "routing_contract": ROUTING_CONTRACT,
        "drift_contract": DRIFT_CONTRACT,
    }
    evaluation["replay_batch_ids"] = {
        check: sample.details["replay_batch_ids"]
        for check, sample in samples[primary.id].items()
        if sample.details.get("replay_batch_ids")
    }
    if captured_summary is not None:
        evaluation["captured_replay"] = captured_summary
    seal_analysis(root, identifier, evaluation, reference_pack)
    _write_json(directory / "evaluation.json", Redactor.from_environment().redact(evaluation))
    destination = out or root / "reports" / safe_filename(identifier)
    write_check_report(directory, all_results, metadata=evaluation)
    write_check_report(destination, all_results, metadata=evaluation)
    return CheckRun(identifier, code, all_results, estimated, destination)
