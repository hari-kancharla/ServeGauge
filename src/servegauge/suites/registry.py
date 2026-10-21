"""Binding suite protocol and initial transport-only S0 inventory."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from servegauge.core.capabilities import declared_structured
from servegauge.core.models import (
    ChatRequest,
    CheckResult,
    EndpointSpec,
    Exchange,
    Item,
    ItemScore,
    ModelProfile,
    ReferencePack,
    SuiteConfig,
)

if TYPE_CHECKING:
    from servegauge.adapters.base import PreparedRequest
    from servegauge.runner.plan import RunPlan


class Suite(Protocol):
    id: str
    version: str

    def items(self, profile: ModelProfile, cfg: SuiteConfig) -> Iterable[Item]: ...
    def build_request(
        self, item: Item, endpoint: EndpointSpec, profile: ModelProfile
    ) -> ChatRequest: ...
    def score_item(self, item: Item, exchanges: list[Exchange]) -> ItemScore: ...
    def aggregate(
        self, scores: list[ItemScore], ref: ReferencePack | None
    ) -> list[CheckResult]: ...


def plan_items(plan: RunPlan) -> list[Item]:
    if any(
        suite not in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"} for suite in plan.suites
    ):
        raise ValueError("suite_not_implemented")
    if plan.suites != ["S0"] or "items" not in plan.overrides.get("S0", {}):
        from servegauge.profiles import load_profile
        from servegauge.selfhost.remote_recipe import STUDY_PROFILE
        from servegauge.selfhost.study_collection import study_fingerprint_items
        from servegauge.suites.s0_capabilities import capability_items
        from servegauge.suites.s1_cache import cache_items
        from servegauge.suites.s1_protocol import protocol_items
        from servegauge.suites.s2_tools import tool_items
        from servegauge.suites.s3_structured import structured_items
        from servegauge.suites.s4_longctx import context_items
        from servegauge.suites.s5_longout import reasoning_items
        from servegauge.suites.s6_fingerprint import fingerprint_items
        from servegauge.suites.s7_accounting import accounting_items
        from servegauge.suites.s8_operations import operational_items

        profile = load_profile(plan.profile)
        return [
            *(capability_items() if "S0" in plan.suites else []),
            *(
                operational_items(plan.level, plan.seed, plan.overrides.get("S8"))
                if "S8" in plan.suites
                else []
            ),
            *(
                protocol_items(
                    plan.level,
                    plan.seed,
                    endpoint=plan.endpoints[0],
                    profile=profile,
                    overrides=plan.overrides.get("S1"),
                )
                if "S1" in plan.suites
                else []
            ),
            *(
                cache_items(plan.level, plan.seed, plan.overrides.get("CACHE"))
                if "S1" in plan.suites
                else []
            ),
            *(
                tool_items(plan.level, plan.seed, plan.overrides.get("S2"))
                if "S2" in plan.suites
                else []
            ),
            *(
                structured_items(plan.level, plan.seed, plan.overrides.get("S3"), profile=profile)
                if "S3" in plan.suites
                else []
            ),
            *(
                context_items(
                    plan.level, plan.seed, plan.endpoints, plan.overrides.get("S4"), profile=profile
                )
                if "S4" in plan.suites
                else []
            ),
            *(
                accounting_items(plan.level, plan.seed, plan.overrides.get("S7"))
                if "S7" in plan.suites
                else []
            ),
            *(
                reasoning_items(plan.level, plan.seed, plan.overrides.get("S5"), profile=profile)
                if "S5" in plan.suites
                else []
            ),
            *(
                (
                    study_fingerprint_items(
                        plan.level, plan.seed, plan.overrides.get("S6"), profile_id=profile.id
                    )
                    if profile.id == STUDY_PROFILE
                    else fingerprint_items(plan.level, plan.seed, plan.overrides.get("S6"))
                )
                if "S6" in plan.suites
                else []
            ),
        ]
    # Explicit transport-only inventory remains for M3 crash/budget regression tests.
    count = plan.overrides.get("S0", {}).get("items", 1)
    if not isinstance(count, int) or not 1 <= count <= 100000:
        raise ValueError("S0 items must be a positive bounded integer")
    return [
        Item(
            item_id=f"transport-{index}",
            suite_id="S0",
            version="m3-transport-v1",
            payload={
                "messages": [
                    {"role": "user", "content": 'TW_TASK: {"type":"echo","text":"ready"}'}
                ],
                "params": {"max_tokens": 32},
            },
            expected={"http_status": 200},
            tags=["transport-only; full S0 acceptance pending M7"],
        )
        for index in range(count)
    ]


def item_checks(item: Item) -> list[str]:
    from servegauge.suites.s1_cache import VERSION as CACHE_VERSION
    from servegauge.suites.s1_cache import cache_check

    if item.version == CACHE_VERSION:
        return [cache_check(item)]
    if item.suite_id == "S8":
        from servegauge.suites.s8_operations import (
            AGENT_CHECKS,
            HEALTH_CHECKS,
            STATIC_CHECKS,
            operational_health_contract,
        )

        return [
            *(AGENT_CHECKS if "agent_goal" in item.expected else STATIC_CHECKS),
            *(HEALTH_CHECKS if operational_health_contract([item]) is not None else []),
        ]
    check = item.expected.get("check_id", item.suite_id)
    if check == "B02":
        return ["B02", "B03.prompt", "B03.completion"]
    if check == "P09":
        return [check + "." + item.expected["variant"]]
    if check == "P06.structure":
        return [check, "P06.calibration"]
    if check == "tools-corpus":
        return ["T01", "T02", "T03", "T05", "T06"]
    if check.startswith("T04"):
        return [check, "T05", "T06"]
    if check == "J02":
        return [
            "J02.validity",
            "J02.semantic",
            *(["J04"] if item.expected["category"] == "flat" else []),
        ]
    if check == "context":
        return ["L04", *(f"L{index:02}.{item.expected['length']}" for index in (1, 2, 3, 4, 5))]
    if check == "L06":
        return ["L06", "L06.fresh", "L06.warm"]
    if check == "R01":
        return ["R01", "R05"]
    if check.startswith("R03."):
        return [
            check,
            *(
                f"R04.{metric}.{check.split('.')[-1]}"
                for metric in ("repeated_ngram", "compression_ratio", "longest_repeat")
            ),
        ]
    return [check]


def item_request(item: Item, endpoint: EndpointSpec | None = None) -> ChatRequest:
    payload = copy.deepcopy(item.payload)
    if endpoint is not None and item.expected.get("agent_goal") is not None:
        supported = endpoint.metadata.get("supported_parameters", [])
        if endpoint.kind == "mock" or "reasoning_effort" in supported:
            payload["params"]["reasoning_effort"] = "low"
        elif "reasoning" in supported:
            payload["params"]["reasoning"] = {"effort": "low"}
    if (
        endpoint is not None
        and endpoint.kind == "openrouter"
        and "reasoning_effort" in payload.get("params", {})
    ):
        effort = payload["params"].pop("reasoning_effort")
        payload["params"]["reasoning"] = {"effort": effort}
    if endpoint is not None and endpoint.kind == "mock" and item.expected.get("mock_task"):
        for message in reversed(payload["messages"]):
            if message["role"] == "user":
                message["content"] += "\nTW_TASK: " + json.dumps(
                    item.expected["mock_task"], ensure_ascii=False
                )
                break
    return ChatRequest.model_validate(payload)


def prepare_item(item: Item, endpoint: EndpointSpec) -> PreparedRequest:
    from servegauge.adapters.base import PreparedRequest, prepare_request

    minimum_output = item.expected.get("minimum_output_tokens")
    if item.expected.get("check_id") == "P12":
        declared = endpoint.metadata.get("context_length")
        source_bound = item.expected.get("context_length")
        if isinstance(declared, int) and isinstance(source_bound, int) and declared > source_bound:
            return PreparedRequest(
                None,
                "INCONCLUSIVE",
                "P12 probe was built for a smaller context; regenerate at this endpoint's bound",
            )
    if minimum_output is not None:
        cap = endpoint.metadata.get("max_completion_tokens")
        if not isinstance(cap, int):
            return PreparedRequest(None, "INCONCLUSIVE", "declared output cap unavailable")
        if cap < minimum_output:
            return PreparedRequest(None, "NOT_SUPPORTED", "long output exceeds declared cap")

    needed = item.expected.get("minimum_context_tokens")
    if needed is not None:
        context = endpoint.metadata.get("context_length")
        if not isinstance(context, int) or context < 1:
            return PreparedRequest(None, "INCONCLUSIVE", "declared context bound unavailable")
        if needed > context:
            return PreparedRequest(
                None,
                "NOT_SUPPORTED",
                "requested length above declared context",
                {"required_context_tokens": needed, "declared_context_tokens": context},
            )
        prompt_cap = endpoint.metadata.get("max_prompt_tokens")
        visible = item.expected.get("visible_prompt_tokens")
        if isinstance(prompt_cap, int) and isinstance(visible, int) and visible > prompt_cap:
            return PreparedRequest(None, "NOT_SUPPORTED", "input length above declared prompt cap")
    if (
        item.suite_id == "S2"
        and len(item.expected.get("calls", [])) > 1
        and endpoint.defaults.get("parallel_tool_calls") is False
    ):
        return PreparedRequest(
            None,
            "INCONCLUSIVE",
            "configured defaults disable the requested parallel behavior",
            {"conflicting_default": "parallel_tool_calls"},
        )
    observed = endpoint.metadata.get("observed_capabilities", {})
    if item.suite_id != "S0" and "answers" in observed and observed["answers"] is not True:
        return PreparedRequest(None, "INCONCLUSIVE", "basic capability preflight did not qualify")
    missing = [
        feature
        for feature in item.expected.get("requires", [])
        if feature in observed
        and observed[feature] is not True
        and not (
            feature == "json_schema"
            and item.expected.get("category") == "flat"
            and item.expected.get("check_id") == "J02"
            and declared_structured(endpoint) is True
        )
    ]
    if missing:
        uncertain = any(observed[f] is None for f in missing)
        return PreparedRequest(
            None,
            "INCONCLUSIVE" if uncertain else "NOT_SUPPORTED",
            "capability probe unavailable" if uncertain else "capability probe rejected",
            {"unsupported_features": missing},
        )
    return prepare_request(item_request(item, endpoint), endpoint)
