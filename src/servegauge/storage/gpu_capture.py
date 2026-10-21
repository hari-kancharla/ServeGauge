"""Exclusive per-slot collection and source-bound reconstruction; never starts a GPU."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import EndpointSpec, Exchange
from servegauge.mock.backends import task_from_messages
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.observatory.provenance import implementation_hash
from servegauge.refs.owner import SelfhostOrigin
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import ReferenceSpec, RunPlan
from servegauge.scoring.common import score_capabilities
from servegauge.selfhost.controller import _authorize, request_hash
from servegauge.selfhost.remote_recipe import STUDY_PROFILE
from servegauge.selfhost.study_protocol import CollectionSpec
from servegauge.selfhost.study_recipe import study_template
from servegauge.storage.analysis import Analysis, reconstruct
from servegauge.storage.comparison import _envelope, _files, _private_json, pin_source
from servegauge.storage.gpu_collection import collection_inputs
from servegauge.storage.gpu_study import _paths, _write
from servegauge.storage.owner_reference import _session
from servegauge.storage.raw_store import read_file
from servegauge.storage.selfhost import SessionJournal
from servegauge.suites.registry import plan_items, prepare_item
from servegauge.suites.s0_capabilities import capability_items
from servegauge.tokencount.counters import mock_prompt_tokens


@dataclass
class StudyContext:
    root: Path
    kit: Path
    kit_hash: str
    spec: CollectionSpec
    assets: dict[str, Any]
    matrix: list[dict[str, Any]]

    @property
    def book(self) -> Path:
        return self.root / "data/raw/gpu-study-bindings" / self.kit_hash

    def slot(self, identifier: str) -> dict[str, Any]:
        selected = [row for row in self.matrix if row["slot_id"] == identifier]
        if len(selected) != 1:
            raise ValueError("study_declared_slot_required")
        return selected[0]


def study_context(root: Path, kit: Path, kit_hash: str) -> StudyContext:
    spec, assets, matrix = collection_inputs(root, kit, kit_hash)
    return StudyContext(root, kit, kit_hash, spec, assets, matrix)


def _entry(context: StudyContext, slot: dict[str, Any], kind: str) -> Path:
    if kind not in {"intent", "binding"}:
        raise ValueError("study_known_record_type_required")
    path = context.book / (content_hash(slot["slot_id"]) + "." + kind + ".json")
    _paths(path)
    return path


def _read(path: Path) -> dict[str, Any]:
    _paths(path)
    value = _private_json(path.read_bytes())
    if (
        not isinstance(value, dict)
        or set(value) != {"hash", "value"}
        or not isinstance(value["value"], dict)
        or value["hash"] != content_hash(value["value"])
    ):
        raise ValueError("study_capture_record_changed")
    return dict(value["value"])


def _record(path: Path, value: dict[str, Any]) -> None:
    _write(path, {"hash": content_hash(value), "value": value})


def _aware(value: str) -> datetime:
    clock = datetime.fromisoformat(value)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("study_aware_capture_clock_required")
    return clock


def _owner_session(
    context: StudyContext, slot: dict[str, Any], *, active: bool, runtime_sha256: str | None = None
) -> dict[str, Any]:
    if any(value is None for value in context.spec.engine_images.values()) or (
        context.spec.backend == "modal"
        and any(value is None for value in context.spec.modal_image_ids.values())
    ):
        raise ValueError("study_complete_reviewed_build_map_required")
    journal = SessionJournal(context.root, slot["session_id"])
    status = journal.status()
    if active and status["state"] != "running":
        raise ValueError("study_observed_running_session_required")
    plan, config = status["plan"], status["plan"]["configuration"]
    expected = study_template(context.spec.backend, slot["recipe"])
    fields = (
        "version",
        "purpose",
        "profile",
        "model_repository",
        "model_revision",
        "engine_version",
        "engine_revision",
        "precision",
        "template",
        "gpu",
        "backend",
    )
    if (
        plan["version"] != "selfhost-study-plan-v1"
        or plan["reference_qualification"] != "not_a_reference_session"
        or any(config[key] != expected[key] for key in fields)
        or journal.spec()["implementation_hash"] != implementation_hash()
    ):
        raise ValueError("study_reviewed_session_recipe_or_source_changed")
    if (
        config["image"] != context.spec.engine_images[config["engine_version"]]
        or (
            context.spec.backend == "modal"
            and config["modal_image_id"] != context.spec.modal_image_ids[config["engine_version"]]
        )
        or plan["budget_policy"] != context.assets["budget_policy"]
    ):
        raise ValueError("study_same_build_and_budget_binding_changed")
    digest = runtime_sha256 or status["facts"].get("runtime_receipt_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValueError("study_observed_runtime_required")
    source = _session(
        context.root,
        SelfhostOrigin(
            kind="selfhost",
            session_id=slot["session_id"],
            plan_hash=plan["plan_hash"],
            runtime_sha256=digest,
        ),
    )
    reserve = source["reservation"]
    if (
        reserve["owner"] is not True
        or reserve["request_hash"] != request_hash(slot["session_id"], plan)
        or Decimal(reserve["amount"]) != Decimal(plan["reserved_upper_usd"])
        or reserve["endpoint_id"] != "selfhost:" + config["backend"]
        or reserve["exchange_id"] != slot["session_id"]
    ):
        raise ValueError("study_original_compute_reservation_required")
    deadline = _aware(reserve["created_at"]) + timedelta(seconds=plan["maximum_duration_seconds"])
    if active and datetime.now(UTC) >= deadline:
        raise ValueError("study_reviewed_session_duration_expired")
    source["collection_deadline"] = deadline.isoformat()
    source["runtime_receipt_sha256"] = digest
    return source


def slot_plan(
    context: StudyContext, slot: dict[str, Any], session: dict[str, Any] | None
) -> RunPlan:
    mock = context.spec.scope == "mock_software_qa"
    if mock != (session is None):
        raise ValueError("study_scope_and_runtime_binding_required")
    metadata: dict[str, Any] = {
        "context_length": 8192,
        "max_completion_tokens": 2048,
        "fingerprint_sampling": context.spec.sampling_declaration(),
        "gpu_study": {
            "kit_hash": context.kit_hash,
            "slot_id": slot["slot_id"],
            "scope": context.spec.scope,
            "session_id": None if mock else slot["session_id"],
            "runtime_sha256": None if session is None else session["runtime_receipt_sha256"],
        },
    }
    if session is None:
        metadata["mock_declaration"] = {
            "seed": slot["capture_seed"],
            "clean": slot["recipe"] == "bf16",
            "backend": "study-sampling-v2-deterministic-echo-v1",
            "scope": "software_QA_not_GPU_precision",
        }
        url, model, key = "http://127.0.0.1/v1", "mock", None
    else:
        metadata.update(
            pricing={"prompt": "0", "completion": "0", "request": "0"},
            price_provenance={
                "basis": "dedicated_vllm_compute_reserved_separately",
                "session_plan_hash": session["plan"]["plan_hash"],
            },
        )
        url, model, key = (
            session["running"]["endpoint_url"],
            STUDY_PROFILE,
            session["plan"]["configuration"]["api_key_env"],
        )
    endpoint = EndpointSpec(
        id="study-capture",
        kind="mock" if mock else "selfhosted",
        base_url=url,
        model=model,
        claimed_profile=STUDY_PROFILE,
        api_key_env=key,
        metadata=metadata,
    )
    return RunPlan(
        plan_id=slot["run_id"],
        profile=STUDY_PROFILE,
        level=context.spec.design.level,
        seed=slot["capture_seed"],
        reference=ReferenceSpec(tier="MOCK" if mock else "BASELINE"),
        endpoints=[endpoint],
        suites=["S0", "S6"],
        max_usd=0,
        retries=0,
        requests_per_minute=context.spec.requests_per_minute,
        concurrency_per_endpoint=context.spec.concurrency_per_endpoint,
        timeout_seconds=60,
        overrides={"S6": context.assets["overrides"]},
    )


class _DeadlineTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport, deadline: datetime) -> None:
        self.transport, self.deadline = transport, deadline

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if datetime.now(UTC) >= self.deadline:
            raise httpx.RequestError("study_session_deadline", request=request)
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self.transport.aclose()


class _StudyMockTransport(httpx.AsyncBaseTransport):
    """Existing sampling-v2 plus an explicitly deterministic one-token echo law.

    The original S0 probe is unchanged. For its owned echo-OK task, P(OK)=1;
    these are genuine probabilities of this scripted interpreter. Other requests
    retain the original v2 behavior, including refusal of unmodeled logprob tasks.
    This separate study backend changes no historical v1/v2 mock observation.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport, seed: int) -> None:
        self.transport, self.seed = transport, seed

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        task = task_from_messages(body.get("messages", []))
        if (
            body.get("logprobs") is True
            and not body.get("stream")
            and task == {"type": "echo", "text": "OK"}
        ):
            prompt = mock_prompt_tokens(body)
            token = {"token": "OK", "bytes": [79, 75], "logprob": 0.0}
            return httpx.Response(
                200,
                json={
                    "id": "study-echo-" + str(self.seed),
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                            "logprobs": {"content": [{**token, "top_logprobs": [token]}]},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": 1,
                        "total_tokens": prompt + 1,
                    },
                },
                request=request,
            )
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self.transport.aclose()


def _mock_transport(slot: dict[str, Any]) -> httpx.AsyncBaseTransport:
    # Original software interventions exercise plumbing; none simulate physical precision.
    recipe = slot["recipe"]
    config = FaultConfig(
        logit_noise={"fp8": 0.3, "mxfp4": 0.9}.get(recipe, 0),
        route_to_other_model=1 if recipe in {"other-engine", "wrong-template"} else 0,
    )
    sampling = httpx.ASGITransport(
        app=create_app(seed=slot["capture_seed"], faults=config, sampling_version="v2")
    )
    return _StudyMockTransport(sampling, slot["capture_seed"])


def _guard_book(context: StudyContext) -> None:
    _paths(context.book)
    if not context.book.exists():
        return
    allowed = {".capture.lock"} | {
        content_hash(row["slot_id"]) + "." + kind + ".json"
        for row in context.matrix
        for kind in ("intent", "binding")
    }
    for path in context.book.iterdir():
        _paths(path)
        if path.name not in allowed or not path.is_file():
            raise ValueError("study_foreign_capture_book_entry")


@contextmanager
def capture_view(context: StudyContext) -> Iterator[None]:
    _guard_book(context)
    if not context.book.exists():
        yield
        return
    lock = context.book / ".capture.lock"
    _paths(lock)
    with FileLock(str(lock), timeout=0):
        yield


def _source_pin(root: Path, run_id: str) -> dict[str, Any]:
    # Bound supporting names before pin_source may traverse any supporting namespace.
    directory = root / "data/raw" / safe_filename(run_id)
    _paths(directory, directory / "analysis.json")
    sealed = _envelope(directory / "analysis.json")
    names = sealed["run_ids"]
    allowed = re.compile(re.escape(run_id) + r"(?:-capabilities|-confirmation-[a-f0-9]{16})?$")
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not allowed.fullmatch(name) for name in names
    ):
        raise ValueError("study_exclusive_supporting_runs_required")
    metadata = sealed["metadata"]
    if (
        metadata.get("agent_traces")
        or metadata.get("baseline_id") is not None
        or metadata.get("requirements") is not None
    ):
        raise ValueError("study_supported_collection_metadata_required")
    return pin_source(root, run_id).model_dump(mode="json")


def _verify_plan(root: Path, analysis: Analysis, expected: RunPlan) -> None:
    actual = analysis.plan.model_copy(deep=True)
    target = actual.endpoints[0]
    observed = target.metadata.pop("observed_capabilities", None)
    if actual != expected or len(analysis.plan.endpoints) != 1:
        raise ValueError("study_original_collection_plan_changed")
    cap_id = expected.plan_id + "-capabilities"
    cap_records = [r for r in analysis.records if r.run_id == cap_id]
    computed = score_capabilities(capability_items(), cap_records, expected.endpoints[0], cap_id)[1]
    if observed != computed:
        raise ValueError("study_raw_capability_binding_changed")
    cap_path = root / "data/raw" / safe_filename(cap_id) / "capture.json"
    _paths(cap_path)
    cap_context = _envelope(cap_path)
    if cap_context["items"] != [item.model_dump(mode="json") for item in capability_items()]:
        raise ValueError("study_original_capability_inventory_changed")
    probes = {item.item_id: item for item in plan_items(expected) if item.suite_id == "S6"}
    if analysis.items != list(probes.values()):
        raise ValueError("study_original_fingerprint_inventory_changed")
    expected_endpoint = analysis.plan.endpoints[0]
    for record in analysis.records:
        if record.run_id != expected.plan_id:
            continue
        if (
            record.item_id not in probes
            or record.sample_idx >= probes[record.item_id].expected["samples"]
        ):
            raise ValueError("study_declared_draw_required")
        decision = prepare_item(probes[record.item_id], expected_endpoint)
        if decision.body is not None and record.request != decision.body:
            raise ValueError("study_actual_sampling_request_changed")
        if decision.body is None and not record.metadata.get("request_not_sent"):
            raise ValueError("study_unoffered_request_cannot_be_observed")


def load_selected_slot(context: StudyContext, slot: dict[str, Any]) -> dict[str, Any] | None:
    _guard_book(context)
    path = _entry(context, slot, "binding")
    if not path.exists():
        return None
    binding, intent = _read(path), _read(_entry(context, slot, "intent"))
    if (
        binding.get("version") != "gpu-study-capture-v1"
        or binding.get("slot") != slot
        or binding.get("kit_hash") != context.kit_hash
        or binding.get("run_id") != slot["run_id"]
        or binding.get("intent_hash") != content_hash(intent)
    ):
        raise ValueError("study_capture_slot_binding_changed")
    session = (
        None
        if context.spec.scope == "mock_software_qa"
        else _owner_session(
            context, slot, active=False, runtime_sha256=intent["session"]["runtime_receipt_sha256"]
        )
    )
    plan = slot_plan(context, slot, session)
    # Later termination is expected; the original runtime/reservation facts stay fixed.
    if intent["plan"] != plan.model_dump(mode="json") or intent["session"] != session:
        if session is None or intent["session"] is None:
            raise ValueError("study_capture_intent_changed")
        compared = dict(session)
        compared["terminated_at"] = intent["session"].get("terminated_at")
        if intent["plan"] != plan.model_dump(mode="json") or intent["session"] != compared:
            raise ValueError("study_capture_intent_changed")
    pin = _source_pin(context.root, slot["run_id"])
    if binding["source"] != pin:
        raise ValueError("study_selected_raw_source_changed")
    ids: list[str] = []
    for name in pin["files_sha256"]:
        if name.endswith((".jsonl", ".jsonl.zst")):
            ids.extend(record.exchange_id for record in read_file(context.root / "data/raw" / name))
    if len(ids) != len(set(ids)):
        raise ValueError("study_duplicate_physical_capture_identity")
    analysis = reconstruct(context.root / "data/raw", slot["run_id"])
    _verify_plan(context.root, analysis, plan)
    manifest = _private_json((context.kit / "kit.json").read_bytes())
    if (
        analysis.manifest["source_tree_hash"] != manifest["source_tree_hash"]
        or analysis.metadata["profile"] != context.assets["profile"]
        or analysis.metadata["policy"] != context.assets["policy"]
    ):
        raise ValueError("study_original_measurement_implementation_changed")
    for record in analysis.records:
        for attempt in record.attempts:
            source = attempt.metadata.get("execution_source", {})
            if source.get("source_tree_hash") != manifest["source_tree_hash"]:
                raise ValueError("study_attempt_implementation_changed")
    invalid_clock = session is not None and not _runtime_interval_valid(analysis.records, session)
    samples = analysis.samples[plan.endpoints[0].id]
    if invalid_clock:
        for sample in samples.values():
            if sample.metric == "fingerprint":
                sample.mark("INCONCLUSIVE", "outside_reviewed_runtime_interval")
    return {
        "binding": binding,
        "binding_hash": content_hash(binding),
        "analysis": analysis,
        "samples": samples,
        "capture_ids": ids,
        "runtime_interval_valid": not invalid_clock,
        "observed_runtime": session["runtime"] if session is not None else None,
    }


def _runtime_interval_valid(records: list[Exchange], session: dict[str, Any]) -> bool:
    start = max(
        _aware(session["reservation"]["created_at"]).timestamp(),
        _aware(session["running_at"]).timestamp(),
    )
    deadline = _aware(session["collection_deadline"]).timestamp()
    if session["terminated_at"] is not None:
        deadline = min(deadline, _aware(session["terminated_at"]).timestamp())
    return all(
        start <= attempt.timings.start_unix <= deadline
        and attempt.timings.total_ms >= 0
        and attempt.timings.start_unix + attempt.timings.total_ms / 1000 <= deadline
        for record in records
        for attempt in record.attempts
    )


def load_slot(root: Path, kit: Path, kit_hash: str, slot_id: str) -> dict[str, Any] | None:
    context = study_context(root, kit, kit_hash)
    with capture_view(context):
        return load_selected_slot(context, context.slot(slot_id))


async def collect_slot(
    root: Path,
    kit: Path,
    kit_hash: str,
    slot_id: str,
    *,
    approved: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    context = study_context(root, kit, kit_hash)
    slot = context.slot(slot_id)
    _guard_book(context)
    with capture_view(context):
        existing = load_selected_slot(context, slot)
    if existing is not None:
        return {
            "state": "already_captured",
            "binding_hash": existing["binding_hash"],
            "model_api_calls": 0,
            "physical_GPU_sessions": 0,
        }
    env = dict(os.environ) if environment is None else environment
    owner = context.spec.scope == "owner_gpu"
    if (
        owner
        and not dry_run
        and env.get("SERVEGAUGE_OWNER_RUN", env.get("TRUEWEIGHTS_OWNER_RUN")) != "1"
    ):
        raise ValueError("study_owner_authorization_required")
    original_intent = _entry(context, slot, "intent")
    with capture_view(context):
        original_runtime = (
            _read(original_intent)["session"]["runtime_receipt_sha256"]
            if owner and original_intent.exists()
            else None
        )
    session = (
        _owner_session(context, slot, active=True, runtime_sha256=original_runtime)
        if owner
        else None
    )
    plan = slot_plan(context, slot, session)
    estimate = estimate_plan(plan)
    if dry_run:
        return {
            "state": "dry_run",
            "plan": plan.model_dump(mode="json"),
            "estimate": estimate,
            "model_api_calls": 0,
            "physical_GPU_sessions": 0,
        }
    if not approved:
        raise ValueError("study_explicit_collection_confirmation_required")
    if session is not None:
        _authorize(session["plan"], env, launch=True, remote=False)
    paths = [
        root / name
        for name in (
            "data/raw",
            "data/runs",
            "data/ledger/spend.jsonl",
            "data/ledger/.budget.lock",
            "budget/dev_spend.json",
            "budget/dev_spend.next.json",
        )
    ]
    _paths(*paths)
    if (root / "data/raw").exists():
        _files(root / "data/raw")
    context.book.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = context.book / ".capture.lock"
    _paths(lock)
    with FileLock(str(lock), timeout=0):
        completed = load_selected_slot(context, slot)
        if completed is not None:
            return {
                "state": "already_captured",
                "binding_hash": completed["binding_hash"],
                "model_api_calls": 0,
                "physical_GPU_sessions": 0,
            }
        path = _entry(context, slot, "intent")
        wanted = {
            "version": "gpu-study-intent-v1",
            "kit_hash": kit_hash,
            "slot": slot,
            "plan": plan.model_dump(mode="json"),
            "session": session,
        }
        if path.exists():
            if not resume or _read(path) != wanted:
                raise ValueError("study_existing_intent_requires_exact_explicit_resume")
        else:
            if resume:
                raise ValueError("study_resume_intent_missing")
            for identifier in (slot["run_id"], slot["run_id"] + "-capabilities"):
                for namespace in (root / "data/raw", root / "data/runs"):
                    target = namespace / safe_filename(identifier)
                    _paths(target)
                    if target.exists():
                        raise ValueError("study_existing_unbound_run_refused")
            _record(path, wanted)
        selected_transport = transport or (
            _mock_transport(slot)
            if not owner
            else httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        )
        if session is not None:
            selected_transport = _DeadlineTransport(
                selected_transport, _aware(session["collection_deadline"])
            )
        manifests = [
            root / "data/runs" / safe_filename(identifier) / "manifest.json"
            for identifier in (slot["run_id"], slot["run_id"] + "-capabilities")
        ]
        _paths(*manifests)
        actual_resume = resume and any(path.is_file() for path in manifests)
        result = await execute_plan(
            plan,
            root=root,
            run_id=slot["run_id"],
            resume=actual_resume,
            approved=True,
            transports={plan.endpoints[0].id: selected_transport},
        )
        binding = {
            "version": "gpu-study-capture-v1",
            "kit_hash": kit_hash,
            "slot": slot,
            "run_id": slot["run_id"],
            "intent_hash": content_hash(wanted),
            "source": _source_pin(root, slot["run_id"]),
            "collection_result": {
                "completed": result.completed,
                "errors": result.errors,
                "skipped": result.skipped,
                "unmeasured": result.unmeasured,
            },
            "scope": context.spec.scope,
            "physical_precision_verified": False,
        }
        _record(_entry(context, slot, "binding"), binding)
        loaded = load_selected_slot(context, slot)
        assert loaded is not None
        return {
            "state": (
                "captured_with_missing_evidence"
                if result.errors
                or result.unmeasured
                or any(
                    sample.status is not None or sample.details.get("missing_logprobs", 0)
                    for sample in loaded["samples"].values()
                    if sample.metric == "fingerprint"
                )
                else "captured"
            ),
            "run_id": result.run_id,
            "binding_hash": loaded["binding_hash"],
            "collection_result": binding["collection_result"],
            "physical_GPU_sessions": 0,
            "model_api_calls": 0
            if not owner
            else sum(len(record.attempts) for record in loaded["analysis"].records),
            "call_count_scope": "recorded_endpoint_attempts; not confirmed delivery or invoice",
        }


def study_status(root: Path, kit: Path, kit_hash: str) -> dict[str, Any]:
    context = study_context(root, kit, kit_hash)
    rows = []
    with capture_view(context):
        for slot in context.matrix:
            value = load_selected_slot(context, slot)
            pending = _entry(context, slot, "intent")
            state = (
                "captured"
                if value is not None
                else "pending_intent"
                if pending.exists()
                else "not_started"
            )
            if state == "pending_intent":
                intent = _read(pending)
                if intent.get("slot") != slot or intent.get("kit_hash") != kit_hash:
                    raise ValueError("study_pending_intent_binding_changed")
            rows.append(
                {
                    **slot,
                    "state": state,
                    "binding_hash": value["binding_hash"] if value is not None else None,
                }
            )
    return {
        "version": "gpu-study-status-v1",
        "scope": context.spec.scope,
        "kit_hash": kit_hash,
        "slots": rows,
        "counts": {
            state: sum(row["state"] == state for row in rows)
            for state in ("captured", "pending_intent", "not_started")
        },
        "model_api_calls": 0,
        "gpu_sessions_started": 0,
        "physical_precision_verified": False,
    }
