"""Bounded async execution with durable capture, reservations and resume manifests."""

from __future__ import annotations

import asyncio
import json
import os
import random
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from filelock import FileLock, Timeout

from servegauge.adapters.base import PreparedRequest
from servegauge.adapters.openrouter import fetch_generation, verify_generation
from servegauge.config import load_budget
from servegauge.core.errors import SafeError
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import exchange_id, safe_filename
from servegauge.core.manifest import build_manifest
from servegauge.core.models import Attempt, EndpointSpec, Exchange, Item
from servegauge.core.redaction import Redactor
from servegauge.runner.budget import BudgetExceeded, BudgetManager
from servegauge.runner.cost import actual_cost, prompt_reservation_tokens, worst_case_cost
from servegauge.runner.plan import RunPlan
from servegauge.runner.ratelimit import Sleep, TokenBucket
from servegauge.runner.retry import delay_seconds, retryable
from servegauge.storage.raw_store import RawStore
from servegauge.suites.registry import item_request, plan_items, prepare_item
from servegauge.transport.http import send_reserved, wire_hash

if TYPE_CHECKING:
    from servegauge.runner.captured import CapturedRun


@dataclass(frozen=True)
class RunResult:
    run_id: str
    completed: int
    skipped: int
    errors: int
    spent_usd: Decimal
    unmeasured: int = 0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".next.json")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def _failure_locations(error: BaseException) -> list[dict[str, Any]]:
    if isinstance(error, BaseExceptionGroup):
        return [row for child in error.exceptions for row in _failure_locations(child)]
    return [
        {
            "type": type(error).__name__,
            "frames": [
                {
                    "file": Path(frame.f_code.co_filename).name,
                    "function": frame.f_code.co_name,
                    "line": line,
                }
                for frame, line in traceback.walk_tb(error.__traceback__)
            ],
        }
    ]


def _journal_append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _journal_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_bytes().splitlines(keepends=True)
        if any(not line.endswith(b"\n") for line in lines):
            raise ValueError("partial attempt journal")
        return [json.loads(line) for line in lines]
    except ValueError:
        raise ValueError("attempt_journal_corrupt") from None


async def key_preflight(
    client: httpx.AsyncClient,
    endpoint: EndpointSpec,
    headers: dict[str, str],
    owner: bool,
    planned: Decimal,
) -> None:
    if endpoint.kind != "openrouter":
        return
    try:
        response = await client.get(endpoint.base_url.rstrip("/") + "/key", headers=headers)
        if response.status_code != 200:
            raise SafeError("key_preflight_unavailable")
        data = response.json()["data"]
        if not isinstance(data, dict):
            raise SafeError("key_preflight_unavailable")
        limit = data.get("limit")
        if limit is not None and (not Decimal(str(limit)).is_finite() or Decimal(str(limit)) < 0):
            raise SafeError("key_preflight_unavailable")
        if not owner and data.get("limit") is None:
            raise BudgetExceeded("development_key_requires_credit_limit")
        remaining = data.get("limit_remaining")
        if remaining is not None and not Decimal(str(remaining)).is_finite():
            raise SafeError("key_preflight_unavailable")
        if remaining is not None and Decimal(str(remaining)) < planned:
            raise BudgetExceeded("key_credit_below_planned_reservations")
        if not owner and remaining is None:
            raise BudgetExceeded("key_remaining_credit_unavailable")
    except (httpx.HTTPError, KeyError, ValueError, TypeError, InvalidOperation):
        raise SafeError("key_preflight_unavailable") from None


async def run_plan(
    plan: RunPlan,
    *,
    root: Path = Path("."),
    run_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    approved: bool = False,
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    sleep: Sleep = asyncio.sleep,
    after_capture: Callable[[Exchange], None] | None = None,
    items_override: list[Item] | None = None,
    budget_scope: str | None = None,
    captured: CapturedRun | None = None,
) -> RunResult:
    items = items_override if items_override is not None else plan_items(plan)
    seen_items: dict[str, Item] = {}
    for item in items:
        if item.item_id in seen_items:
            raise ValueError("duplicate_item_identity")
        dependencies = item.expected.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) or value not in seen_items for value in dependencies
        ):
            raise ValueError("dependencies_must_name_earlier_items")
        if dependencies and (
            item.expected.get("samples", 1) != 1
            or any(seen_items[value].expected.get("samples", 1) != 1 for value in dependencies)
        ):
            raise ValueError("dependent_phases_require_single_draw_items")
        seen_items[item.item_id] = item
    endpoints = [e for e in plan.endpoints if e.metadata.get("status", 0) == 0]
    if not endpoints:
        raise ValueError("no_eligible_endpoints")
    if not approved and any(e.kind != "mock" for e in endpoints) and not dry_run:
        raise BudgetExceeded("explicit_yes_required")
    if captured is not None:
        captured.validate(
            plan, items, transports or {}, resume=resume, dry_run=dry_run, approved=approved
        )
    prepared = {
        endpoint.id: [
            (
                item,
                captured.prepared(endpoint, item)
                if captured is not None
                else prepare_item(item, endpoint),
            )
            for item in items
        ]
        for endpoint in endpoints
    }
    for endpoint in endpoints:
        for _, decision in prepared[endpoint.id]:
            if decision.body is not None:
                worst_case_cost(decision.body, endpoint)
    decisions = {
        (endpoint.id, item.suite_id, item.item_id): decision
        for endpoint in endpoints
        for item, decision in prepared[endpoint.id]
    }
    identifier = run_id or f"run-{uuid.uuid4().hex}"
    policy = load_budget(root)
    budget = BudgetManager(
        root, budget_scope or identifier, policy, max_usd=Decimal(str(plan.max_usd))
    )
    if dry_run:
        return RunResult(identifier, 0, 0, 0, Decimal(0))
    directory = root / "data/runs" / safe_filename(identifier)
    config = plan.model_dump(mode="json")
    redactor = Redactor.from_environment()
    key_values = [os.environ.get(e.api_key_env, "") for e in endpoints if e.api_key_env]
    redactor = redactor.with_values(*key_values)
    manifest_file = directory / "manifest.json"
    hashes = {
        suite: content_hash([i.model_dump(mode="json") for i in items if i.suite_id == suite])
        for suite in plan.suites
    }
    if resume:
        if not manifest_file.exists():
            raise ValueError("resume_manifest_missing")
        previous = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            previous["config_hash"] != content_hash(redactor.redact(config))
            or previous["item_set_hashes"] != hashes
            or previous.get("budget_scope", identifier) != (budget_scope or identifier)
        ):
            raise ValueError("resume_config_mismatch")
    elif directory.exists():
        raise ValueError("run_id_already_exists; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / ".run.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise SafeError("run_already_active") from None
    try:
        if not resume:
            manifest = build_manifest(
                identifier,
                config,
                hashes,
                [e.model_dump(mode="json") for e in endpoints],
                redactor=redactor,
                reference_pack_id=plan.reference.pack,
                suite_versions={
                    suite: sorted({i.version for i in items if i.suite_id == suite})
                    for suite in plan.suites
                },
            )
            manifest["router_flags"] = {
                endpoint.id: [
                    {
                        "item_id": item.item_id,
                        "provider": decision.body["provider"],
                        "plugins": decision.body["plugins"],
                    }
                    for item, decision in prepared[endpoint.id]
                    if decision.body is not None and endpoint.kind == "openrouter"
                ]
                for endpoint in endpoints
            }
            manifest["excluded_endpoints"] = redactor.redact(
                [e.model_dump(mode="json") for e in plan.endpoints if e not in endpoints]
            )
            manifest["budget_scope"] = budget_scope or identifier
            _write_json(manifest_file, manifest)
            _write_json(
                directory / "items.json",
                redactor.redact([i.model_dump(mode="json") for i in items]),
            )
            from servegauge.storage.analysis import capture_context

            capture_context(
                root,
                identifier,
                {"manifest": manifest, "items": [item.model_dump(mode="json") for item in items]},
            )
        runtime_manifest = (
            build_manifest(
                identifier,
                config,
                hashes,
                [e.model_dump(mode="json") for e in endpoints],
                redactor=redactor,
            )
            if resume
            else manifest
        )
        execution_source = {
            key: runtime_manifest[key]
            for key in (
                "git_sha",
                "git_dirty",
                "git_state",
                "source_tree_hash",
                "source_identity",
                "packages",
            )
            if key in runtime_manifest
        }
        journal = directory / "attempts.jsonl"
        historical: dict[str, list[Attempt]] = {}
        known_costs: dict[str, tuple[Decimal, str]] = {}
        for event in _journal_read(journal):
            historical.setdefault(event["exchange_id"], []).append(
                Attempt.model_validate(event["attempt"])
            )
            known_costs[event["reservation_id"]] = (
                Decimal(event["actual_cost_usd"]),
                event["cost_source"],
            )
        if resume:
            budget.recover_interrupted(known_costs)
        store = RawStore(root / "data/raw", redactor=redactor)
        prior = {e.exchange_id: e for e in store.exchanges(run_id=identifier)}
        skipped = len(prior)
        records = list(prior.values())
        completed = {(record.endpoint_id, record.item_id): record for record in records}
        ready = {
            (endpoint.id, item.item_id): asyncio.Event() for endpoint in endpoints for item in items
        }
        for completed_key in completed:
            ready[completed_key].set()
        queued: list[tuple[EndpointSpec, Item, int, PreparedRequest]] = []
        for item in items:
            samples = int(item.expected.get("samples", 1))
            for sample_idx in range(samples):
                for endpoint in endpoints:
                    eid = exchange_id(
                        identifier, endpoint.id, item.suite_id, item.item_id, sample_idx
                    )
                    if eid not in prior:
                        decision = decisions[(endpoint.id, item.suite_id, item.item_id)]
                        queued.append((endpoint, item, sample_idx, decision))
        if captured is not None:
            queued = captured.order(queued)
        queue: asyncio.Queue[tuple[EndpointSpec, Item, int, PreparedRequest]] = asyncio.Queue()
        for entry in queued:
            queue.put_nowait(entry)
        semaphores = {e.id: asyncio.Semaphore(plan.concurrency_per_endpoint) for e in endpoints}
        buckets = {
            e.id: TokenBucket(plan.requests_per_minute, plan.concurrency_per_endpoint, sleep=sleep)
            for e in endpoints
        }
        token_buckets = {
            e.id: TokenBucket(plan.tokens_per_minute, plan.tokens_per_minute, sleep=sleep)
            for e in endpoints
            if plan.tokens_per_minute is not None
        }
        clients = {
            e.id: httpx.AsyncClient(
                transport=(transports or {}).get(e.id),
                timeout=plan.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            )
            for e in endpoints
        }
        headers_by_id: dict[str, dict[str, str]] = {}
        payment_blocked: set[tuple[str, str | None]] = set()
        try:
            for endpoint in endpoints:
                headers = {"content-type": "application/json"}
                headers_by_id[endpoint.id] = headers
                if all(decision.body is None for _, decision in prepared[endpoint.id]):
                    continue
                if endpoint.api_key_env:
                    key = os.environ.get(endpoint.api_key_env)
                    if not key:
                        raise SafeError("api_key_environment_missing")
                    headers["authorization"] = "Bearer " + key
                headers_by_id[endpoint.id] = headers
                planned = sum(
                    (
                        worst_case_cost(decision.body, endpoint)
                        * int(item.expected.get("samples", 1))
                        for item, decision in prepared[endpoint.id]
                        if decision.body is not None
                    ),
                    Decimal(0),
                )
                await key_preflight(
                    clients[endpoint.id],
                    endpoint,
                    headers,
                    budget.owner,
                    min(planned, budget.available()),
                )

            async def execute(
                endpoint: EndpointSpec, item: Item, sample_idx: int, decision: PreparedRequest
            ) -> Exchange:
                eid = exchange_id(identifier, endpoint.id, item.suite_id, item.item_id, sample_idx)
                if decision.body is None:
                    if captured is not None:
                        captured.skip(endpoint, item, sample_idx)
                    record = Exchange(
                        exchange_id=eid,
                        run_id=identifier,
                        endpoint_id=endpoint.id,
                        suite_id=item.suite_id,
                        item_id=item.item_id,
                        sample_idx=sample_idx,
                        request=redactor.redact(
                            captured.logical_request(endpoint, item, sample_idx)
                            if captured is not None
                            else item_request(item, endpoint).model_dump(mode="json")
                        ),
                        attempts=[],
                        cost_usd=0,
                        cost_source="estimate",
                        metadata={
                            **decision.metadata,
                            "verdict": decision.verdict,
                            "skip_reason": decision.reason,
                            "request_not_sent": True,
                        },
                    )
                    if captured is not None:
                        captured.capture(record)
                    if store.append(record):
                        records.append(record)
                        if after_capture:
                            after_capture(record)
                    return record
                body = decision.body
                url = endpoint.base_url.rstrip("/") + "/chat/completions"
                headers = headers_by_id[endpoint.id]
                attempts = list(historical.get(eid, []))
                allowed_statuses = item.expected.get("allowed_statuses", [200])

                def needs_retry(attempt: Attempt) -> bool:
                    return retryable(attempt) and (
                        attempt.error is not None or attempt.status not in allowed_statuses
                    )

                overrun: BudgetExceeded | None = None
                rng = random.Random(int(eid[:16], 16) ^ plan.seed)
                while (not attempts or needs_retry(attempts[-1])) and len(attempts) <= plan.retries:
                    extensions = (
                        captured.extensions(endpoint, item, sample_idx, len(attempts) + 1)
                        if captured is not None
                        else None
                    )
                    await buckets[endpoint.id].acquire()
                    if endpoint.id in token_buckets:
                        await token_buckets[endpoint.id].acquire(
                            prompt_reservation_tokens(body, endpoint)
                            + int(body.get("max_tokens", 256))
                        )
                    reserved = worst_case_cost(body, endpoint)
                    reservation = budget.reserve(reserved, endpoint.id, eid, wire_hash(url, body))
                    attempt = await send_reserved(
                        clients[endpoint.id],
                        url,
                        body,
                        headers,
                        budget,
                        reservation,
                        len(attempts) + 1,
                        redactor,
                        timeout_seconds=plan.timeout_seconds,
                        request_extensions=extensions,
                    )
                    if captured is not None:
                        captured.annotate_attempt(endpoint, item, sample_idx, attempt)
                    attempt.metadata["execution_source"] = execution_source
                    if isinstance(endpoint.metadata.get("pricing"), dict):
                        attempt.metadata["accounting_price_snapshot"] = redactor.redact(
                            {
                                "pricing": endpoint.metadata["pricing"],
                                "provenance": endpoint.metadata.get("price_provenance"),
                            }
                        )
                    if endpoint.kind == "openrouter" and attempt.status == 402:
                        payment_blocked.add((endpoint.base_url, endpoint.api_key_env))
                    generation: dict[str, Any] | None = None
                    if endpoint.kind == "openrouter" and attempt.response and attempt.status == 200:
                        fetched = await fetch_generation(
                            clients[endpoint.id],
                            endpoint.base_url,
                            str(attempt.response.get("id", "")),
                            headers,
                            sleep=sleep,
                        )
                        verification = verify_generation(endpoint, fetched.metadata)
                        generation = fetched.metadata or None
                        attempt.metadata.update(
                            {
                                "generation_record": generation,
                                "generation_record_status": fetched.status,
                                "generation_poll_attempts": fetched.polls,
                                "routing_status": verification.status,
                                "exclude_from_scoring": verification.exclude_from_scoring,
                                "tag_verified": verification.tag_verified,
                                "candidate_fallback": verification.true_fallback,
                                "fallback_schema_verified": False,
                            }
                        )
                    actual, source = actual_cost(
                        attempt.response, endpoint, reserved, generation=generation
                    )
                    attempt.metadata.update(
                        {
                            "actual_cost_usd": str(actual),
                            "reserved_usd": str(reserved),
                            "cost_source": source,
                        }
                    )
                    _journal_append(
                        journal,
                        {
                            "exchange_id": eid,
                            "reservation_id": reservation.reservation_id,
                            "actual_cost_usd": str(actual),
                            "cost_source": source,
                            "attempt": attempt.model_dump(mode="json"),
                        },
                    )
                    attempts.append(attempt)
                    try:
                        budget.reconcile(reservation.reservation_id, actual, source)
                    except BudgetExceeded as cap_error:
                        if cap_error.code != "actual_exceeded_reservation":
                            raise
                        overrun = cap_error
                        break
                    if captured is not None:
                        captured.validate_attempt(endpoint, item, sample_idx, attempt)
                    if needs_retry(attempt) and len(attempts) <= plan.retries:
                        await sleep(
                            delay_seconds(
                                attempt, plan.retry_base_seconds, plan.retry_max_seconds, rng
                            )
                        )
                final = attempts[-1]
                error = overrun.code if overrun else final.error
                if final.metadata.get("exclude_from_scoring"):
                    error = "ROUTING_MISMATCH"
                if error is None and final.status not in allowed_statuses:
                    error = f"http_status_{final.status}"
                if error is None and final.response is None and final.status == 200:
                    error = "invalid_response_json"
                costs = [Decimal(a.metadata.get("actual_cost_usd", "0")) for a in attempts]
                usage = final.response.get("usage") if final.response else None
                if usage is not None and not isinstance(usage, dict):
                    final.metadata["invalid_usage_shape"] = True
                record = Exchange(
                    exchange_id=eid,
                    run_id=identifier,
                    endpoint_id=endpoint.id,
                    suite_id=item.suite_id,
                    item_id=item.item_id,
                    sample_idx=sample_idx,
                    request=redactor.redact(body),
                    attempts=attempts,
                    status=final.status,
                    response=final.response,
                    provider_meta=final.metadata.get("generation_record"),
                    sse_events=final.sse_events,
                    usage=usage if isinstance(usage, dict) else None,
                    cost_usd=float(sum(costs, Decimal(0))),
                    cost_source="generation_record"
                    if all(a.metadata.get("cost_source") == "generation_record" for a in attempts)
                    else "provider"
                    if all(a.metadata.get("cost_source") == "provider" for a in attempts)
                    else "estimate",
                    error=error,
                    request_url=redactor.url(url),
                    request_headers=redactor.redact(headers),
                    response_headers=final.response_headers,
                    response_text=final.response_text,
                    timings=final.timings,
                    metadata={**final.metadata, **decision.metadata},
                )
                if captured is not None:
                    captured.capture(record)
                if store.append(record):
                    records.append(record)
                    if after_capture:
                        after_capture(record)
                if overrun:
                    raise overrun
                return record

            async def worker() -> None:
                while not queue.empty():
                    try:
                        endpoint, item, sample_idx, decision = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    for dependency in item.expected.get("depends_on", []):
                        await ready[(endpoint.id, dependency)].wait()
                    prerequisites = [
                        completed[(endpoint.id, dependency)]
                        for dependency in item.expected.get("depends_on", [])
                    ]
                    if any(
                        record.error
                        or record.status != 200
                        or record.metadata.get("request_not_sent")
                        for record in prerequisites
                    ):
                        decision = PreparedRequest(
                            None,
                            "INCONCLUSIVE",
                            "prefix prerequisite unavailable",
                            {"prerequisite_not_observed": True},
                        )
                    if prerequisites:
                        decision = PreparedRequest(
                            decision.body,
                            decision.verdict,
                            decision.reason,
                            {
                                **decision.metadata,
                                "completed_dependencies": [
                                    record.exchange_id for record in prerequisites
                                ],
                            },
                        )
                    async with semaphores[endpoint.id]:
                        if (
                            decision.body is not None
                            and (
                                endpoint.base_url,
                                endpoint.api_key_env,
                            )
                            in payment_blocked
                        ):
                            decision = PreparedRequest(
                                None,
                                "INCONCLUSIVE",
                                "OpenRouter account/key payment rejection; check funded credits",
                                {"account_credit_blocked": True},
                            )
                        record = await execute(endpoint, item, sample_idx, decision)
                    completed[(endpoint.id, item.item_id)] = record
                    ready[(endpoint.id, item.item_id)].set()
                    queue.task_done()

            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(len(endpoints) * plan.concurrency_per_endpoint):
                        group.create_task(worker())
            except Exception as error:
                _write_json(
                    directory / "failure.json",
                    {
                        "execution_source": execution_source,
                        "exceptions": _failure_locations(error),
                    },
                )
                raise
        finally:
            for client in clients.values():
                await client.aclose()
        if captured is not None:
            captured.finish()
        result = RunResult(
            identifier,
            len(records),
            skipped,
            sum(r.error is not None for r in records),
            budget.snapshot()["run_spent"],
            sum(bool(r.metadata.get("request_not_sent")) for r in records),
        )
        _write_json(
            directory / "result.json",
            {
                "run_id": result.run_id,
                "completed": result.completed,
                "skipped": result.skipped,
                "errors": result.errors,
                "spent_usd": str(result.spent_usd),
            },
        )
        return result
    finally:
        lock.release()
