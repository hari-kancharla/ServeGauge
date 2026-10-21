"""The external provider supplies captured observations, never detector verdicts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from servegauge.comparison import provider
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, Timings
from servegauge.mock.sealed_replay import ReplayAddress, ReplayObservation
from servegauge.runner.ledger import Ledger

BODY = {"model": "synthetic", "messages": [{"role": "user", "content": "same"}], "stream": False}


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list[dict]]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provider, "_state", None)
    rows, addresses = [], []
    for index, content in enumerate(["first", "second"]):
        attempt = Attempt(
            n=1,
            status=200,
            response={"choices": [{"message": {"content": content}}]},
            timings=Timings(start_unix=0, total_ms=0),
        )
        record = Exchange(
            exchange_id=f"exchange-{index}",
            run_id="source",
            endpoint_id="source-endpoint",
            suite_id="S1",
            item_id=f"item-{index}",
            sample_idx=0,
            request=BODY,
            attempts=[attempt],
        )
        row = ReplayObservation(
            case_id="synthetic",
            role="primary",
            source_position=index,
            record_hash=content_hash(record.model_dump(mode="json")),
            request_hash=content_hash(BODY),
            record=record,
        )
        rows.append(row.model_dump(mode="json"))
        addresses.append(
            ReplayAddress(
                case_id="synthetic",
                role="primary",
                source_run_id="source",
                endpoint_id="source-endpoint",
                suite_id="S1",
                item_id=f"item-{index}",
                sample_idx=0,
                exchange_id=f"exchange-{index}",
                attempt_n=1,
            ).model_dump(mode="json")
        )
    blob = canonical_json(rows)
    (tmp_path / "replay-book.json").write_bytes(blob)
    options = {
        "config": {"book": "replay-book.json", "book_sha256": hashlib.sha256(blob).hexdigest()}
    }
    return options, addresses


async def test_identical_prompts_consume_distinct_exact_source_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    values = [
        await provider.call_api(
            canonical_json(BODY).decode(), options, {"vars": {"source_address": address}}
        )
        for address in addresses
    ]
    assert [v["output"]["response"]["choices"][0]["message"]["content"] for v in values] == [
        "first",
        "second",
    ]
    assert all("verdict" not in v["output"] and v["cost"] == 0 for v in values)
    audit = json.loads((tmp_path / "provider-audit.json").read_text())
    assert audit["tape"]["consumed_attempts"] == 2 and audit["tape"]["remaining_slots"] == 0
    assert audit["pending_reservations"] == 0 and audit["new_spend_usd"] == 0
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0


@pytest.mark.parametrize("kind", ["address", "body", "duplicate", "order"])
async def test_request_scope_order_and_no_cycling_are_harness_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    body = canonical_json(BODY).decode()
    selected = dict(addresses[0])
    if kind == "address":
        selected["endpoint_id"] = "different"
    elif kind == "body":
        body = canonical_json({**BODY, "model": "different"}).decode()
    elif kind == "order":
        selected = addresses[1]
    else:
        await provider.call_api(body, options, {"vars": {"source_address": selected}})
    result = await provider.call_api(body, options, {"vars": {"source_address": selected}})
    assert result["error"] == "captured_provider_refused" and "output" not in result
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0


@pytest.mark.parametrize("change", ["hash", "bytes", "noncanonical", "duplicate_keys"])
async def test_tampered_book_and_ambiguous_wire_bytes_never_normalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    body = canonical_json(BODY).decode()
    if change == "hash":
        options["config"]["book_sha256"] = "0" * 64
    elif change == "bytes":
        (tmp_path / "replay-book.json").write_text("[]")
    elif change == "noncanonical":
        body = " " + body
    else:
        body = '{"model":"other","model":"synthetic"}'
    result = await provider.call_api(body, options, {"vars": {"source_address": addresses[0]}})
    assert result["error"] == "captured_provider_refused" and "output" not in result


async def test_book_is_rechecked_between_persistent_worker_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    body = canonical_json(BODY).decode()
    first = await provider.call_api(body, options, {"vars": {"source_address": addresses[0]}})
    assert "output" in first
    (tmp_path / "replay-book.json").write_text("[]")
    result = await provider.call_api(body, options, {"vars": {"source_address": addresses[1]}})
    assert result["error"] == "captured_provider_refused"


async def test_existing_audit_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    (tmp_path / "provider-audit.json").write_text("foreign bytes")
    result = await provider.call_api(
        canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[0]}}
    )
    assert result["error"] == "captured_provider_refused"
    assert (tmp_path / "provider-audit.json").read_text() == "foreign bytes"


async def test_refused_worker_cannot_later_finish_with_a_clean_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    await provider.call_api("malformed", options, {"vars": {"source_address": addresses[0]}})
    later = await provider.call_api(
        canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[0]}}
    )
    assert later["error"] == "captured_provider_refused"
    assert json.loads((tmp_path / "provider-audit.json").read_text())["harness_refused"] is True


async def test_changed_worker_audit_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    await provider.call_api(
        canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[0]}}
    )
    (tmp_path / "provider-audit.json").write_text("changed bytes")
    with pytest.raises(ValueError, match="captured_provider_audit_changed"):
        await provider.call_api(
            canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[1]}}
        )
    assert (tmp_path / "provider-audit.json").read_text() == "changed bytes"


async def test_tool_injected_base_path_must_equal_the_isolated_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, addresses = prepare(tmp_path, monkeypatch)
    options["config"]["basePath"] = str(tmp_path)
    result = await provider.call_api(
        canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[0]}}
    )
    assert "output" in result
    options["config"]["basePath"] = str(tmp_path.parent)
    result = await provider.call_api(
        canonical_json(BODY).decode(), options, {"vars": {"source_address": addresses[1]}}
    )
    assert result["error"] == "captured_provider_refused"
