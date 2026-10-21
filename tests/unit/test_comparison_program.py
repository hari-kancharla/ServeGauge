"""Compile original requests before any local runner can consume captured responses."""

from __future__ import annotations

from pathlib import Path

import pytest
from unit.test_comparison_sources import hashes
from unit.test_comparison_sources import source as source

from servegauge.adapters.base import PreparedRequest
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, Item, Timings
from servegauge.mock.sealed_replay import ReplayObservation
from servegauge.runner.plan import RunPlan
from servegauge.storage.comparison import prepare_sources
from servegauge.storage.comparison_program import _compile_exchange, compile_program
from servegauge.suites.registry import item_request, prepare_item
from servegauge.suites.s1_protocol import probe
from servegauge.validation.capture import endpoint


def test_captured_mock_serialization_is_restored_without_changing_values_or_real_prompts():
    row, mock, item, plan = phase_input([200])
    restored = Item.model_validate_json(canonical_json(item.model_dump(mode="json")))
    assert item_request(item, mock) != item_request(restored, mock)
    compiled = _compile_exchange(row, mock, restored, plan, "0" * 64)
    assert compiled.wire_body == row.record.request
    assert compiled.source_item_hash == content_hash(restored.model_dump(mode="json"))
    assert compiled.item.model_dump() == restored.model_dump()
    provider = mock.model_copy(update={"kind": "selfhosted"})
    assert item_request(item, provider) == item_request(restored, provider)
    assert item_request(item, provider).messages == item.payload["messages"]


def test_compile_preserves_original_bodies_items_counts_and_phase_scope(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    before = hashes(out)
    program = compile_program(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before
    assert program.state == "compiled_only" and program.owner_comparison_executed is False
    assert len(program.cases) == 2
    for case in program.cases:
        assert case.population_n == 4 and case.replay_adds_independent_n is False
        assert {phase.source_run_id for phase in case.phases} == {
            "source",
            "preflight",
            "confirm-" + case.case_id[-1],
        }
        assert {row.source.role for phase in case.phases for row in phase.exchanges} == {
            "primary",
            "paired",
            "preflight",
            "confirmation",
        }
        for phase in case.phases:
            assert [row.source.source_position for row in phase.exchanges] == sorted(
                row.source.source_position for row in phase.exchanges
            )
            for row in phase.exchanges:
                assert row.wire_body is not None
                assert canonical_json(row.wire_body) == canonical_json(row.source.record.request)
                assert row.source_item_hash == content_hash(row.item.model_dump(mode="json"))
                assert row.item == next(i for i in phase.items if i.item_id == row.item.item_id)
    assert (
        program.program_hash()
        == compile_program(
            out, selection_hash=content_hash(selection.model_dump(mode="json"))
        ).program_hash()
    )


@pytest.mark.parametrize("field", ["model", "stream", "messages", "provider"])
def test_changed_compiler_request_refuses_instead_of_normalizing(source, monkeypatch, field):
    root, selection, out = source
    prepare_sources(root, selection, out)
    before = hashes(out)

    def changed(item, endpoint):
        original = prepare_item(item, endpoint)
        assert original.body is not None
        body = dict(original.body)
        body[field] = {
            "model": "other",
            "stream": not body["stream"],
            "messages": [],
            "provider": {"only": ["other"]},
        }[field]
        return PreparedRequest(body, metadata=original.metadata)

    monkeypatch.setattr("servegauge.storage.comparison_program.prepare_item", changed)
    with pytest.raises(ValueError, match="wire_request"):
        compile_program(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before


def test_wrong_anchor_or_changed_snapshot_never_compiles(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    with pytest.raises(ValueError):
        compile_program(out, selection_hash="0" * 64)
    path = next((out / "sources").glob("*/data/raw/*/capture.json"))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        compile_program(out, selection_hash=content_hash(selection.model_dump(mode="json")))


def test_protected_path_rejected_before_access(monkeypatch, tmp_path):
    protected = tmp_path / "Starter Files" / "capture"
    accesses = []
    original = Path.is_symlink

    def watched(path):
        accesses.append(path)
        return original(path)

    monkeypatch.setattr(Path, "is_symlink", watched)
    with pytest.raises(ValueError, match="path_refused"):
        compile_program(protected, selection_hash="0" * 64)
    assert not accesses


def phase_input(statuses, *, retries=2, skip=False):
    target = endpoint("source", 1, clean=True)
    item = probe(
        "P01",
        0,
        "cap",
        "captured prompt",
        {"type": "echo", "text": "x"},
        extra={"minimum_output_tokens": 100000} if skip else {},
    )
    plan = RunPlan(
        plan_id="fixture",
        profile="gpt-oss-120b",
        level="core",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0,
        endpoints=[target],
        suites=["S1"],
        retries=retries,
    )
    decision = prepare_item(item, target)
    request = decision.body or item_request(item, target).model_dump(mode="json")
    record = Exchange(
        exchange_id="e",
        run_id="source",
        endpoint_id=target.id,
        suite_id=item.suite_id,
        item_id=item.item_id,
        sample_idx=0,
        request=request,
        attempts=[
            Attempt(n=n + 1, status=status, timings=Timings(start_unix=0, total_ms=0))
            for n, status in enumerate(statuses)
        ],
        metadata={
            "request_not_sent": True,
            "verdict": decision.verdict,
            "skip_reason": decision.reason,
        }
        if skip
        else {},
    )
    row = ReplayObservation(
        case_id="case-a",
        role="primary",
        source_position=0,
        record=record,
        record_hash=content_hash(record.model_dump(mode="json")),
        request_hash=content_hash(request),
    )
    return row, target, item, plan


@pytest.mark.parametrize(
    ("statuses", "retries", "accepted"),
    [
        ([200], 2, True),
        ([503, 200], 2, True),
        ([503, 503, 503], 2, True),
        ([503], 0, True),
        ([200, 200], 2, False),
        ([503], 2, False),
        ([503, 503, 503, 503], 2, False),
    ],
)
def test_compilation_requires_the_ordinary_retry_decisions(statuses, retries, accepted):
    values = phase_input(statuses, retries=retries)
    if accepted:
        assert _compile_exchange(*values, "0" * 64).wire_body is not None
    else:
        with pytest.raises(ValueError, match="retry"):
            _compile_exchange(*values, "0" * 64)


def test_unsent_descriptor_and_skip_decision_are_compiled_without_a_response():
    row, target, item, plan = phase_input([], skip=True)
    result = _compile_exchange(row, target, item, plan, "0" * 64)
    assert result.wire_body is None and result.preparation_verdict == "NOT_SUPPORTED"
    assert result.logical_request == row.record.request and not result.source.record.attempts
    row.record.metadata["verdict"] = "INCONCLUSIVE"
    with pytest.raises(ValueError, match="skip_decision"):
        _compile_exchange(row, target, item, plan, "0" * 64)
