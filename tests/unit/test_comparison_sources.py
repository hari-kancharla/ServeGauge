"""Sealed source preparation is private, read-only and separate from comparison runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Attempt, Timings
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import capture_context, reconstruct, seal_analysis
from servegauge.storage.comparison import (
    CaseSelection,
    ComparisonRequest,
    ComparisonSelection,
    pin_source,
    prepare_sources,
    propose_selection,
    save_selection,
    verify_sources,
)
from servegauge.storage.comparison_replay import load_replay_book
from servegauge.storage.raw_store import RawStore
from servegauge.suites.config import suite_config
from servegauge.suites.s1_protocol import probe, tool
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import load_policy


def hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def source(tmp_path: Path):
    root = tmp_path / "workspace"
    targets = [endpoint(name, i, clean=i == 2) for i, name in enumerate(["a", "b", "base"])]
    plan = RunPlan(
        plan_id="sealed-comparison-fixture",
        profile="gpt-oss-120b",
        level="core",
        seed=15,
        reference={"tier": "BASELINE"},
        max_usd=0,
        endpoints=targets,
        suites=["S0", "S1"],
    )
    run = "source"
    primary = [
        probe(
            "P01",
            i,
            "cap",
            "same exact captured request " + str(i),
            {"type": "echo", "text": " token" * 24},
            params={"max_tokens": 2},
            stream=i % 2 == 0,
            extra={"full_text": " token" * 24},
        )
        for i in range(4)
    ]
    unrelated = probe("P02", 0, "stop", "unrelated", {"type": "echo", "text": "stop"})
    structured = probe(
        "P09",
        0,
        "required",
        "structured call fixture",
        {"name": "convert", "arguments": {"value": 1, "unit": "m"}},
        tools=[tool()],
        tool_choice="required",
        extra={"mode": "required"},
    )
    caps = [probe("S0.answers", 0, "answer", "answer", {"type": "echo", "text": "ok"})]
    inventory = {run: [*primary, unrelated, structured], "preflight": caps}
    inventory.update({f"confirm-{t.id}": [primary[0]] for t in targets[:2]})
    store = RawStore(root / "data/raw")
    for identifier, items in inventory.items():
        capture_context(
            root,
            identifier,
            {
                "manifest": {"run_id": identifier, "config": plan.model_dump(mode="json")},
                "items": [i.model_dump(mode="json") for i in items],
            },
        )
        for target in targets:
            if identifier.startswith("confirm-") and identifier != "confirm-" + target.id:
                continue
            backend = ScriptedBackend(
                int(target.metadata["mock_declaration"]["seed"]),
                FaultConfig(ignore_max_tokens=target.id != "base"),
            )
            for record in captures(items, backend, target, identifier):
                record.attempts[0].response = record.response
                record.attempts[0].sse_events = record.sse_events
                if identifier == run and target.id == "a" and record.item_id == primary[1].item_id:
                    record.attempts.insert(
                        0,
                        Attempt(
                            n=1,
                            status=503,
                            timings=Timings(start_unix=0, total_ms=0),
                            response={"error": {"code": "fixture_retry"}},
                        ),
                    )
                    record.attempts[1].n = 2
                assert store.append(record)
    metadata = {
        "run_id": run,
        "profile": load_profile(plan.profile).model_dump(mode="json"),
        "policy": load_policy().model_dump(mode="json"),
        "suite_config": suite_config(),
        "target_ids": ["a", "b"],
        "baseline_id": "base",
        "tests": "P01",
        "capability_run_id": "preflight",
        "replay_by_endpoint": {t.id: {"P01": [f"confirm-{t.id}"]} for t in targets[:2]},
    }
    seal_analysis(root, run, metadata)
    analysis = reconstruct(root / "data/raw", run)
    cases = []
    for target in targets[:2]:
        check = next(
            c for c in analysis.checks if c.check_id == "P01" and c.endpoint_id == target.id
        )
        assert check.verdict == "FAIL" and check.n == 4, check.model_dump_json()
        cases.append(
            CaseSelection(
                case_id="case-" + target.id,
                run_id=run,
                endpoint_id=target.id,
                suite_id="S1",
                check_id="P01",
                check_hash=content_hash(check.model_dump(mode="json")),
            )
        )
    selection = ComparisonSelection(sources=[pin_source(root, run)], cases=cases)
    return root, selection, root / ".cache/comparisons/reviewed"


def test_materialized_support_preserves_all_channels_and_population_counts(source):
    root, selection, out = source
    before = hashes(root / "data")
    result = prepare_sources(root, selection, out)
    assert result["state"] == "prepared_only" and result["owner_comparison_executed"] is False
    assert result["setup_seconds"] is None
    assert hashes(root / "data") == before
    verified = verify_sources(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert verified == result
    a, b = [
        json.loads((out / "cases" / (c.case_id + ".json")).read_text()) for c in selection.cases
    ]
    assert a["population_n"] == b["population_n"] == 4
    assert a["logical_unit_keys"] == b["logical_unit_keys"]
    assert a["recorded_attempts"] == len(a["observations"]) + 1
    assert {row["role"] for row in a["observations"]} == {
        "primary",
        "paired",
        "confirmation",
        "preflight",
    }
    assert all(row["endpoint_id"] in {"a", "base"} for row in a["observations"])
    assert all(row["endpoint_id"] in {"b", "base"} for row in b["observations"])
    assert not any(row["item_id"].startswith("P02") for row in a["observations"])
    assert {row["run_id"] for row in a["observations"]} == {"source", "preflight", "confirm-a"}
    assert all(row["record_hash"] for row in a["observations"])
    assert not (out / "docs/report/head_to_head.md").exists()
    copied = reconstruct(out / "sources" / safe_filename("source") / "data/raw", "source")
    original = reconstruct(root / "data/raw", "source")
    assert copied.checks == original.checks and copied.records == original.records
    assert any(r.sse_events and r.sse_events[0].raw_lines for r in copied.records)
    assert any(r.response["choices"][0]["message"].get("tool_calls") for r in copied.records)


def test_verified_replay_book_preserves_cases_population_and_source_scope(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    before = hashes(out)
    book = load_replay_book(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before
    assert {case["population_n"] for case in book.cases} == {4}
    assert all(case["replay_adds_independent_n"] is False for case in book.cases)
    assert {row.role for row in book.observations} == {
        "primary",
        "paired",
        "preflight",
        "confirmation",
    }
    for row in book.observations:
        assert row.record.endpoint_id in (
            {"a", "base"} if row.case_id == "case-a" else {"b", "base"}
        )
        assert row.record_hash == content_hash(row.record.model_dump(mode="json"))
    assert book.tape().audit()["consumed_attempts"] == 0
    assert book.tape().audit()["recorded_attempts"] == sum(
        c["recorded_attempts"] for c in book.cases
    )


def test_replay_book_requires_external_selection_anchor_and_untampered_inputs(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    with pytest.raises(ValueError):
        load_replay_book(out, selection_hash="0" * 64)
    case = out / "cases/case-a.json"
    case.write_bytes(case.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_replay_book(out, selection_hash=content_hash(selection.model_dump(mode="json")))


def test_selection_proposal_uses_actual_raw_cards_without_asserting_owner_approval(source):
    root, selection, out = source
    request = ComparisonRequest.model_validate(
        {"cases": [c.model_dump(exclude={"check_hash"}) for c in selection.cases]}
    )
    before = hashes(root)
    assert propose_selection(root, request) == selection
    assert hashes(root) == before and not out.exists()


def test_saved_selection_is_private_and_never_replaces_owner_content(source):
    root, selection, _ = source
    out = root / ".cache/comparisons/proposal.json"
    save_selection(root, selection, out)
    before = out.read_bytes()
    assert ComparisonSelection.model_validate_json(before) == selection
    if os.name == "posix":
        assert out.stat().st_mode & 0o077 == 0
    with pytest.raises(OSError):
        save_selection(root, selection, out)
    assert out.read_bytes() == before
    with pytest.raises(ValueError):
        save_selection(root, selection, root / "docs/selection.json")
    assert not (root / "docs").exists()


def test_actual_preparation_script_uses_pinned_selection_with_all_sockets_blocked(source):
    root, selection, out = source
    script = Path(__file__).resolve().parents[2] / "scripts/comparison_sources.py"
    selection_file = root / "reviewed.json"
    selection_file.write_text(selection.model_dump_json())
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "OPENROUTER_API_KEY",
            "SERVEGAUGE_OWNER_RUN",
            "TRUEWEIGHTS_OWNER_RUN",
        }
    }
    code = (
        "import socket,runpy,sys\n"
        "def blocked(*a,**k):raise RuntimeError('network_refused')\n"
        "class NoSockets(socket.socket):\n"
        " def __new__(cls,*a,**k):return blocked()\n"
        "socket.socket=NoSockets;socket.getaddrinfo=blocked\n"
        "script=sys.argv.pop(1);sys.argv[0]=script;runpy.run_path(script,run_name='__main__')\n"
    )

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code, str(script), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )

    args = ["--root", str(root), "--selection", str(selection_file), "--out", str(out)]
    prepared = run(["prepare", *args])
    assert prepared.returncode == 0, prepared.stderr
    before = hashes(out)
    verified = run(["verify", *args])
    assert verified.returncode == 0 and json.loads(verified.stdout) == json.loads(prepared.stdout)
    assert hashes(out) == before
    rejected = run(["PRIVATE_PARSER_VALUE"])
    assert rejected.returncode == 3 and "PRIVATE_PARSER_VALUE" not in rejected.stderr
    assert "Traceback" not in rejected.stderr
    request = {"cases": [c.model_dump(exclude={"check_hash"}) for c in selection.cases]}
    ambiguous = root / "ambiguous-request.json"
    ambiguous.write_text('{"cases":"PRIVATE_PARSER_VALUE",' + json.dumps(request)[1:])
    refused = run(["select", "--root", str(root), "--request", str(ambiguous)])
    assert refused.returncode == 3 and "PRIVATE_PARSER_VALUE" not in refused.stderr
    assert hashes(out) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint_id", "not-a-source"),
        ("suite_id", "S2"),
        ("run_id", "other"),
        ("check_id", "P02"),
        ("check_hash", "0" * 64),
    ],
)
def test_wrong_case_identity_refuses_without_creating_output(source, field, value):
    root, selection, out = source
    selection.cases[0] = selection.cases[0].model_copy(update={field: value})
    with pytest.raises(ValueError):
        prepare_sources(root, selection, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["raw", "context", "missing-confirmation", "extra-record"])
def test_changed_or_missing_pinned_support_refuses_before_writes(source, change):
    root, selection, out = source
    paths = sorted((root / "data/raw").rglob("*.jsonl.zst"))
    if change == "raw":
        paths[0].write_bytes(paths[0].read_bytes() + b"changed")
    elif change == "context":
        (root / "data/raw" / safe_filename("preflight") / "capture.json").write_text("{}")
    elif change == "missing-confirmation":
        paths[0].unlink()
    else:
        (paths[0].parent / "extra.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        prepare_sources(root, selection, out)
    assert not out.exists()


def test_existing_foreign_destination_and_public_destination_are_never_modified(source):
    root, selection, out = source
    out.mkdir(parents=True)
    (out / "foreign.txt").write_text("owner content")
    before = hashes(out)
    with pytest.raises(ValueError):
        prepare_sources(root, selection, out)
    assert hashes(out) == before
    with pytest.raises(ValueError):
        prepare_sources(root, selection, root / "docs/comparison")
    assert not (root / "docs").exists()


@pytest.mark.parametrize("change", ["case", "copied-raw", "extra", "symlink"])
def test_verification_detects_changed_outputs_read_only(source, change, tmp_path):
    root, selection, out = source
    prepare_sources(root, selection, out)
    if change == "case":
        (out / "cases/case-a.json").write_text("{}")
    elif change == "copied-raw":
        next((out / "sources").rglob("*.jsonl.zst")).write_bytes(b"changed")
    elif change == "extra":
        (out / "foreign.txt").write_text("changed")
    else:
        target = out / "cases/case-a.json"
        copy = tmp_path / "external.json"
        shutil.copyfile(target, copy)
        target.unlink()
        target.symlink_to(copy)
    before = hashes(out)
    with pytest.raises(ValueError):
        verify_sources(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before


def test_protected_and_symlink_paths_are_refused_before_content_access(source, tmp_path):
    root, selection, out = source
    with pytest.raises(ValueError):
        prepare_sources(root, selection, root / "Starter Files/forbidden")
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        pin_source(linked, "source")
    assert not out.exists()


def test_rewritten_manifest_cannot_supply_its_own_selection_anchor(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selection_hash"] = "0" * 64
    path.write_text(json.dumps(manifest))
    before = hashes(out)
    with pytest.raises(ValueError):
        verify_sources(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before


@pytest.mark.parametrize("field", ["sources", "cases"])
def test_mutated_selection_must_be_revalidated_before_use(source, field):
    root, selection, out = source
    if field == "sources":
        selection.sources.append(selection.sources[0])
    else:
        selection.cases[1] = selection.cases[0]
    with pytest.raises(ValueError):
        prepare_sources(root, selection, out)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["preflight", "paired", "duplicate"])
def test_self_consistent_new_pins_do_not_hide_missing_or_ambiguous_support(source, kind):
    import zstandard as zstd

    from servegauge.core.hashing import canonical_json
    from servegauge.storage.raw_store import read_file

    root, selection, out = source
    raw = root / "data/raw"
    source_run = "preflight" if kind == "preflight" else "source"
    endpoint_id = "a" if kind in {"preflight", "duplicate"} else "base"
    directory = raw / safe_filename(source_run) / safe_filename(endpoint_id)
    path = next(directory.glob("*.jsonl.zst"))
    rows = list(read_file(path))
    removed = rows.pop(0)
    if kind == "duplicate":
        rows += [removed, removed]
    path.write_bytes(
        zstd.ZstdCompressor().compress(
            b"".join(canonical_json(r.model_dump(mode="json")) + b"\n" for r in rows)
        )
    )
    seal_path = raw / safe_filename("source") / "analysis.json"
    envelope = json.loads(seal_path.read_text())
    if kind != "duplicate":
        del envelope["value"]["record_hashes"][removed.exchange_id]
        envelope["hash"] = content_hash(envelope["value"])
        seal_path.write_bytes(canonical_json(envelope) + b"\n")
    selection.sources[0] = pin_source(root, "source")
    with pytest.raises(ValueError, match="preflight|paired|duplicate"):
        prepare_sources(root, selection, out)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["missing", "wrong-endpoint"])
def test_agent_trace_claims_require_actual_scoped_support_even_with_new_pins(source, kind):
    from servegauge.core.hashing import canonical_json

    root, selection, out = source
    raw = root / "data/raw"
    analysis = reconstruct(raw, "source")
    record = next(r for r in analysis.records if r.run_id == "source" and r.endpoint_id == "a")
    path = raw / safe_filename("source") / "analysis.json"
    envelope = json.loads(path.read_text())
    envelope["value"]["metadata"]["agent_traces"] = {
        "b": {
            "fixture": {
                "run_ids": ["source"],
                "exchange_ids": ["missing" if kind == "missing" else record.exchange_id],
            }
        }
    }
    envelope["hash"] = content_hash(envelope["value"])
    path.write_bytes(canonical_json(envelope) + b"\n")
    selection.sources[0] = pin_source(root, "source")
    with pytest.raises(ValueError, match="agent_support"):
        prepare_sources(root, selection, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "channel",
    [
        "response",
        "attempt",
        "split-sse",
        "byte-fragments",
        "escaped-context",
        "sse-logprobs",
        "escaped-sse",
        "sse-tool-arguments",
    ],
)
def test_decoded_credentials_refuse_even_when_compressed_bytes_and_pins_look_clean(source, channel):
    import zstandard as zstd

    from servegauge.core.hashing import canonical_json
    from servegauge.core.models import SseEvent
    from servegauge.core.secrets import scan_paths
    from servegauge.storage.raw_store import read_file

    root, selection, out = source
    raw = root / "data/raw"
    fake = "s" + "k-" + "A" * 40  # Fake-pattern adversarial input, never a real credential.
    seal_path = raw / safe_filename("source") / "analysis.json"
    seal = json.loads(seal_path.read_text())
    if channel == "escaped-context":
        path = raw / safe_filename("source") / "capture.json"
        envelope = json.loads(path.read_text())
        envelope["value"]["manifest"]["opaque_value"] = fake
        envelope["hash"] = content_hash(envelope["value"])
        seal["value"]["capture_context_hash"] = content_hash(envelope["value"])
        escaped = "".join(f"\\u{ord(c):04x}" for c in fake)
        path.write_bytes(canonical_json(envelope).replace(fake.encode(), escaped.encode()) + b"\n")
    else:
        path = next((raw / safe_filename("source") / safe_filename("a")).glob("*.jsonl.zst"))
        rows = list(read_file(path))
        row = next(r for r in rows if r.item_id.startswith("P02"))
        if channel == "response":
            row.response["opaque_value"] = fake
        elif channel == "attempt":
            row.attempts[0].response = {"opaque_value": fake}
        elif channel == "byte-fragments":
            row.response["choices"][0]["logprobs"] = {
                "content": [
                    {"token": "safe", "bytes": list(part.encode())} for part in [fake[:6], fake[6:]]
                ]
            }
        elif channel in {"sse-logprobs", "escaped-sse", "sse-tool-arguments"}:
            row.response = None
            row.attempts[0].response = None
            if channel == "escaped-sse":
                data = canonical_json({"opaque_value": fake}).decode()
                escaped = "".join(f"\\u{ord(c):04x}" for c in fake)
                row.sse_events = [SseEvent(t_ms=0, data=data.replace(fake, escaped))]
            elif channel == "sse-logprobs":
                row.sse_events = [
                    SseEvent(
                        t_ms=i,
                        data=json.dumps(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "logprobs": {
                                            "content": [
                                                {"token": "safe", "bytes": list(part.encode())}
                                            ]
                                        },
                                    }
                                ]
                            }
                        ),
                    )
                    for i, part in enumerate([fake[:6], fake[6:]])
                ]
            else:
                arguments = json.dumps({"opaque_value": fake})
                split = arguments.index(fake) + 6
                row.sse_events = [
                    SseEvent(
                        t_ms=i,
                        data=json.dumps(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {"index": 0, "function": {"arguments": part}}
                                            ]
                                        },
                                    }
                                ]
                            }
                        ),
                    )
                    for i, part in enumerate([arguments[:split], arguments[split:]])
                ]
        else:
            row.response = None
            row.attempts[0].response = None
            row.sse_events = [
                SseEvent(
                    t_ms=i, data=json.dumps({"choices": [{"index": 0, "delta": {"content": part}}]})
                )
                for i, part in enumerate([fake[:6], fake[6:]])
            ]
        seal["value"]["record_hashes"][row.exchange_id] = content_hash(row.model_dump(mode="json"))
        path.write_bytes(
            zstd.ZstdCompressor().compress(
                b"".join(canonical_json(r.model_dump(mode="json")) + b"\n" for r in rows)
            )
        )
    seal["hash"] = content_hash(seal["value"])
    seal_path.write_bytes(canonical_json(seal) + b"\n")
    assert not scan_paths([path]), "Fixture must exercise decoded-only detection"
    selection.sources[0].files_sha256 = {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
        for name in selection.sources[0].files_sha256
    }
    before = hashes(root / "data")
    with pytest.raises(ValueError, match="credential"):
        prepare_sources(root, selection, out)
    assert hashes(root / "data") == before and not out.exists()


def test_credential_like_selection_metadata_cannot_be_saved_or_copied(source):
    root, selection, out = source
    selection.cases[0].case_id = "s" + "k-" + "a" * 40
    request = ComparisonRequest.model_validate(
        {"cases": [c.model_dump(exclude={"check_hash"}) for c in selection.cases]}
    )
    before = hashes(root)
    with pytest.raises(ValueError, match="credential"):
        propose_selection(root, request)
    with pytest.raises(ValueError, match="credential"):
        save_selection(root, selection, root / ".cache/comparisons/proposal.json")
    with pytest.raises(ValueError, match="credential"):
        prepare_sources(root, selection, out)
    assert hashes(root) == before and not out.exists()


def test_undecodable_sse_privacy_scope_refuses_without_dropping_source_data(source):
    import zstandard as zstd

    from servegauge.core.hashing import canonical_json
    from servegauge.core.models import SseEvent
    from servegauge.storage.raw_store import read_file

    root, selection, out = source
    raw = root / "data/raw"
    path = next((raw / safe_filename("source") / safe_filename("a")).glob("*.jsonl.zst"))
    rows = list(read_file(path))
    row = next(r for r in rows if r.item_id.startswith("P02"))
    row.sse_events = [SseEvent(t_ms=0, data="not-json")]
    path.write_bytes(
        zstd.ZstdCompressor().compress(
            b"".join(canonical_json(r.model_dump(mode="json")) + b"\n" for r in rows)
        )
    )
    seal_path = raw / safe_filename("source") / "analysis.json"
    seal = json.loads(seal_path.read_text())
    seal["value"]["record_hashes"][row.exchange_id] = content_hash(row.model_dump(mode="json"))
    seal["hash"] = content_hash(seal["value"])
    seal_path.write_bytes(canonical_json(seal) + b"\n")
    selection.sources[0] = pin_source(root, "source")
    before = hashes(root / "data")
    with pytest.raises(ValueError, match="sse_privacy_decoding"):
        prepare_sources(root, selection, out)
    assert hashes(root / "data") == before and not out.exists()


@pytest.mark.parametrize("channel", ["record", "context", "sse"])
def test_duplicate_json_keys_cannot_hide_decoded_private_values(source, channel):
    import zstandard as zstd

    from servegauge.core.hashing import canonical_json
    from servegauge.core.models import SseEvent
    from servegauge.storage.raw_store import read_file

    root, selection, out = source
    raw = root / "data/raw"
    fake = "s" + "k-" + "A" * 40
    escaped = "".join(f"\\u{ord(c):04x}" for c in fake)
    hidden = ('"opaque_value":"' + escaped + '","opaque_value":"clean"').encode()
    ordinary = b'"opaque_value":"clean"'
    seal_path = raw / safe_filename("source") / "analysis.json"
    seal = json.loads(seal_path.read_text())
    if channel == "context":
        path = raw / safe_filename("source") / "capture.json"
        envelope = json.loads(path.read_text())
        envelope["value"]["manifest"]["opaque_value"] = "clean"
        envelope["hash"] = content_hash(envelope["value"])
        seal["value"]["capture_context_hash"] = content_hash(envelope["value"])
        path.write_bytes(canonical_json(envelope).replace(ordinary, hidden) + b"\n")
    else:
        path = next((raw / safe_filename("source") / safe_filename("a")).glob("*.jsonl.zst"))
        rows = list(read_file(path))
        row = next(r for r in rows if r.item_id.startswith("P02"))
        if channel == "record":
            row.response["opaque_value"] = "clean"
        else:
            row.response = None
            row.sse_events = [SseEvent(t_ms=0, data="{" + hidden.decode() + "}")]
        seal["value"]["record_hashes"][row.exchange_id] = content_hash(row.model_dump(mode="json"))
        encoded = b"".join(canonical_json(r.model_dump(mode="json")) + b"\n" for r in rows)
        if channel == "record":
            assert encoded.count(ordinary) == 1
            encoded = encoded.replace(ordinary, hidden)
        path.write_bytes(zstd.ZstdCompressor().compress(encoded))
    seal["hash"] = content_hash(seal["value"])
    seal_path.write_bytes(canonical_json(seal) + b"\n")
    selection.sources[0].files_sha256 = {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
        for name in selection.sources[0].files_sha256
    }
    before = hashes(root / "data")
    with pytest.raises(ValueError, match="duplicate|sse_privacy_decoding"):
        prepare_sources(root, selection, out)
    assert hashes(root / "data") == before and not out.exists()


@pytest.mark.parametrize("name", ["manifest.json", "selection.json", "cases/case-a.json"])
def test_self_consistent_output_hashes_do_not_approve_ambiguous_json(source, name):
    from servegauge.core.hashing import canonical_json

    root, selection, out = source
    prepare_sources(root, selection, out)
    path = out / name
    value = json.loads(path.read_text())
    original = b'"version":' + canonical_json(value["version"])
    encoded = canonical_json(value)
    assert encoded.count(original) == 1
    path.write_bytes(encoded.replace(original, b'"version":"ambiguous",' + original) + b"\n")
    if name != "manifest.json":
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    before = hashes(out)
    with pytest.raises(ValueError, match="duplicate"):
        verify_sources(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    assert hashes(out) == before
