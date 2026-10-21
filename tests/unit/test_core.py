import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import exchange_id, safe_filename
from servegauge.core.manifest import build_manifest
from servegauge.core.models import ChatRequest, EndpointSpec, Item, ModelProfile, SuiteConfig
from servegauge.core.redaction import Redactor
from servegauge.core.secrets import scan_repository


def test_canonical_hashes_and_safe_ids() -> None:
    assert canonical_json({"z": 1, "a": "雨"}) == '{"a":"雨","z":1}'.encode()
    assert content_hash({"a": 2, "b": 1}) == content_hash({"b": 1, "a": 2})
    assert exchange_id("ab", "c", "x", "y", 1) != exchange_id("a", "bc", "x", "y", 1)
    path = safe_filename("or:deepinfra/turbo?x")
    assert ":" not in path and "/" not in path and "?" not in path
    assert safe_filename("a:b") != safe_filename("a/b")
    assert len(safe_filename("x" * 300)) <= 109
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


def test_binding_models_and_safe_validation() -> None:
    endpoint = EndpointSpec(
        id="one",
        kind="mock",
        base_url="http://127.0.0.1/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
    )
    assert endpoint.api_key_env is None
    assert ChatRequest(messages=[]).stream is False
    item = Item(item_id="a", suite_id="S1", version="1", payload={}, expected={})
    assert item.license == "CC-BY-4.0"
    assert SuiteConfig().level == "smoke"
    assert ModelProfile(id="x", weights="x").unverified == []
    with pytest.raises(ValidationError) as exc:
        EndpointSpec(id="sensitive-input", kind="bad")
    assert "sensitive-input" not in str(exc.value)


def test_manifest_reproducible_hashes_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_SECRET", "sensitive-value")
    config = {"password": "sensitive-value", "safe": 1}
    first = build_manifest("run-a", config, {"S1": "abc"}, [], redactor=Redactor.from_environment())
    assert first["config_hash"] == content_hash({"password": "[REDACTED]", "safe": 1})
    assert "sensitive-value" not in json.dumps(first)
    assert first["git_sha"]
    assert first["packages"]["servegauge"]


def test_scan_tracked_and_untracked_and_cassettes(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    path = tmp_path / "tests/cassettes/private.json"
    path.parent.mkdir(parents=True)
    path.write_text("sk-" + "credential" * 6)
    assert len(scan_repository(tmp_path)) == 1
