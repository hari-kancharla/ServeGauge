import copy
import hashlib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from servegauge.core.errors import SafeError
from servegauge.core.models import ReferencePack
from servegauge.refs.pack import reference_for
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import evaluation_configuration
from servegauge.runner.plan import RunPlan
from servegauge.storage.accounting_identity import COUNTER_ROOTS, accounting_counter_identity
from servegauge.suites.s7_accounting import accounting_items
from servegauge.tokencount.glm_assets import FILES, MODEL, REVISION
from servegauge.verdicts.config import load_policy


def configuration(profile="glm-5.3"):
    plan = RunPlan(
        plan_id="model-b-contract",
        profile=profile,
        level="core",
        seed=43,
        reference={"tier": "BASELINE"},
        max_usd=0,
        endpoints=[resolve_endpoint("http://127.0.0.1:8011/v1", "mock", profile)],
        suites=["S7"],
    )
    return evaluation_configuration(plan, accounting_items("core", 43, {"items": 1}), load_policy())


def test_counter_identity_contains_actual_pins_runtime_and_no_provider_claim():
    identity = accounting_counter_identity("glm-5.3")
    assert identity is not None
    assert identity["hf_model"] == MODEL and identity["hf_revision"] == REVISION
    assert identity["files_sha256"] == FILES
    assert identity["runtime_versions"]
    assert identity["provider_rendering_verified"] is False
    assert identity["billing_correctness_proven"] is False
    assert identity["implementation_roots"] == list(COUNTER_ROOTS)
    for name, digest in identity["implementation_sha256"].items():
        assert digest == hashlib.sha256(Path(name).read_bytes()).hexdigest()


def test_glm_prompt_completion_and_hint_references_pin_the_actual_local_counter():
    cfg = configuration()
    identity = accounting_counter_identity("glm-5.3")
    assert identity is not None
    for check, value in cfg["checks"].items():
        if check.startswith(("B01", "B02", "B05")):
            assert value["local_counter"] == identity
        else:
            assert "local_counter" not in value
        if check.startswith(("B01", "B05")):
            assert value["counter"]["counter_version"] == identity["counter_version"]


def test_historical_gpt_oss_configuration_does_not_acquire_any_new_counter_fields(monkeypatch):
    import servegauge.storage.accounting_identity as identity

    def forbidden(*args, **kwargs):
        raise AssertionError("no counter file/runtime reads for gpt-oss")

    monkeypatch.setattr(identity, "generator_manifest", forbidden)
    monkeypatch.setattr(identity, "version", forbidden)
    assert accounting_counter_identity("gpt-oss-120b") is None
    cfg = configuration("gpt-oss-120b")
    for value in cfg["checks"].values():
        assert "local_counter" not in value
        if "counter" in value:
            assert value["counter"] == {
                "counter_date": "2026-10-18",
                "counter_version": "hf-api-adapter-v1",
            }


@pytest.mark.parametrize("failure", ["file", "runtime", "incomplete"])
def test_unavailable_identity_fails_safely_before_a_comparison(failure, monkeypatch):
    import servegauge.storage.accounting_identity as identity

    def missing(*args, **kwargs):
        if failure == "file":
            raise OSError("upstream/private/path")
        if failure == "incomplete":
            return {"complete": False}
        raise PackageNotFoundError("private-runtime-detail")

    monkeypatch.setattr(
        identity, "version" if failure == "runtime" else "generator_manifest", missing
    )
    with pytest.raises(SafeError, match="^accounting_counter_identity_unavailable$"):
        accounting_counter_identity("glm-5.3")


@pytest.mark.parametrize("check", ["B01", "B02", "B05"])
@pytest.mark.parametrize("change", ["missing", "revision", "asset", "code", "runtime"])
def test_changed_counter_identity_cannot_reuse_a_saved_reference(check, change):
    cfg = configuration()
    pack = ReferencePack(
        pack_id="original", profile="glm-5.3", tier="BASELINE", manifest={"configuration": cfg}
    )
    different = copy.deepcopy(cfg)
    counter = different["checks"][check]
    if change == "missing":
        counter.pop("local_counter")
    else:
        identity = counter["local_counter"]
        if change == "revision":
            identity["hf_revision"] = "0" * 40
        elif change == "asset":
            identity["files_sha256"]["tokenizer.json"] = "0" * 64
        elif change == "code":
            identity["implementation_sha256"][next(iter(identity["implementation_sha256"]))] = (
                "0" * 64
            )
        else:
            identity["runtime_versions"]["tokenizers"] = "unknown"
    with pytest.raises(ValueError, match="reference_configuration_mismatch"):
        reference_for(pack, target="target", check=check, item_ids=[], configuration=different)
