"""Finite experiment recipes preserve the original self-host contract and budget path."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from unit.test_selfhost_plan import policy

from servegauge.core.hashing import canonical_json
from servegauge.runner.budget import BudgetPolicy
from servegauge.selfhost.plan import (
    SessionConfig,
    build_plan,
    engine_args,
    parse_session_config,
    validate_plan,
)
from servegauge.selfhost.study_recipe import (
    ENGINE_REVISIONS,
    MODEL,
    REVISION,
    STUDY_PROFILE,
    VARIANTS,
    WRONG_TEMPLATE,
    StudySessionConfig,
    study_template,
)


def reviewed(variant="bf16", backend="ssh", **changes):
    value = study_template(backend, variant)
    value.update(
        image="vllm/vllm-openai@sha256:" + "a" * 64,
        hourly_usd="4.20",
        setup_max_usd="0.25",
        billing_basis_reviewed=True,
        recipe_reviewed=True,
        dedicated_host_reviewed=backend == "ssh",
    )
    if backend == "ssh":
        value["ssh_host"] = "owner@gpu.example.test"
    else:
        value.update(
            modal_app="servegauge-study",
            modal_image_id="im-fixture",
            inbound_cidrs=["198.51.100.7/32"],
        )
    return StudySessionConfig.model_validate(value | changes)


@pytest.mark.parametrize("backend", ["ssh", "modal"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_incomplete_study_templates_cannot_launch(backend, variant):
    value = study_template(backend, variant)
    assert value["version"] == "selfhost-study-config-v1"
    assert value["purpose"] == "perturbation-study"
    assert value["profile"] == STUDY_PROFILE
    assert value["model_repository"] == MODEL and value["model_revision"] == REVISION
    assert value["engine_revision"] == ENGINE_REVISIONS[value["engine_version"]]
    assert value["image"] is value["hourly_usd"] is None
    assert value["billing_basis_reviewed"] is value["recipe_reviewed"] is False
    with pytest.raises(ValueError):
        StudySessionConfig.model_validate(value)


@pytest.mark.parametrize("variant", VARIANTS)
def test_recipe_has_exact_version_pins_precision_and_template_arguments(variant):
    configuration = reviewed(variant)
    plan = build_plan(configuration, policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    assert plan["version"] == "selfhost-study-plan-v1"
    assert plan["reference_qualification"] == "not_a_reference_session"
    assert plan["reserved_upper_usd"] == "1.02"
    assert plan["maximum_duration_seconds"] == 660
    assert plan["engine_args"] == engine_args(configuration)
    validate_plan(plan)
    args = plan["engine_args"]
    assert args[:3] == ["vllm", "serve", MODEL]
    assert args[args.index("--revision") + 1] == REVISION
    assert args[args.index("--tokenizer-revision") + 1] == REVISION
    assert args[args.index("--served-model-name") + 1] == STUDY_PROFILE
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--kv-cache-dtype") + 1] == "auto"
    assert json.loads(args[args.index("--default-chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    if variant == "fp8":
        assert args[args.index("--quantization") + 1] == "fp8_per_tensor"
    elif variant == "mxfp4":
        assert args[args.index("--quantization") + 1] == "mxfp4"
        assert args[args.index("--linear-backend") + 1] == "marlin"
    else:
        assert "--quantization" not in args
    if variant == "wrong-template":
        assert args[args.index("--chat-template") + 1] == WRONG_TEMPLATE
    else:
        assert "--chat-template" not in args
    assert "--api-key" not in args and "R2" not in plan.get("reference_tier", "")


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "reference"},
        {"profile": "gpt-oss-120b"},
        {"model_repository": "unreviewed/model"},
        {"model_revision": "main"},
        {"engine_version": "0.30.0"},
        {"engine_revision": "0" * 40},
        {"precision": "int4"},
        {"template": "unreviewed"},
        {"precision": "fp8", "template": "wrong-v1"},
        {"extra_args": ["--trust-remote-code"]},
        {"image": "vllm/vllm-openai:latest"},
        {"api_key_env": "OPENROUTER_API_KEY"},
        {"api_key_env": "SERVEGAUGE_OWNER_RUN"},
        {"gpu": "H100:8"},
        {"hourly_usd": "NaN"},
        {"hourly_usd": "0"},
        {"recipe_reviewed": False},
        {"billing_basis_reviewed": False},
        {"dedicated_host_reviewed": False},
        {"ssh_host": "host; injected"},
    ],
)
def test_unreviewed_or_arbitrary_recipe_changes_refuse(change):
    with pytest.raises(ValueError):
        reviewed(**change)


def test_study_still_requires_positive_caps_and_shared_owner_policy():
    for amount, limits in [("0", {}), ("0.5", {}), ("2", {"selfhost_usd_per_session": 0})]:
        with pytest.raises(ValueError):
            build_plan(reviewed(), policy(**limits), max_usd=Decimal(amount), shutdown_minutes=10)


def test_every_saved_v1_plan_reconstructs_with_exact_bytes():
    fixture = json.loads(Path("tests/fixtures/selfhost-v1-plans.json").read_bytes())
    assert len(fixture["cases"]) == 16
    for case in fixture["cases"]:
        configuration = parse_session_config(case["configuration"])
        assert type(configuration) is SessionConfig
        actual = build_plan(
            configuration,
            BudgetPolicy.model_validate(case["policy"]),
            max_usd=Decimal(case["max_usd"]),
            shutdown_minutes=case["shutdown_minutes"],
        )
        assert canonical_json(actual) == canonical_json(case["plan"])
        validate_plan(actual)


def test_version_dispatch_rejects_unknown_versions_without_guessing():
    value = reviewed().model_dump(mode="json")
    assert isinstance(parse_session_config(value), StudySessionConfig)
    with pytest.raises(ValueError):
        parse_session_config(value | {"version": "selfhost-config-v900"})


@pytest.mark.parametrize("value", [None, [], 3, "invalid"])
def test_configuration_dispatch_requires_an_object(value):
    with pytest.raises(ValueError):
        parse_session_config(value)
