"""Self-host costs and pinned launch arguments are reviewed before any side effect."""

from decimal import Decimal

import pytest

from servegauge.runner.budget import BudgetPolicy
from servegauge.selfhost.plan import SessionConfig, build_plan, engine_args


def config(**changes):
    value = {
        "version": "selfhost-config-v1",
        "backend": "ssh",
        "profile": "gpt-oss-120b",
        "engine_version": "0.11.0",
        "image": "vllm/vllm-openai@sha256:" + "a" * 64,
        "model_revision": "b" * 40,
        "gpu": "H100",
        "hourly_usd": "4.20",
        "setup_max_usd": "0.25",
        "shutdown_grace_seconds": 60,
        "billing_basis_reviewed": True,
        "recipe_reviewed": True,
        "dedicated_host_reviewed": True,
        "api_key_env": "SELFHOST_API_KEY",
        "ssh_host": "owner@gpu.example.test",
        "ssh_key_env": "SELFHOST_SSH_KEY_PATH",
        "ssh_port": 22,
    }
    return SessionConfig.model_validate(value | changes)


def policy(**changes):
    return BudgetPolicy.model_validate(
        {
            "monthly_usd": 50,
            "per_run_usd": 25,
            "dev_live_usd": 5,
            "selfhost_usd_per_session": 20,
            "stop_at_fraction": "0.9",
        }
        | changes
    )


def test_exact_cost_bound_includes_grace_and_setup_without_float_rounding():
    plan = build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    assert plan["maximum_duration_seconds"] == 660
    assert plan["reserved_upper_usd"] == "1.02"
    assert plan["gpu_compute_upper_usd"] == "0.77"
    assert plan["billing_status"] == "declared_bound_not_invoice"
    assert plan["reference_qualification"] == "pending_owner_capture"
    assert plan["profile"] == "gpt-oss-120b"
    assert plan["engine_args"] == engine_args(config())
    assert "R2" not in plan.get("reference_tier", "")


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "0", "21", "0.5"])
def test_nonfinite_or_insufficient_or_excessive_session_cap_refuses(amount):
    with pytest.raises(ValueError):
        build_plan(config(), policy(), max_usd=Decimal(amount), shutdown_minutes=10)


@pytest.mark.parametrize("minutes", [0, -1, 1441, True, 1.5])
def test_shutdown_must_be_a_bounded_positive_integer(minutes):
    with pytest.raises(ValueError):
        build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=minutes)


def test_zero_installed_budget_and_shared_policy_caps_refuse():
    for limits in [
        {"selfhost_usd_per_session": 0},
        {"per_run_usd": 0},
        {"monthly_usd": 0},
    ]:
        with pytest.raises(ValueError):
            build_plan(config(), policy(**limits), max_usd=Decimal("2"), shutdown_minutes=10)


def test_stopping_fraction_applies_to_declared_session_cap():
    # The session has room in the gross cap but not the admission stopping amount.
    with pytest.raises(ValueError):
        build_plan(config(), policy(), max_usd=Decimal("1.10"), shutdown_minutes=10)


@pytest.mark.parametrize(
    "change",
    [
        {"image": "vllm/vllm-openai:latest"},
        {"image": "other/image@sha256:" + "a" * 64},
        {"model_revision": "main"},
        {"engine_version": "0.10.0"},
        {"engine_version": "latest"},
        {"hourly_usd": "NaN"},
        {"hourly_usd": "0"},
        {"setup_max_usd": "-1"},
        {"billing_basis_reviewed": False},
        {"recipe_reviewed": False},
        {"dedicated_host_reviewed": False},
        {"profile": "../other"},
        {"gpu": "H100:8"},
        {"api_key_env": "TOKEN=secret"},
        {"api_key_env": "OPENROUTER_API_KEY"},
        {"ssh_host": "-oProxyCommand=evil"},
        {"ssh_host": "owner@gpu; touch anything"},
        {"ssh_key_env": "/Users/owner/.ssh/id_key"},
        {"ssh_port": 0},
        {"shutdown_grace_seconds": -1},
        {"unknown": "ignored"},
    ],
)
def test_unreviewed_pins_rates_credentials_and_injected_arguments_refuse(change):
    with pytest.raises(ValueError):
        config(**change)


def test_engine_command_has_exact_pins_tool_flags_and_auth_environment_only():
    args = engine_args(config())
    assert args[:2] == ["vllm", "serve"]
    assert "openai/gpt-oss-120b" in args
    assert args[args.index("--revision") + 1] == "b" * 40
    assert args[args.index("--tokenizer-revision") + 1] == "b" * 40
    assert args[args.index("--tool-call-parser") + 1] == "openai"
    assert "--enable-auto-tool-choice" in args
    assert "--api-key" not in args
    assert args[args.index("--tensor-parallel-size") + 1] == "1"


def test_modal_requires_app_and_does_not_accept_ssh_destination():
    common = {"backend": "modal", "ssh_host": None, "ssh_key_env": None}
    with pytest.raises(ValueError):
        config(**common)
    value = config(
        **common,
        modal_app="servegauge-owner",
        modal_image_id="im-fixture",
        inbound_cidrs=["198.51.100.7/32"],
    )
    assert value.modal_app == "servegauge-owner"
    with pytest.raises(ValueError):
        config(backend="modal", modal_app="servegauge-owner")


@pytest.mark.parametrize("cidrs", [[], ["0.0.0.0/0"], ["::/0"], ["not-a-cidr"], ["224.0.0.1/32"]])
def test_modal_never_prepares_public_or_ambiguous_inbound_access(cidrs):
    with pytest.raises(ValueError):
        config(
            backend="modal",
            ssh_host=None,
            ssh_key_env=None,
            modal_app="servegauge-owner",
            modal_image_id="im-fixture",
            inbound_cidrs=cidrs,
        )


def test_prepare_does_not_read_credentials_or_touch_the_environment(monkeypatch):
    import os

    monkeypatch.setenv("SELFHOST_API_KEY", "local-fixture-credential")
    before = dict(os.environ)
    plan = build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    assert "local-fixture-credential" not in str(plan)
    assert dict(os.environ) == before
