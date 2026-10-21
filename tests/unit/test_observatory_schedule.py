"""UTC scheduling/configuration contracts; no clocks, sleeps or external accounts."""

from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from servegauge.observatory.schedule import (
    ObservatoryConfig,
    cadence_window,
    load_observatory,
    window_id,
)


@pytest.mark.parametrize(
    "cadence,stamp,start,end",
    [
        (
            "probe",
            "2026-10-19T05:59:59+00:00",
            "2026-10-19T00:00:00+00:00",
            "2026-10-19T06:00:00+00:00",
        ),
        (
            "daily",
            "2026-10-19T23:59:59+00:00",
            "2026-10-19T00:00:00+00:00",
            "2026-10-20T00:00:00+00:00",
        ),
        (
            "weekly",
            "2026-01-01T12:00:00+00:00",
            "2025-12-29T00:00:00+00:00",
            "2026-01-05T00:00:00+00:00",
        ),
        (
            "monthly",
            "2026-12-31T23:59:59+00:00",
            "2026-12-01T00:00:00+00:00",
            "2027-01-01T00:00:00+00:00",
        ),
    ],
)
def test_windows_are_aligned_to_UTC_calendar_boundaries(cadence, stamp, start, end):
    observed = cadence_window(cadence, datetime.fromisoformat(stamp))
    assert observed.start.isoformat() == start and observed.end.isoformat() == end
    assert observed.start.tzinfo == UTC and observed.end.tzinfo == UTC


def test_equivalent_offsets_keep_identity_and_boundary_changes_identity():
    a = cadence_window("daily", datetime.fromisoformat("2026-10-19T20:00:00-04:00"))
    b = cadence_window("daily", datetime.fromisoformat("2026-10-20T00:00:00+00:00"))
    previous = cadence_window("daily", datetime.fromisoformat("2026-10-19T23:59:59+00:00"))
    assert a == b and window_id("demo", a) == window_id("demo", b)
    assert window_id("demo", a) != window_id("demo", previous)
    assert window_id("other", a) != window_id("demo", a)


def test_naive_time_and_unknown_cadence_refuse():
    with pytest.raises(ValueError):
        cadence_window("daily", datetime(2026, 10, 20))
    with pytest.raises(ValueError):
        cadence_window("unknown", datetime(2026, 10, 20, tzinfo=UTC))


def test_disabled_default_has_no_implicit_models_or_publishing():
    cfg = ObservatoryConfig(id="empty")
    assert not cfg.enabled and not cfg.models and not cfg.publishing.enabled
    assert set(cfg.cadences) == {"probe", "daily", "weekly", "monthly"}
    assert cfg.cadences["probe"].tests == ["FP1", "FP3"]
    assert cfg.cadences["daily"].level == "core"
    assert cfg.cadences["monthly"].level == "full"


@pytest.mark.parametrize(
    "change",
    [
        {"enabled": True},
        {"mode": "unknown"},
        {"budgets": {"daily": -1}},
        {"budgets": {"daily": float("nan")}},
        {"publishing": {"credential": "do-not-accept-values"}},
        {"cadences": {"daily": {"level": "smoke", "tests": ["not-a-check"]}}},
    ],
)
def test_invalid_or_dangerous_configuration_is_rejected(change):
    with pytest.raises(ValidationError):
        ObservatoryConfig(id="invalid", **change)


def test_loader_never_echoes_invalid_configuration_contents(tmp_path):
    private = "PRIVATE_OBSERVATORY_CONFIG_CANARY"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"id": "bad", "unexpected": private}))
    with pytest.raises(ValueError) as caught:
        load_observatory(path)
    assert private not in str(caught.value)


def test_mock_mode_cannot_hold_a_real_endpoint_or_credentials():
    endpoint = {
        "id": "real",
        "kind": "openai_compat",
        "base_url": "https://example.test/v1",
        "model": "example",
        "claimed_profile": "gpt-oss-120b",
        "api_key_env": "EXAMPLE_KEY",
    }
    with pytest.raises(ValidationError):
        ObservatoryConfig(
            id="bad",
            enabled=True,
            mode="mock",
            models=[{"id": "model", "profile": "gpt-oss-120b", "endpoints": [endpoint]}],
        )


def test_saved_schema_and_examples_match_typed_configuration():
    import json
    from pathlib import Path

    schema = ObservatoryConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    assert json.loads(Path("configs/observatory.schema.json").read_text()) == schema
    assert not load_observatory(Path("configs/observatory.yaml")).enabled
    cfg = load_observatory(Path("configs/observatory.mock.yaml"))
    assert cfg.enabled and cfg.mode == "mock" and not cfg.publishing.enabled
    assert cfg.cadences["daily"].fingerprint_level == "full"
    assert cfg.cadences["monthly"].overrides["S6"]["methods"] == ["FP1", "FP2", "FP3", "FP4"]


@pytest.mark.parametrize("length", [True, 0, 255, 131073])
def test_mock_context_override_refuses_unsupported_bounds(length):
    from pathlib import Path

    cfg = load_observatory(Path("configs/observatory.mock.yaml")).model_dump(mode="json")
    cfg["models"][0]["mock_context_length"] = length
    with pytest.raises(ValidationError):
        ObservatoryConfig.model_validate(cfg)


async def test_mock_context_override_changes_actual_metadata_and_default_is_preserved():
    from pathlib import Path

    from servegauge.observatory.tick import _transports
    from servegauge.runner.check import _mock_metadata
    from servegauge.runner.plan import RunPlan

    cfg = load_observatory(Path("configs/observatory.mock.yaml"))
    model = cfg.models[0]
    for override, expected in [(None, 131072), (512, 512)]:
        model.mock_context_length = override
        plan = RunPlan(
            plan_id="context",
            profile=model.profile,
            level="core",
            seed=17,
            reference={"tier": "MOCK"},
            max_usd=0,
            endpoints=model.endpoints,
            suites=["S0"],
        )
        await _mock_metadata(plan, _transports(model, 17, "mock"))
        assert all(e.metadata["context_length"] == expected for e in plan.endpoints)


def test_real_mode_cannot_accept_a_mock_context_declaration():
    from pathlib import Path

    cfg = load_observatory(Path("configs/observatory.mock.yaml")).model_dump(mode="json")
    cfg["mode"] = "owner"
    for model in cfg["models"]:
        model["faults"] = None
        model["reference"] = {"tier": "R3"}
        model["mock_context_length"] = 512
        for endpoint in model["endpoints"]:
            endpoint["kind"] = "openai_compat"
            endpoint["base_url"] = "https://example.test/v1"
    with pytest.raises(ValidationError, match="mock_only"):
        ObservatoryConfig.model_validate(cfg)
