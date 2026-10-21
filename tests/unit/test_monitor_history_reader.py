"""Sealed metadata barriers; candidate inspection never collects missing evidence."""

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.history import history_projection
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.schedule import ObservatoryConfig, cadence_window, window_id
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.monitor import MonitorJournal


def window(root, *, change=None, resumed=False):
    from datetime import UTC, datetime

    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    config = ObservatoryConfig.model_validate(
        {
            "version": "observatory-v1",
            "id": "local-one",
            "enabled": True,
            "mode": "mock",
            "models": [
                {
                    "id": "one",
                    "profile": "gpt-oss-120b",
                    "endpoints": [endpoint.model_dump(mode="json")],
                    "reference": {"tier": "MOCK"},
                }
            ],
        }
    )
    cadence = cadence_window("daily", datetime(2026, 1, 1, tzinfo=UTC))
    identifier = window_id(config.id, cadence)
    seed = int(content_hash(["observatory-job-v1", config.seed, identifier, "one"])[:15], 16)
    plan = RunPlan(
        plan_id=identifier,
        seed=seed,
        profile="gpt-oss-120b",
        level="core",
        endpoints=[endpoint],
        suites=["S1"],
        max_usd=0,
        reference={"tier": "MOCK"},
    )
    spec = {
        "version": "observatory-window-v1",
        "window_id": identifier,
        "observatory_id": "local-one",
        "cadence": "daily",
        "start": cadence.start.isoformat(),
        "end": cadence.end.isoformat(),
        "binding": {
            "config_hash": content_hash(config.model_dump(mode="json")),
            "source_hash": implementation_hash(),
            "measurement_hash": "b" * 64,
        },
        "config": config.model_dump(mode="json"),
        "jobs": [
            {
                "model_id": "one",
                "run_id": "monitor-" + content_hash([identifier, "one"]),
                "plan": plan.model_dump(mode="json"),
                "items": [],
                "estimate": {},
            }
        ],
    }
    if change:
        change(spec)
    journal = MonitorJournal(root / "data/raw", identifier)
    journal.create(spec)
    if resumed:
        journal.append({"type": "window_resumed"})
    return journal


def test_absent_history_is_explicit_and_read_only(tmp_path):
    result = history_projection(tmp_path, "local-one")
    assert result["inputs"] == {} and result["steps"] == []
    assert result["qualified"] is False and result["verdict"] == "INCONCLUSIVE"
    assert not (tmp_path / "data").exists()


def test_unfinished_window_and_resume_do_not_recollect_or_fill_zero_bits(tmp_path):
    journal = window(tmp_path, resumed=True)
    before = {str(p): p.read_bytes() for p in (tmp_path / "data/raw").rglob("*") if p.is_file()}
    result = history_projection(tmp_path, "local-one")
    assert len(result["steps"]) == 1 and result["steps"][0]["status"] == "INCONCLUSIVE"
    assert next(iter(result["inputs"].values()))["complete"] is False
    assert result["steps"][0]["event"] is None
    assert before == {
        str(p): p.read_bytes() for p in (tmp_path / "data/raw").rglob("*") if p.is_file()
    }
    assert not (tmp_path / "data/runs").exists() and not (tmp_path / "budget").exists()
    assert len(journal.events()) == 1


@pytest.mark.parametrize(
    "case",
    ["identity", "config_hash", "cadence", "source_shape", "duplicate_jobs", "model", "endpoint"],
)
def test_invalid_sealed_spec_refuses_before_projection(tmp_path, case):
    def alter(spec):
        if case == "identity":
            spec["window_id"] = "different"
        elif case == "config_hash":
            spec["binding"]["config_hash"] = "c" * 64
        elif case == "cadence":
            spec["end"] = spec["start"]
        elif case == "source_shape":
            spec["binding"]["source_hash"] = "unknown"
        elif case == "duplicate_jobs":
            spec["jobs"].append(spec["jobs"][0])
        elif case == "model":
            spec["jobs"][0]["model_id"] = "absent"
        else:
            spec["jobs"][0]["plan"]["endpoints"][0]["id"] = "foreign"

    window(tmp_path, change=alter)
    with pytest.raises(ValueError):
        history_projection(tmp_path, "local-one")


def test_other_namespace_is_not_pooled_and_raw_symlink_refuses(tmp_path):
    window(tmp_path)
    assert history_projection(tmp_path, "other")["steps"] == []
    location = tmp_path / "linked"
    location.mkdir()
    (location / "data").mkdir()
    (location / "data/raw").symlink_to(tmp_path / "data/raw", target_is_directory=True)
    with pytest.raises(ValueError):
        history_projection(location, "local-one")


def test_complete_marker_without_captured_jobs_is_not_a_complete_window(tmp_path):
    journal = window(tmp_path)
    journal.append(
        {
            "type": "window_completed",
            "result": {"window_id": journal.spec()["window_id"], "jobs": [], "resumed": False},
        }
    )
    with pytest.raises(ValueError):
        history_projection(tmp_path, "local-one")
