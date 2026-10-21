"""Hosted triggers remain disabled unless every owner and persistence gate holds."""

from pathlib import Path

import pytest
import yaml

from servegauge.runner.observatory_workflow import prepare_workflow, route


@pytest.mark.parametrize(
    "schedule,cadence",
    [
        ("23 */6 * * *", "probe"),
        ("41 2 * * *", "daily"),
        ("11 3 * * 1", "weekly"),
        ("53 4 1 * *", "monthly"),
    ],
)
def test_exact_schedule_routes(schedule, cadence):
    assert route("schedule", {"schedule": schedule}) == cadence


@pytest.mark.parametrize(
    "event,payload",
    [
        ("push", {}),
        ("schedule", {"schedule": "* * * * *"}),
        ("workflow_dispatch", {"inputs": {"cadence": "$(untrusted)"}}),
    ],
)
def test_unknown_event_or_cadence_refuses(event, payload):
    with pytest.raises(ValueError):
        route(event, payload)


def owner_fixture(tmp_path):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    state = tmp_path / "persistent"
    (state / "configs").mkdir(parents=True)
    (state / "configs/observatory.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "owner-workflow-fixture",
                "enabled": True,
                "mode": "owner",
                "models": [
                    {
                        "id": "fixture",
                        "profile": "gpt-oss-120b",
                        "reference": {"tier": "R3"},
                        "endpoints": [
                            {
                                "id": "fixture",
                                "kind": "openai_compat",
                                "base_url": "https://fixture.invalid/v1",
                                "model": "fixture",
                                "claimed_profile": "gpt-oss-120b",
                            }
                        ],
                    }
                ],
            }
        )
    )
    (state / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    event = {"repository": {"default_branch": "main"}, "inputs": {"cadence": "daily"}}
    environment = {
        "OBSERVATORY_ENABLED": "true",
        "SERVEGAUGE_OWNER_RUN": "1",
        "GITHUB_REF": "refs/heads/main",
        "SERVEGAUGE_STATE_ROOT": str(state),
    }
    return workspace, state, event, environment


def test_owner_plan_uses_persistent_state_and_cached_exact_configuration(tmp_path):
    workspace, state, event, environment = owner_fixture(tmp_path)
    plan = prepare_workflow("workflow_dispatch", event, environment, workspace)
    assert plan.root == state and plan.cadence == "daily" and plan.config.mode == "owner"
    assert plan.config.enabled and plan.config.publishing.enabled is False


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "no_owner",
        "branch",
        "ephemeral_state",
        "missing_budget",
        "parent_segment",
        "case_alias",
        "missing_state",
    ],
)
def test_owner_gate_refuses_before_any_runner_call(tmp_path, change):
    workspace, state, event, environment = owner_fixture(tmp_path)
    if change == "disabled":
        environment["OBSERVATORY_ENABLED"] = "false"
    elif change == "no_owner":
        environment.pop("SERVEGAUGE_OWNER_RUN")
    elif change == "branch":
        environment["GITHUB_REF"] = "refs/heads/unreviewed"
    elif change == "ephemeral_state":
        environment["SERVEGAUGE_STATE_ROOT"] = str(workspace / "state")
    elif change in {"parent_segment", "case_alias"}:
        import shutil

        shutil.copytree(state, workspace / "state")
        if change == "parent_segment":
            outside = tmp_path / "outside"
            outside.mkdir()
            environment["SERVEGAUGE_STATE_ROOT"] = str(outside / ".." / "checkout" / "state")
        else:
            alias = Path(str(workspace).upper()) / "STATE"
            if not alias.exists():
                pytest.skip("Case-sensitive filesystem has no case alias")
            environment["SERVEGAUGE_STATE_ROOT"] = str(alias)
    elif change == "missing_state":
        environment["SERVEGAUGE_STATE_ROOT"] = str(tmp_path / "absent")
    else:
        (state / "configs/budgets.yaml").unlink()
    with pytest.raises(ValueError):
        prepare_workflow("workflow_dispatch", event, environment, workspace)


def test_workflow_yaml_has_no_publication_or_raw_upload_and_has_owner_guards():
    assert not Path(".github/workflows/observatory.yml").exists()
    workflow = yaml.safe_load(Path("configs/workflows/observatory.yml.example").read_text())
    assert set(workflow["on"]) == {"workflow_dispatch", "schedule"}
    job = workflow["jobs"]["observatory"]
    assert "OBSERVATORY_ENABLED" in job["if"] and "default_branch" in job["if"]
    assert job["runs-on"] == ["self-hosted", "servegauge-owner"]
    assert job["environment"] == "observatory-owner"
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    text = Path("configs/workflows/observatory.yml.example").read_text()
    assert "upload-artifact" not in text and "--confirm" not in text


async def test_execution_uses_cached_config_and_never_sends_or_uploads(tmp_path, monkeypatch):
    from servegauge.runner.observatory_workflow import execute

    workspace, state, event, environment = owner_fixture(tmp_path)
    plan = prepare_workflow("workflow_dispatch", event, environment, workspace)
    (state / "configs/observatory.yaml").write_text("changed after planning")
    seen = []

    async def fake_tick(config, cadence, *, root, approved):
        assert config is plan.config and config.mode == "owner"
        assert root == state and cadence == "daily" and approved
        seen.append("tick")
        return {
            "status": "completed",
            "window_id": "fixture-window",
            "jobs": [{"run_id": "fixture-run"}],
            "measurement_exit_code": 0,
        }

    def fake_export(root, out, *, runs):
        assert root == state and runs == ["fixture-run"]
        assert out.parent == state / "exports"
        seen.append("local_export")
        return {"files": ["README.md"], "publication_blockers": []}

    monkeypatch.setattr("servegauge.runner.observatory_workflow.tick", fake_tick)
    monkeypatch.setattr("servegauge.runner.observatory_workflow.stage_export", fake_export)
    result = await execute(plan)
    assert seen == ["tick", "local_export"] and not result["published"]
    assert not result["provider_contact_sent"]


def test_main_refuses_missing_context_without_printing_environment(monkeypatch, capsys):
    from servegauge.runner.observatory_workflow import main

    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setenv("FIXTURE_SECRET_TOKEN", "fixture-private-context")
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 3
    text = capsys.readouterr().out
    assert "fixture-private-context" not in text and "Traceback" not in text
