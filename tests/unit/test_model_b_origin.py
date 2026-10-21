import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec
from servegauge.profiles import load_profile
from servegauge.refs.owner import Qualification, bind_endpoint, qualification_template

SOURCE = "https://docs.z.ai/guides/llm/glm-5.3"


def endpoint(variant="coding"):
    return EndpointSpec.model_validate_json(
        Path(f"configs/references/glm-5.3-{variant}-endpoint.json").read_bytes()
    )


def reviewed(variant="coding"):
    """Synthetic mapping assertion only; no maker capture or reference tier granted."""
    profile = load_profile("glm-5.3")
    declaration = endpoint(variant)
    form = qualification_template("R1", profile, declaration, maker="zai")
    assert form["reviewed_at"] is None and form["model_mapping_reviewed"] is False
    form.update(
        reviewed_at="2026-10-21T00:00:00+00:00",
        source_sha256="d" * 64,
        model_mapping_reviewed=True,
    )
    return Qualification.model_validate(form), profile, declaration


@pytest.mark.parametrize("variant", ["standard", "coding"])
def test_dated_researched_endpoint_forms_stay_incomplete_by_default(variant):
    profile = load_profile("glm-5.3")
    declaration = endpoint(variant)
    form = qualification_template("R1", profile, declaration, maker="zai")
    expected = SOURCE if variant == "coding" else "https://docs.z.ai/guides/overview/quick-start"
    assert form["source_url"] == expected
    assert form["source_sha256"] is None
    assert form["model_mapping_reviewed"] is False
    with pytest.raises(ValueError):
        Qualification.model_validate(form)
    request, profile, declaration = reviewed(variant)
    bound = bind_endpoint(request, profile, declaration)
    assert bound.metadata["qualified_reference"] == {
        "version": "qualified-reference-binding-v1",
        "qualification_hash": content_hash(request.model_dump(mode="json")),
    }
    assert "pricing" not in bound.metadata


@pytest.mark.parametrize("change", ["host", "base", "source", "profile", "maker", "mapping"])
def test_coding_source_does_not_authorize_other_routes_profiles_or_unreviewed_mapping(change):
    request, profile, declaration = reviewed()
    value = request.model_dump(mode="json")
    if change == "host":
        declaration.base_url = "https://unrelated.example/v1"
    elif change == "base":
        declaration.base_url = "https://api.z.ai/api/paas/v4"
    elif change == "source":
        value["source_url"] = "https://docs.z.ai/unreviewed"
    elif change == "profile":
        profile.id = "other-glm"
        declaration.claimed_profile = profile.id
        value.update(profile=profile.id, profile_hash=content_hash(profile.model_dump(mode="json")))
    elif change == "maker":
        value["origin"]["maker"] = "deepseek"
    else:
        value["model_mapping_reviewed"] = False
    value["endpoint_hash"] = content_hash(declaration.model_dump(mode="json"))
    with pytest.raises(ValueError):
        bind_endpoint(Qualification.model_validate(value), profile, declaration)


def test_shipped_plan_has_no_paid_prerequisites_and_cli_never_grants_a_tier():
    plan = json.loads(Path("configs/references/glm-5.3-owner-plan.json").read_bytes())
    assert plan["max_usd"] == 0 and "pricing" not in plan["endpoints"][0]["metadata"]
    for variant in ("standard", "coding"):
        result = CliRunner().invoke(
            app,
            [
                "refs",
                "template",
                "--profile",
                "glm-5.3",
                "--tier",
                "R1",
                "--maker",
                "zai",
                "--endpoint",
                f"configs/references/glm-5.3-{variant}-endpoint.json",
            ],
        )
        assert result.exit_code == 0
        form = json.loads(result.stdout)
        assert form["reviewed_at"] is None and form["model_mapping_reviewed"] is False


@pytest.mark.parametrize("variant", ["standard", "coding"])
def test_actual_prepare_cli_retains_synthetic_owner_assertion_and_grants_no_tier(
    tmp_path, monkeypatch, variant
):
    import httpx

    from servegauge.storage.workspace import initialize

    request, _, declaration = reviewed(variant)
    initialize(tmp_path)
    folder = tmp_path / "configs/references"
    folder.mkdir()
    (folder / "endpoint.json").write_text(declaration.model_dump_json())
    (folder / "review.json").write_text(request.model_dump_json())
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("reference preparation must not construct a network client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    command = [
        "refs",
        "prepare",
        "--profile",
        "glm-5.3",
        "--endpoint",
        "configs/references/endpoint.json",
        "--qualification",
        "configs/references/review.json",
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.stdout
    prepared = json.loads(result.stdout)
    assert prepared["status"] == "prepared_only"
    assert prepared["model_api_calls"] == 0 and prepared["reference_tier_granted"] is False
    assert prepared["qualification_hash"] == content_hash(request.model_dump(mode="json"))
    request.model_mapping_reviewed = False
    (folder / "review.json").write_text(request.model_dump_json())
    assert CliRunner().invoke(app, command).exit_code == 3
