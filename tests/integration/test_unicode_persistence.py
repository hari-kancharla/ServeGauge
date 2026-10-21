from pathlib import Path

import yaml

from servegauge.core.models import CheckResult, EndpointSpec
from servegauge.profiles import load_profile
from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun, build_pack, load_pack, save_pack
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import load_plan, load_requirements
from servegauge.storage.check_report import write_check_report


def test_unicode_round_trips_with_simulated_CP1252_defaults(tmp_path, monkeypatch):
    original_read, original_write = Path.read_text, Path.write_text

    def read(path, encoding=None, errors=None):
        return original_read(path, encoding=encoding or "cp1252", errors=errors)

    def write(path, data, encoding=None, errors=None, newline=None):
        return original_write(
            path, data, encoding=encoding or "cp1252", errors=errors, newline=newline
        )

    profile = load_profile("gpt-oss-120b")
    profile.format_markers.append("東京")
    profile.unverified.append("format_markers")
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "gpt-oss-120b.yaml").write_bytes(
        yaml.safe_dump(profile.model_dump(), allow_unicode=True).encode("utf-8")
    )
    endpoint = EndpointSpec(
        id="direct:東京",
        kind="openai_compat",
        model="model",
        base_url="https://example.com/v1",
        claimed_profile="gpt-oss-120b",
        labels={"region": "東京"},
    )
    config = {
        "plan_id": "unicode",
        "profile": "gpt-oss-120b",
        "level": "smoke",
        "seed": 1,
        "reference": {"tier": "BASELINE"},
        "max_usd": 0,
        "endpoints": [endpoint.model_dump()],
        "suites": ["S3"],
    }
    path = tmp_path / "plan.yaml"
    path.write_bytes(yaml.safe_dump(config, allow_unicode=True).encode("utf-8"))
    requirements = tmp_path / "requires.yaml"
    requirements.write_bytes(
        yaml.safe_dump(
            {"app": "東京", "requires": [{"id": "json", "tests": ["J01"]}]}, allow_unicode=True
        ).encode("utf-8")
    )
    snapshots = tmp_path / "configs/endpoints"
    snapshots.mkdir(parents=True)
    (snapshots / "sample.yaml").write_bytes(
        yaml.safe_dump({"endpoints": [endpoint.model_dump()]}, allow_unicode=True).encode("utf-8")
    )
    source = ReferenceRun(
        run_id="unicode",
        profile="gpt-oss-120b",
        configuration={"label": "東京"},
        item_set_hashes={"S2": "items"},
        baseline_endpoint_id=endpoint.id,
        endpoints=[ReferenceEndpoint(id=endpoint.id, profile="gpt-oss-120b", kind="openai_compat")],
        checks={"T02": {endpoint.id: {"東京": [1.0]}}},
    )
    pack = build_pack(source, "BASELINE", seed=1)
    pack_path = save_pack(pack, tmp_path / "refs")
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)
    assert load_profile("gpt-oss-120b", directory).format_markers[-1] == "東京"
    assert load_plan(path).endpoints[0].labels["region"] == "東京"
    assert load_requirements(requirements).app == "東京"
    assert (
        resolve_endpoint(
            endpoint.base_url, endpoint.model, endpoint.claimed_profile, root=tmp_path
        ).labels["region"]
        == "東京"
    )
    assert load_pack(str(pack_path)) == pack
    report = tmp_path / "report"
    result = CheckResult(
        check_id="J01",
        check_class="comparative",
        suite_id="S3",
        endpoint_id=endpoint.id,
        run_id="unicode",
        n=1,
        verdict="INCONCLUSIVE",
        reason="東京",
    )
    write_check_report(report, [result], metadata={"run_id": "unicode"})
    assert "東京" in (report / "checks.md").read_bytes().decode("utf-8")
