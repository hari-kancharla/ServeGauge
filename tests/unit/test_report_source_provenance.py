"""Reports preserve recorded source facts and do not assign provenance to missing attempts."""

import copy
import json

from unit.test_endpoint_report import fixture

from servegauge.report.endpoint_report import _document as snapshot
from servegauge.report.endpoint_report import endpoint_context
from servegauge.report.source_provenance import source_provenance


def source(digest="a" * 64):
    return {
        "source_tree_hash": digest,
        "git_sha": "b" * 40,
        "git_dirty": False,
        "git_state": "bound_checkout",
        "source_identity": {
            "version": "capture-source-v2",
            "scope": "loaded-package-python-v1",
            "package": "servegauge",
            "source_hash": digest,
            "import_source_hash": digest,
            "python_files": 100,
            "status": "consistent_with_import_snapshot",
        },
    }


def test_report_distinguishes_original_context_resumed_sources_and_unknown_attempts():
    analysis = fixture()
    analysis.manifest.update(source())
    analysis.records[0].attempts[0].metadata["execution_source"] = source()
    analysis.records[1].attempts[0].metadata["execution_source"] = source("c" * 64)
    analysis.records[2].attempts[0].metadata["execution_source"] = source("c" * 64)
    before = copy.deepcopy(analysis.records)
    result = source_provenance(analysis.manifest, analysis.records)
    assert result["context"]["scope"] == "loaded-package-python-v1"
    assert result["recorded_attempts"] == 4 and result["unknown_attempts"] == 1
    assert result["loaded_package_attempts"] == 3 and result["legacy_attempts"] == 0
    assert sorted(row["attempts"] for row in result["sources"]) == [1, 2]
    assert {row["source_tree_hash"] for row in result["sources"]} == {"a" * 64, "c" * 64}
    assert analysis.records == before
    assert snapshot(analysis, redact_content=True)["metadata"]["source_provenance"] == result
    context = endpoint_context(analysis, redact_content=True)
    assert any(row["value"] == "Loaded package Python source" for row in context["provenance"])
    assert any("3 of 4 attempts" in row["value"] for row in context["provenance"])
    assert not any(row["label"] == "Source provenance" for row in context["provenance"])


def test_legacy_context_stays_legacy_and_cannot_fill_unknown_attempt_sources():
    analysis = fixture()
    result = source_provenance(analysis.manifest, analysis.records)
    assert result["context"]["scope"] == "legacy-cwd-git-inventory"
    assert result["sources"] == []
    assert result["unknown_attempts"] == result["recorded_attempts"] == 4
    assert result["loaded_package_attempts"] == 0


def test_mismatched_or_opaque_identity_is_unknown_and_private_text_is_never_projected():
    analysis = fixture()
    value = source()
    value["source_identity"]["source_hash"] = "d" * 64
    value["git_sha"] = "PRIVATE_SOURCE_CANARY"
    value["absolute_path"] = "/private/source/canary"
    analysis.records[0].attempts[0].metadata["execution_source"] = value
    analysis.records[1].attempts[0].metadata["execution_source"] = {
        "source_tree_hash": "PRIVATE_SOURCE_CANARY"
    }
    result = source_provenance(analysis.manifest, analysis.records)
    assert result["unknown_attempts"] == 4
    assert "PRIVATE_SOURCE_CANARY" not in json.dumps(result)
    assert "/private/source/canary" not in json.dumps(result)


def test_public_reference_copy_retains_versioned_source_scope_without_new_numeric_claims():
    from unit.test_refs import reference_run

    from servegauge.refs.pack import build_pack
    from servegauge.refs.publication import public_pack

    run = reference_run()
    run.manifest.update(source())
    pack = build_pack(run, "BASELINE", seed=8)
    before = pack.model_dump(mode="json")
    public = public_pack(pack)
    assert public.manifest["source_manifest"]["source_identity"] == source()["source_identity"]
    assert public.manifest["source_manifest"]["git_state"] == "bound_checkout"
    assert public.outcomes == pack.outcomes and public.calibration == pack.calibration
    assert pack.model_dump(mode="json") == before


def test_legacy_attempt_inventory_is_retained_without_becoming_a_loaded_code_identity():
    analysis = fixture()
    analysis.records[0].attempts[0].metadata["execution_source"] = {"source_tree_hash": "c" * 64}
    result = source_provenance(analysis.manifest, analysis.records)
    assert result["loaded_package_attempts"] == 0
    assert result["legacy_attempts"] == 1 and result["unknown_attempts"] == 3
    assert result["sources"][0]["scope"] == "legacy-cwd-git-inventory"
