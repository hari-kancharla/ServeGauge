"""Raw-only quartet analysis, missingness and label-blind scoring contracts."""

import copy
import json

import numpy as np
import pytest
from integration.test_gpu_study_collection_workflow import setup_kit
from unit.test_gpu_study_protocol import specification
from unit.test_local_embeddings import fixture_model

from servegauge.scoring.gpu_study import score_quartet
from servegauge.storage.gpu_capture import collect_slot, load_slot
from servegauge.storage.gpu_results import analyze_study, write_study_report
from servegauge.verdicts.config import DecisionPolicy


class FixtureEncoder:
    def __init__(self, spec):
        self.spec = spec
        self.identity = {
            "artifact_hash": spec.artifact_hash,
            "runtime": {"fixture": "original-software-QA-v1"},
        }

    def encode(self, texts):
        values = np.asarray(
            [[len(text) + 1, sum(map(ord, text)) % 11 + 1] for text in texts], dtype=float
        )
        return values / np.linalg.norm(values, axis=1)[:, None]


def reviewed_setup(tmp_path, monkeypatch):
    folder = tmp_path / "embedding-fixture"
    folder.mkdir()
    source, embedding = fixture_model(folder)
    fingerprints = specification().fingerprints | {
        "embedding_model": embedding.model_dump(mode="json")
    }
    return (
        *setup_kit(
            tmp_path,
            monkeypatch,
            design={
                "version": "gpu-study-v1",
                "study_id": "qa",
                "seed": 17,
                "replicates": 2,
                "independence_reviewed": True,
            },
            fingerprints=fingerprints,
            within_capture_components_reviewed=True,
            iid_condition_quartets_reviewed=True,
        ),
        FixtureEncoder(embedding),
    )


async def test_selected_raw_quartet_scores_match_pure_measurement_before_labels(
    tmp_path, monkeypatch
):
    spec, kit, digest, matrix, encoder = reviewed_setup(tmp_path, monkeypatch)
    quartet = matrix[:4]
    for slot in quartet:
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    result = analyze_study(tmp_path, kit, digest, encoder=encoder)
    assert result["scope"] == "mock_software_qa" and result["reference_tier"] == "MOCK"
    assert result["physical_precision_verified"] is False
    assert result["status"] == "INCONCLUSIVE"
    assert result["capture_summary"]["planned_slots"] == 80
    assert result["capture_summary"]["bound_slots"] == 4
    assert result["capture_summary"]["complete_quartets"] == 1
    assert result["capture_summary"]["missing_slots"] == 76
    unit = result["units"][0]
    assert unit["condition"] == "bf16" and unit["partition"] == "calibration"
    assert all(
        unit["metrics"][name]["estimate"] is not None
        for name in ("FP1", "FP2", "FP3-mean", "FP3-dispersion", "FP4")
    )
    loaded = [load_slot(tmp_path, kit, digest, slot["slot_id"]) for slot in quartet]
    measurement = score_quartet(
        [value["samples"] for value in loaded],
        policy=DecisionPolicy.model_validate(loaded[0]["analysis"].metadata["policy"]),
        seed=unit["measurement_seed"],
        encoder=encoder,
    )
    for method, row in measurement.items():
        assert unit["metrics"][method]["estimate"] == row["estimate"]
        assert unit["metrics"][method]["ci_low"] == row["ci_low"]
        assert unit["metrics"][method]["ci_high"] == row["ci_high"]
    for methods in result["comparisons"].values():
        for curve in methods.values():
            assert curve["auc"]["estimate"] is None
            assert curve["missing"]["evaluation"] == {"negative": 2, "positive": 2}
    # Derived cached checks never enter the study computation.
    before = copy.deepcopy(result)
    again = analyze_study(tmp_path, kit, digest, encoder=encoder)
    assert again == before
    out = tmp_path / "reports/gpu-studies/partial"
    written = write_study_report(tmp_path, result, out=out)
    assert written["physical_precision_verified"] is False
    assert (out / "README.md").is_file() and (out / "curves.csv").is_file()
    assert (out / "roc.svg").is_file() and (out / "roc.png").is_file()
    assert "SOFTWARE QA" in (out / "README.md").read_text()
    assert "No physical GPU" in (out / "README.md").read_text()
    with pytest.raises(ValueError):
        write_study_report(tmp_path, result, out=out)


def test_empty_study_has_all_planned_missing_units_and_no_fabricated_zero(tmp_path, monkeypatch):
    spec, kit, digest, matrix = setup_kit(tmp_path, monkeypatch)
    result = analyze_study(tmp_path, kit, digest)
    assert result["capture_summary"]["bound_slots"] == 0
    assert len(result["units"]) == 20
    for unit in result["units"]:
        for row in unit["metrics"].values():
            assert row["estimate"] is None and row["n_components"] == 0
            assert len(row["capture_counts"]) == 4
    for methods in result["comparisons"].values():
        for curve in methods.values():
            assert curve["auc"]["estimate"] is None
            assert curve["missing"]["calibration"] == {"negative": 2, "positive": 2}
    assert not (tmp_path / "data").exists()


async def test_one_missing_logprob_does_not_delete_usable_text_or_become_zero(
    tmp_path, monkeypatch
):
    import httpx

    from servegauge.mock.backends import task_from_messages
    from servegauge.storage.gpu_capture import _mock_transport

    spec, kit, digest, matrix, encoder = reviewed_setup(tmp_path, monkeypatch)

    class MissingLogprob(httpx.AsyncBaseTransport):
        def __init__(self, transport):
            self.transport, self.changed = transport, False

        async def handle_async_request(self, request):
            body = json.loads(request.content)
            response = await self.transport.handle_async_request(request)
            if (
                not self.changed
                and body.get("logprobs")
                and task_from_messages(body["messages"]).get("type") == "sample"
            ):
                value = json.loads(await response.aread())
                await response.aclose()
                value["choices"][0]["logprobs"] = None
                self.changed = True
                return httpx.Response(response.status_code, json=value, request=request)
            return response

        async def aclose(self):
            await self.transport.aclose()

    for slot in matrix[:4]:
        transport = MissingLogprob(_mock_transport(slot)) if slot == matrix[0] else None
        await collect_slot(
            tmp_path, kit, digest, slot["slot_id"], approved=True, transport=transport
        )
    value = analyze_study(tmp_path, kit, digest, encoder=encoder)
    metrics = value["units"][0]["metrics"]
    assert metrics["FP1"]["estimate"] is not None and metrics["FP4"]["estimate"] is not None
    assert metrics["FP3-mean"]["estimate"] is None
    assert metrics["FP3-dispersion"]["estimate"] is None
    assert metrics["FP3-mean"]["capture_counts"][0]["missing_logprobs"] == 1


async def test_reused_capture_identity_refuses_before_statistics(tmp_path, monkeypatch):
    spec, kit, digest, matrix, encoder = reviewed_setup(tmp_path, monkeypatch)
    import servegauge.storage.gpu_results as results

    for slot in matrix[:4]:
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    original = results.load_selected_slot
    first = original(results.study_context(tmp_path, kit, digest), matrix[0])["capture_ids"][0]

    def reused(context, slot):
        value = original(context, slot)
        if slot == matrix[1] and value is not None:
            value["capture_ids"].append(first)
        return value

    monkeypatch.setattr(results, "load_selected_slot", reused)
    with pytest.raises(ValueError, match="reuse"):
        analyze_study(tmp_path, kit, digest, encoder=encoder)


async def test_observed_mixed_host_runtimes_cannot_be_pooled_as_matched_controls(
    tmp_path, monkeypatch
):
    spec, kit, digest, matrix, encoder = reviewed_setup(tmp_path, monkeypatch)
    import servegauge.storage.gpu_results as results

    for slot in matrix[:4]:
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    original = results.load_selected_slot

    def mixed(context, slot):
        value = original(context, slot)
        if value is not None:
            value["observed_runtime"] = {
                "gpu_observed": "NVIDIA H100",
                "driver_version": "570.26" if slot == matrix[0] else "different-driver",
            }
        return value

    monkeypatch.setattr(results, "load_selected_slot", mixed)
    result = analyze_study(tmp_path, kit, digest, encoder=encoder)
    assert result["runtime_cohort"]["matched"] is False
    assert all(
        row["estimate"] is None and row["reason"] == "unmatched_observed_runtime_cohort"
        for row in result["units"][0]["metrics"].values()
    )


def test_measurement_seed_ignores_execution_provenance_but_preserves_observations():
    from servegauge.scoring.common import Samples
    from servegauge.storage.gpu_results import _measurement_seed

    sample = Samples(check_id="FP1", check_class="comparative", metric="fingerprint")
    sample.details = {
        "design_hash": "original-design",
        "sampling_design": {"session_size": 1},
        "observations": [
            {
                "cell": "cell-1",
                "draw": 0,
                "text": "red",
                "metadata": {"execution_source": {"git_state": "bound_checkout"}},
            }
        ],
    }
    original = copy.deepcopy(sample)
    seed = _measurement_seed([{"FP1": sample}])
    sample.details["observations"][0]["metadata"]["execution_source"] = {
        "git_state": "not_checkout",
        "git_dirty": False,
    }
    assert _measurement_seed([{"FP1": sample}]) == seed
    assert sample.details["observations"][0]["metadata"]["execution_source"] == {
        "git_state": "not_checkout",
        "git_dirty": False,
    }
    sample.details["observations"][0]["text"] = "blue"
    assert _measurement_seed([{"FP1": sample}]) != seed
    assert _measurement_seed([{"FP1": original}]) == seed
