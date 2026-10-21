import pytest
from tests_support import exchange

from servegauge.diagnose.rules import diagnoses
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.scoring.tokenizer_hints import tokenizer_samples
from servegauge.suites.registry import prepare_item
from servegauge.suites.s7_accounting import accounting_items


def test_derived_prompt_counter_also_refuses_a_different_asset_revision():
    from servegauge.scoring.accounting_scoring import accounting_samples
    from servegauge.suites.config import suite_config

    item, endpoint, record = hint_capture()
    profile = load_profile("gpt-oss-120b")
    profile.tokenizer["hf_revision"] = "unqualified"
    groups = accounting_samples(
        [item], [record], endpoint.id, profile, suite_config()["accounting"]
    )
    sample = groups["B01.B05"]
    assert sample.status == "INCONCLUSIVE" and not sample.values


def hint_capture():
    item = next(
        i for i in accounting_items("smoke", 17, {"items": 1}) if i.expected["check_id"] == "B05"
    )
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    body = prepare_item(item, endpoint).body
    response = ScriptedBackend(71, FaultConfig()).respond(body)
    record = exchange(
        endpoint_id=endpoint.id,
        item_id=item.item_id,
        request=body,
        response=response,
        usage=response["usage"],
    )
    return item, endpoint, record


def test_claimed_tokenizer_count_is_a_comparative_hint_with_captured_layout():
    item, endpoint, record = hint_capture()
    sample = tokenizer_samples([item], [record], endpoint.id, load_profile("gpt-oss-120b"))["B05"]
    assert sample.status is None and sample.check_class == "comparative" and sample.two_sided
    assert sample.values and sample.evidence == [record.exchange_id]
    assert sample.details["input_hashes"] and sample.details["tokenizer_contract"]
    assert sample.details["model_identity_proof"] is False
    assert "fingerprint" in sample.details["diagnosis_requires"]
    assert (
        tokenizer_samples(
            [item.model_copy(update={"version": "prompt-count-v1"})],
            [record],
            endpoint.id,
            load_profile("gpt-oss-120b"),
        )
        == {}
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "rejected",
        "unsent",
        "bad-count",
        "bad-revision",
        "no-local",
        "redacted",
    ],
)
def test_unqualified_tokenizer_hints_abstain_without_dropping_items(problem):
    item, endpoint, record = hint_capture()
    profile = load_profile("gpt-oss-120b")
    if problem == "rejected":
        record.status = 400
    elif problem == "unsent":
        record.metadata["request_not_sent"] = True
    elif problem == "bad-count":
        record.usage["prompt_tokens"] = True
    elif problem == "bad-revision":
        profile.tokenizer["hf_revision"] = "unqualified"
    elif problem == "no-local":
        profile.family = "unknown"
    elif problem == "redacted":
        from servegauge.core.redaction import REDACTED

        record.request["messages"][0]["content"] = REDACTED
    records = (
        [] if problem == "missing" else [record, record] if problem == "duplicate" else [record]
    )
    sample = tokenizer_samples([item], records, endpoint.id, profile)["B05"]
    assert sample.status in {"ERROR", "INCONCLUSIVE"} and not sample.values


def observation(check, *, run="same", position="different-model-band-or-beyond", replay=False):
    from servegauge.core.models import CheckResult

    return CheckResult(
        check_id=check,
        suite_id="S6" if check.startswith("FP") else "S7",
        check_class="comparative",
        endpoint_id="target",
        run_id=run,
        n=200,
        estimate=0.4,
        ci_low=0.35,
        ci_high=0.45,
        reference_estimate=0,
        reference_tier="MOCK",
        delta=0.4,
        delta_ci_low=0.35,
        delta_ci_high=0.45,
        margin=0.03,
        p_value=0.001,
        adjusted_p=0.003,
        verdict="FAIL",
        reason="observed difference",
        evidence=[run + "-" + check],
        details={"fingerprint_position": position, "replay_confirmed": replay},
    )


def test_tokenizer_hint_needs_two_distinct_same_run_fingerprint_methods():
    hint, first, second = [observation(name) for name in ("B05", "FP1", "FP2")]
    assert not diagnoses([hint])
    assert not diagnoses([hint, first])
    assert not diagnoses([hint, first, first.model_copy(deep=True)])
    assert not diagnoses([hint, first, second.model_copy(update={"run_id": "different"})])
    found = diagnoses([hint, first, second])
    assert len(found) == 1 and found[0].code == "DIFFERENT_MODEL_LIKE"
    assert set(found[0].check_ids) == {"B05", "FP1", "FP2"}
    assert set(found[0].evidence) == {"same-B05", "same-FP1", "same-FP2"}
    assert found[0].confidence == "low"


@pytest.mark.parametrize(
    "position", ["same-model-band", "between-bands", "calibration-overlap", "below-same-model-band"]
)
def test_other_band_positions_do_not_establish_different_model_like(position):
    assert not diagnoses(
        [observation("B05"), observation("FP1"), observation("FP2", position=position)]
    )


def test_info_fingerprint_or_unreproduced_prerequisite_cannot_inflate_support():
    hint, first, second = [observation(name, replay=True) for name in ("B05", "FP1", "FP2")]
    second.verdict = "INFO"
    second.check_class = "report_only"
    second.details["signal_detected"] = True
    assert not diagnoses([hint, first, second])
    second.verdict = "FAIL"
    second.check_class = "comparative"
    first.details["replay_confirmed"] = False
    assert diagnoses([hint, first, second])[0].confidence == "low"
    first.details["replay_confirmed"] = True
    assert diagnoses([hint, first, second])[0].confidence == "high"


@pytest.mark.parametrize("check", ["B02", "B03.prompt", "B04"])
def test_full_accounting_failures_have_accounting_hypotheses(check):
    found = diagnoses([observation(check)])
    assert found[0].code == "ACCOUNTING_MISMATCH" and found[0].failure_class == "ACCOUNTING"


def test_cross_run_success_cannot_supply_a_missing_structural_prerequisite():
    schema = observation("T02")
    trigger = observation("T01", run="other")
    trigger.verdict = "PASS"
    assert not diagnoses([schema, trigger])
