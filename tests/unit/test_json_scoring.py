import pytest
from tests_support import exchange

from servegauge.core.models import EndpointSpec
from servegauge.items.schemas import LADDER
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.json_scoring import json_samples
from servegauge.suites.registry import prepare_item
from servegauge.suites.s3_structured import structured_items


def endpoint():
    return EndpointSpec(
        id="e",
        kind="mock",
        model="mock",
        base_url="http://127.0.0.1/v1",
        claimed_profile="gpt-oss-120b",
        metadata={
            "declared_json_schema": {"value": True, "verified": True, "source": "mock:/v1/models"},
            "strict_schema_guarantee": {
                "verified": True,
                "source": "mock:/v1/models",
                "features": list(LADDER),
            },
        },
    )


def captured(items, faults=None):
    engine = ScriptedBackend(8, faults or FaultConfig())
    output = []
    for item in items:
        body = prepare_item(item, endpoint()).body
        response = engine.respond(body)
        output.append(
            exchange(
                endpoint_id="e",
                item_id=item.item_id,
                request=body,
                response=response,
                usage=response["usage"],
            )
        )
    return output


def test_clean_ladder_modes_and_deliberate_truncation():
    items = structured_items("smoke", 8, {"items": 33})
    samples = json_samples(items, captured(items), endpoint())
    assert set(samples) == {"J01", "J02.validity", "J02.semantic", "J03", "J04", "J05"}
    assert all(sample.status is None for sample in samples.values())
    for key in ("J01", "J02.semantic", "J03"):
        assert len(samples[key].values) == 33 and all(samples[key].values.values())
    for key in ("J02.validity", "J04", "J05"):
        assert not any(samples[key].values.values())
    faulted = json_samples(items, captured(items, FaultConfig(ignore_max_tokens=True)), endpoint())
    assert sum(faulted["J05"].values.values()) == 33
    assert faulted["J05"].replay_candidates


def test_schema_invalidity_cannot_imply_unverified_native_contract_failure():
    items = [
        i for i in structured_items("smoke", 8, {"items": 33}) if i.expected["check_id"] == "J02"
    ]
    records = captured(items)
    for record in records:
        record.response["choices"][0]["message"]["content"] = "{}"
    trusted = json_samples(items, records, endpoint())
    assert sum(trusted["J02.validity"].values.values()) == 33
    assert not any(trusted["J02.semantic"].values.values())
    unverified = endpoint()
    unverified.metadata.pop("strict_schema_guarantee")
    uncertain = json_samples(items, records, unverified)
    assert uncertain["J02.validity"].status == "INCONCLUSIVE"
    assert uncertain["J02.semantic"].status is None


@pytest.mark.parametrize(
    "declaration,code,status",
    [
        (True, "unsupported_response_format", None),
        (True, "invalid_schema", "INCONCLUSIVE"),
        (False, "unsupported_response_format", "NOT_APPLICABLE"),
        (None, "unsupported_response_format", "INCONCLUSIVE"),
    ],
)
def test_declared_support_is_sourced_and_rejections_must_isolate_feature(declaration, code, status):
    items = [
        i
        for i in structured_items("smoke", 8, {"items": 33})
        if i.expected["check_id"] == "J02" and i.expected["category"] == "flat"
    ]
    records = captured(items)
    for record in records:
        record.status, record.response = 400, {"error": {"code": code}}
    source = endpoint()
    if declaration is None:
        source.metadata.pop("declared_json_schema")
    else:
        source.metadata["declared_json_schema"]["value"] = declaration
    samples = json_samples(items, records, source)
    assert samples["J04"].status == status
    assert samples["J02.semantic"].status == "NOT_SUPPORTED"
    if status is None:
        assert sum(samples["J04"].values.values()) == 3
        assert len(samples["J04"].replay_candidates) == 3
    source.metadata["observed_capabilities"] = {"json_schema": False}
    decision = prepare_item(items[0], source)
    assert (decision.body is not None) == (declaration is True)


@pytest.mark.parametrize("failure", ["missing", "transport", "unsupported", "censored", "envelope"])
def test_JSON_capture_and_output_caps_do_not_become_model_failures(failure):
    item = structured_items("smoke", 8, {"items": 1})[0]
    records = captured([item])
    if failure == "missing":
        records = []
    elif failure == "transport":
        records[0].error = "transport_timeout"
    elif failure == "unsupported":
        records[0].metadata = {"request_not_sent": True, "verdict": "NOT_SUPPORTED"}
    elif failure == "censored":
        records[0].response["choices"][0]["finish_reason"] = "length"
    else:
        records[0].response = {"invalid": True}
    sample = json_samples([item], records, endpoint())["J01"]
    assert sample.status in {"ERROR", "NOT_SUPPORTED", "INCONCLUSIVE"}
