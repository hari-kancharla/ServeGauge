"""A verified binding extends only source compatibility, never process/request scope."""

import copy

import pytest
from test_monitor_admission_loader import qualification
from test_monitor_transport_batch import captured_batch as captured_batch_fixture

from servegauge.core.hashing import content_hash
from servegauge.observatory.admission import load_admission
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.transport import transport_batch

captured_batch = captured_batch_fixture


@pytest.mark.parametrize("context", [512, 131072])
def test_reviewed_source_retains_actual_units_and_current_only_default(
    tmp_path, captured_batch, context
):
    producer, _ = qualification(tmp_path)
    admission = load_admission(tmp_path).admission
    analysis, endpoint, arguments = captured_batch
    analysis = copy.deepcopy(analysis)
    analysis.plan.endpoints[0].metadata["context_length"] = context
    arguments.update(policy=load_rolling_policy(), producer_source_hash="b" * 64)
    binding = {**producer, "source_hash": "b" * 64, "loaded_source_hash": "b" * 64}
    assert not transport_batch(analysis, endpoint, **arguments).ready
    result = transport_batch(
        analysis, endpoint, **arguments, admission=admission, producer_binding=binding
    )
    assert result.ready and result.n == 400 and sum(result.values) == 390 and not result.qualified


@pytest.mark.parametrize(
    "case",
    [
        "context_low",
        "context_high",
        "context_missing",
        "context_bool",
        "binding",
        "runtime",
        "policy",
        "source",
        "unknown",
        "defaults",
    ],
)
def test_admission_does_not_expand_other_guards(tmp_path, captured_batch, case):
    producer, _ = qualification(tmp_path)
    admission = load_admission(tmp_path).admission
    analysis, endpoint, arguments = captured_batch
    analysis = copy.deepcopy(analysis)
    metadata = analysis.plan.endpoints[0].metadata
    metadata["context_length"] = 512
    arguments["policy"] = load_rolling_policy()
    binding = copy.deepcopy(producer)
    if case == "context_low":
        metadata["context_length"] = 511
    elif case == "context_high":
        metadata["context_length"] = 131073
    elif case == "context_missing":
        metadata.pop("context_length")
    elif case == "context_bool":
        metadata["context_length"] = True
    elif case == "binding":
        binding = None
    elif case == "runtime":
        binding["runtime"]["complete"] = False
    elif case == "policy":
        arguments["policy"] = arguments["policy"].model_copy(update={"max_streams": 99})
    elif case == "source":
        arguments["producer_source_hash"] = "b" * 64
    elif case == "unknown":
        arguments["producer_source_hash"] = binding["source_hash"] = binding[
            "loaded_source_hash"
        ] = content_hash("unknown")
    else:
        analysis.plan.endpoints[0].defaults["temperature"] = 0
    assert not transport_batch(
        analysis, endpoint, **arguments, admission=admission, producer_binding=binding
    ).ready
