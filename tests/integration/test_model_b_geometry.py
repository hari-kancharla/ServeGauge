"""Actual native GLM corpus geometry, independently counted by the local official backend."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from tests_support import exchange

from servegauge.core.models import EndpointSpec
from servegauge.items.longout import render_output
from servegauge.profiles import load_profile
from servegauge.scoring.common import invariant_violation
from servegauge.storage.source_identity import capture_source
from servegauge.suites.config import suite_config
from servegauge.suites.s1_protocol import probe, protocol_items
from servegauge.suites.s3_structured import structured_items
from servegauge.suites.s4_longctx import context_items
from servegauge.suites.s5_longout import reasoning_items
from servegauge.tokencount.glm_assets import FILES, cache_directory
from servegauge.tokencount.model_text import profile_codec
from servegauge.verdicts.config import load_policy


@pytest.mark.slow
def test_context_positions_prefixes_long_output_and_caps_match_native_oracle(tmp_path):
    executable = os.environ.get("SERVEGAUGE_GLM_REFERENCE_PYTHON")
    if not executable:
        pytest.skip("explicit local Transformers runtime and pinned GLM assets required")
    profile = load_profile("glm-5.3")
    codec = profile_codec(profile)
    assert codec is not None
    cfg = suite_config()
    endpoint = EndpointSpec(
        id="glm-geometry",
        kind="mock",
        base_url="http://127.0.0.1:8011/v1",
        model="mock",
        claimed_profile=profile.id,
        metadata={"context_length": 131072},
    )
    contexts = context_items(
        "smoke",
        991,
        [endpoint],
        {"lengths": [1000, 4000], "depths": [0, 50, 100], "task_types": 2, "repeats": 1},
        profile=profile,
    )
    outputs = reasoning_items(
        "smoke",
        991,
        {"items": 1, "samples": 1, "long_items": 2, "lengths": [4000, 8000]},
        profile=profile,
    )
    endpoint.metadata["context_length"] = 1024
    protocols = protocol_items("smoke", 991, endpoint=endpoint, profile=profile)
    structured = structured_items("smoke", 991, {"items": 1}, profile=profile)
    texts = set()
    for item in contexts:
        for message in item.payload["messages"]:
            texts.add(message["content"])
        if item.expected["check_id"] == "context":
            marker = (
                item.expected["mock_task"]["keys"]["single"]
                + "="
                + item.expected["answers"]["single"]
            )
            text = item.payload["messages"][0]["content"]
            texts.add(text[: text.index(marker)])
    for item in outputs:
        if item.expected["check_id"].startswith("R03"):
            texts.add(item.expected["full_text"])
            texts.add(
                render_output(
                    item.expected["start"], item.expected["count"] - 1, item.expected["kind"]
                )
            )
    for item in protocols:
        if item.expected["check_id"] == "P12":
            texts.add(item.payload["messages"][0]["content"])
    texts.update(["é café 北京", ""])
    corpus = {"version": "original-glm-native-geometry-v1", "texts": sorted(texts), "chats": []}
    corpus_path = tmp_path / "original-corpus.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [
            executable,
            str(Path("tests/model_b_official.py").absolute()),
            str(cache_directory()),
            str(corpus_path),
        ],
        env={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ},
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    oracle = json.loads(result.stdout)
    assert oracle["assets"] == FILES and oracle["model_inference_calls"] == 0
    for text in texts:
        assert codec.encode(text) == oracle["texts"][text]
        assert codec.decode(codec.encode(text)) == oracle["decoded_texts"][text]
    rows = []
    for item in contexts:
        expected = item.expected
        assert expected["text_counter_profile"] == profile.id and "glm" in item.version
        visible = sum(
            len(oracle["texts"][message["content"]]) for message in item.payload["messages"]
        )
        assert visible == expected["visible_prompt_tokens"]
        assert expected["minimum_context_tokens"] == visible + cfg["long_context"]["output_cap"]
        if expected["check_id"] == "context":
            assert (
                expected["length"]
                <= visible
                <= expected["length"] + cfg["long_context"]["maximum_length_excess"]
            )
            text = item.payload["messages"][0]["content"]
            marker = expected["mock_task"]["keys"]["single"] + "=" + expected["answers"]["single"]
            assert expected["needle_position_tokens"] == len(
                oracle["texts"][text[: text.index(marker)]]
            )
        rows.append(
            {"item_id": item.item_id, "check": expected["check_id"], "visible_tokens": visible}
        )
    for item in outputs:
        if item.expected["check_id"].startswith("R03"):
            expected = item.expected
            target = int(expected["check_id"].split(".")[1])
            count = len(oracle["texts"][expected["full_text"]])
            previous = render_output(expected["start"], expected["count"] - 1, expected["kind"])
            assert len(oracle["texts"][previous]) < target <= count
            assert expected["minimum_output_tokens"] == count + cfg["reasoning"]["output_headroom"]
            assert expected["visible_output_tokens"] == count and "glm" in item.version
    for item in protocols:
        if item.expected["check_id"] == "P12":
            count = len(oracle["texts"][item.payload["messages"][0]["content"]])
            assert item.expected["visible_prompt_tokens"] == count > 1024
    assert "glm" in next(item.version for item in structured if item.expected["check_id"] == "J05")
    text = "é café 北京"
    count = len(oracle["texts"][text])
    item = probe("P01", 0, "native", "Original cap case", {}, extra={"full_text": text})
    record = exchange(
        item_id=item.item_id,
        request={
            "messages": [{"role": "user", "content": "Original cap case"}],
            "max_tokens": count,
        },
        response={"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": text}}]},
        usage={"completion_tokens": count, "completion_tokens_details": {"reasoning_tokens": 0}},
    )
    assert not invariant_violation(item, record, load_policy(), profile=profile)
    record.request["max_tokens"] = count - 1
    assert invariant_violation(item, record, load_policy(), profile=profile)
    receipt = {
        "version": "model-b-native-geometry-v1",
        "source": capture_source()["source_identity"],
        "original_texts": len(texts),
        "context_and_prefix_items": len(contexts),
        "long_output_items": 4,
        "rows": rows,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "all_native_ids_and_decoding_match_official": True,
        "cap_known_answers_match": True,
        "model_api_calls": 0,
        "new_spend_usd": 0,
        "scope": (
            "local tokenizer/corpus/wire projection only; no GLM provider or weight validation"
        ),
    }
    (tmp_path / "native-geometry-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
