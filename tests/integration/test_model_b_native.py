"""Opt-in comparison with actual independent local Transformers tokenization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from servegauge.storage.accounting_identity import accounting_counter_identity
from servegauge.storage.source_identity import capture_source
from servegauge.tokencount.glm_assets import FILES, cache_directory, verified_assets
from servegauge.tokencount.glm_counter import (
    glm_decode,
    glm_text_ids,
    normalize_messages,
    render_glm_chat,
)


@pytest.mark.slow
@pytest.mark.parametrize("corpus_name", ["model-b-cases-v1.json", "model-b-extra-cases-v1.json"])
def test_official_local_tokenizer_ids_and_rendering_agree(tmp_path, corpus_name):
    executable = os.environ.get("SERVEGAUGE_GLM_REFERENCE_PYTHON")
    if not executable:
        pytest.skip("explicit local Transformers runtime and checksum-pinned GLM assets required")
    assert verified_assets()
    corpus_path = (Path("tests/fixtures") / corpus_name).absolute()
    corpus = json.loads(corpus_path.read_bytes())
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ},
    }
    result = subprocess.run(
        [
            executable,
            str(Path("tests/model_b_official.py").absolute()),
            str(cache_directory()),
            str(corpus_path),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    official = json.loads(result.stdout)
    assert official["weights_loaded"] is False and official["model_inference_calls"] == 0
    assert official["assets"] == FILES
    assert official["transformers"] == "5.20.1"
    assert official["tokenizers"] == "0.23.3" and official["jinja2"] == "3.1.6"
    for text, ids in official["texts"].items():
        assert glm_text_ids(text) == ids
        assert glm_decode(ids) == official["decoded_texts"][text]
    for case in corpus["chats"]:
        assert normalize_messages(case["messages"]) == case["native_messages"]
        rendered = render_glm_chat(
            case["messages"], tools=case.get("tools"), **case.get("controls", {})
        )
        expected = official["chats"][case["name"]]
        assert hashlib.sha256(rendered.encode()).hexdigest() == expected["render_sha256"]
        assert glm_text_ids(rendered) == expected["token_ids"]
    receipt = {
        "version": "model-b-official-tokenizer-v2",
        "source": capture_source()["source_identity"],
        "counter": accounting_counter_identity("glm-5.3"),
        "oracle_script_sha256": hashlib.sha256(
            Path("tests/model_b_official.py").read_bytes()
        ).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "reference": official,
        "matched_chats": len(corpus["chats"]),
        "matched_texts": len(corpus["texts"]),
        "scope": "local text/template agreement only; provider rendering and billing unmeasured",
    }
    (tmp_path / "official-tokenizer-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
