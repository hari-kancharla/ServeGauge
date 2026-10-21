import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from servegauge.core.errors import SafeError
from servegauge.embeddings.local import LocalEmbeddingSpec, LocalEncoder, materialize_model


def fixture_model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = {
        "modules.json": json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        ).encode(),
        "config.json": b'{"model_type":"bert"}',
        "model.safetensors": b"fixture bytes, not real model weights",
        "tokenizer.json": b'{"fixture":true}',
        "1_Pooling/config.json": b'{"pooling_mode_mean_tokens":true}',
        "LICENSE": b"Fixture only; not a model license claim",
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    spec = LocalEmbeddingSpec(
        model_id="fixture-local-model",
        revision="a" * 40,
        dimensions=2,
        max_tokens=16,
        normalize_embeddings=True,
        license_file="LICENSE",
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    return source, spec


def test_materialized_assets_are_content_addressed_and_ignore_unpinned_files(tmp_path):
    source, spec = fixture_model(tmp_path)
    (source / "unlisted.json").write_text("not part of this artifact")
    cache = tmp_path / "cache"
    destination = materialize_model(spec, source, cache)
    assert destination.name == spec.artifact_hash
    assert not (destination / "unlisted.json").exists()
    assert materialize_model(spec, source, cache) == destination
    assert (destination / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()


@pytest.mark.parametrize(
    "case",
    ["changed_source", "changed_cache", "unlisted_cache", "remote_module", "pickle", "path_escape"],
)
def test_unpinned_or_executable_model_inputs_are_refused(tmp_path, case):
    source, spec = fixture_model(tmp_path)
    cache = tmp_path / "cache"
    if case == "changed_source":
        (source / "tokenizer.json").write_text("changed")
    elif case in {"changed_cache", "unlisted_cache"}:
        path = materialize_model(spec, source, cache)
        if case == "changed_cache":
            (path / "tokenizer.json").write_text("changed")
        else:
            (path / "extra.json").write_text("unlisted")
    elif case == "remote_module":
        data = b'[{"idx":0,"name":"0","path":"","type":"arbitrary_custom_code.Module"}]'
        (source / "modules.json").write_bytes(data)
        spec.files["modules.json"] = hashlib.sha256(data).hexdigest()
    elif case == "pickle":
        spec.files["pytorch_model.bin"] = "0" * 64
    else:
        spec.files["../outside.json"] = "0" * 64
    with pytest.raises((ValueError, SafeError)):
        materialize_model(spec, source, cache)


class FakeModel:
    max_seq_length = 32
    device = "cpu"
    dtype = "torch.float32"

    def eval(self):
        return self

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False, "truncation": False}
        return {"input_ids": [list(range(len(text) + 2)) for text in texts]}

    def encode(self, texts, **kwargs):
        assert kwargs["device"] == "cpu" and kwargs["precision"] == "float32"
        assert kwargs["normalize_embeddings"] is True and kwargs["prompt"] == ""
        assert kwargs["convert_to_numpy"] is True and kwargs["convert_to_tensor"] is False
        x = np.array([[len(t), sum(map(ord, t)) % 7 + 1] for t in texts], dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_local_encoder_preserves_order_and_records_pinned_configuration(tmp_path, monkeypatch):
    import servegauge.embeddings.local as module

    source, spec = fixture_model(tmp_path)
    model = FakeModel()
    monkeypatch.setattr(module, "_sentence_model", lambda path, spec: model)
    encoder = LocalEncoder.load(spec, source, tmp_path / "cache")
    values = encoder.encode(["red", "blue"])
    assert values.shape == (2, 2) and np.allclose(np.linalg.norm(values, axis=1), 1)
    assert encoder.identity["artifact_hash"] == spec.artifact_hash
    assert encoder.identity["device"] == "cpu"
    assert model.max_seq_length == 16
    assert np.array_equal(values, encoder.encode(["red", "blue"]))


@pytest.mark.parametrize(
    "case", ["empty", "too_long", "nonfinite", "wrong_shape", "zero_normalized"]
)
def test_missing_truncated_or_invalid_embeddings_never_become_observations(
    tmp_path, monkeypatch, case
):
    import servegauge.embeddings.local as module

    source, spec = fixture_model(tmp_path)
    model = FakeModel()
    monkeypatch.setattr(module, "_sentence_model", lambda path, spec: model)
    encoder = LocalEncoder.load(spec, source, tmp_path / "cache")
    texts = ["red"]
    if case == "empty":
        texts = [""]
    elif case == "too_long":
        texts = ["long input exceeding the pinned tokenizer window"]
    else:
        result = {
            "nonfinite": np.array([[np.nan, 0]]),
            "wrong_shape": np.ones((1, 3)),
            "zero_normalized": np.zeros((1, 2)),
        }[case]
        model.encode = lambda *args, **kwargs: result
    with pytest.raises(SafeError):
        encoder.encode(texts)


def test_real_loader_is_optional_and_never_imported_on_module_import():
    # A fresh process also makes this valid when optional real tests ran earlier.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import servegauge.embeddings.local; "
            "assert 'sentence_transformers' not in sys.modules",
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "attribute,value", [("device", "mps"), ("dtype", "torch.float16"), ("max_seq_length", 8)]
)
def test_runtime_must_match_pinned_cpu_precision_and_window(
    tmp_path, monkeypatch, attribute, value
):
    import servegauge.embeddings.local as module

    source, spec = fixture_model(tmp_path)
    model = FakeModel()
    setattr(model, attribute, value)
    monkeypatch.setattr(module, "_sentence_model", lambda path, spec: model)
    with pytest.raises(SafeError, match="pinned_cpu_embedding_configuration_unavailable"):
        LocalEncoder.load(spec, source, tmp_path / "cache")


def test_missing_optional_runtime_is_a_controlled_error(tmp_path, monkeypatch):
    import servegauge.embeddings.local as module

    source, spec = fixture_model(tmp_path)

    def unavailable(name):
        raise ImportError("upstream text must not escape")

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    with pytest.raises(
        SafeError, match="^optional_embed_runtime_unavailable_install_locked_extra$"
    ):
        LocalEncoder.load(spec, source, tmp_path / "cache")
