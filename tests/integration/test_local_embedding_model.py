"""Optional real-weight acceptance; assets must be supplied, never downloaded here."""

import os
from pathlib import Path

import numpy as np
import pytest

from servegauge.core.errors import SafeError
from servegauge.embeddings.local import LocalEmbeddingSpec, LocalEncoder

pytestmark = pytest.mark.slow


def test_pinned_real_model_cpu_repeatability_order_and_truncation(tmp_path):
    assets = os.environ.get("SERVEGAUGE_EMBED_ASSETS")
    if not assets:
        pytest.skip("Set SERVEGAUGE_EMBED_ASSETS to the pinned local MiniLM asset folder")
    manifest = Path(__file__).resolve().parents[2] / "configs/embedding_minilm_v1.json"
    spec = LocalEmbeddingSpec.model_validate_json(manifest.read_bytes())
    encoder = LocalEncoder.load(spec, Path(assets), tmp_path / "models")
    texts = [
        "A red boat crosses the river.",
        "A scarlet boat sails across the river.",
        "The keyboard requires two batteries.",
    ]
    first = encoder.encode(texts)
    assert first.shape == (3, 384)
    assert np.array_equal(first, encoder.encode(texts))
    assert np.allclose(np.linalg.norm(first, axis=1), 1, rtol=0, atol=2e-6)
    assert np.allclose(first[::-1], encoder.encode(texts[::-1]), rtol=0, atol=2e-6)
    assert np.allclose(first[:1], encoder.encode(texts[:1]), rtol=0, atol=2e-6)
    similarity = first @ first.T
    assert similarity[0, 1] > similarity[0, 2]
    reloaded = LocalEncoder.load(spec, Path(assets), tmp_path / "models")
    assert np.array_equal(first, reloaded.encode(texts))
    with pytest.raises(SafeError, match="embedding_input_would_be_truncated"):
        encoder.encode(["boat " * 300])
    assert encoder.identity["device"] == "cpu"
    assert encoder.identity["artifact_hash"] == spec.artifact_hash

    # Exercise the actual vectors without claiming a provider or model-power trial.
    from servegauge.stats.energy import energy_distance
    from servegauge.stats.fingerprint_energy import energy_cell, energy_randomization_test

    a = [first[[0, 1]], first[[1, 0]]]
    b = [first[[2, 2]], first[[2, 2]]]
    cell = energy_cell([a, b, b, a])
    assert cell.distance(cell.a, cell.b) == pytest.approx(
        energy_distance(np.concatenate(a), np.concatenate(b)), abs=1e-12
    )
    result = energy_randomization_test([cell], seed=41, permutations=2000)
    assert result.n == 2 and result.exact and result.permutations == 4
