"""Explicit local embedding runtime for captured reference-pack operations."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from servegauge.core.errors import SafeError
from servegauge.core.hashing import canonical_json
from servegauge.embeddings.local import LocalEmbeddingSpec, LocalEncoder
from servegauge.scoring.common import Samples


@lru_cache(maxsize=2)
def _load(manifest: bytes, source: str, cache: str) -> LocalEncoder:
    spec = LocalEmbeddingSpec.model_validate_json(manifest)
    return LocalEncoder.load(spec, Path(source), Path(cache))


def get_encoder(sample: Samples) -> LocalEncoder:
    source = os.environ.get("SERVEGAUGE_EMBED_ASSETS")
    if not source:
        raise ValueError("local_embedding_assets_required")
    spec = LocalEmbeddingSpec.model_validate(sample.details["design"]["embedding_model"])
    try:
        return _load(
            canonical_json(spec.model_dump(mode="json")),
            str(Path(source).absolute()),
            str(Path(os.environ.get("SERVEGAUGE_EMBED_CACHE", ".cache/embeddings")).absolute()),
        )
    except SafeError as error:
        raise ValueError(error.code) from None
