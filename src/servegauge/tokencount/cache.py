"""Fetch pinned public vocabulary data only during explicit setup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

VOCAB_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
HF_MODEL = "openai/gpt-oss-120b"
HF_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
HF_FILES = {
    "tokenizer.json": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    "chat_template.jinja": "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
}


def cache_dir() -> Path:
    return Path(
        os.environ.get(
            "SERVEGAUGE_TOKENIZER_CACHE",
            os.environ.get("TRUEWEIGHTS_TOKENIZER_CACHE", ".cache/tokenizers"),
        )
    ).resolve()


def configure_offline() -> Path:
    path = cache_dir()
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(path)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return path


def setup_cache() -> Path:
    path = configure_offline()
    target = path / "o200k_base.tiktoken"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != VOCAB_SHA256:
        response = httpx.get(VOCAB_URL, timeout=60, follow_redirects=False)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != VOCAB_SHA256:
            raise ValueError("Tokenizer vocabulary checksum mismatch")
        path.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(target)
    # tiktoken uses a SHA-1 URL cache key; harmony uses the original basename.
    cache_key = hashlib.sha1(VOCAB_URL.encode(), usedforsecurity=False).hexdigest()
    (path / cache_key).write_bytes(target.read_bytes())
    for name, checksum in HF_FILES.items():
        hf_target = path / "hf-gpt-oss" / name
        if hf_target.exists() and hashlib.sha256(hf_target.read_bytes()).hexdigest() == checksum:
            continue
        response = httpx.get(
            f"https://huggingface.co/{HF_MODEL}/resolve/{HF_REVISION}/{name}",
            timeout=60,
            follow_redirects=True,
            trust_env=False,
        )
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != checksum:
            raise ValueError("Tokenizer template checksum mismatch")
        hf_target.parent.mkdir(parents=True, exist_ok=True)
        temporary = hf_target.with_suffix(".download")
        temporary.write_bytes(response.content)
        temporary.replace(hf_target)
    return target
