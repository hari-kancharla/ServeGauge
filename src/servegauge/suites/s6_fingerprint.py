"""Versioned collection design, kept separate from frozen historical suite settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from servegauge.core.hashing import content_hash
from servegauge.core.models import Item
from servegauge.embeddings.local import LocalEmbeddingSpec
from servegauge.items.fp_prompts import CHOICES, INSTRUCTIONS, LANGUAGES, TOPICS
from servegauge.suites.s1_protocol import probe


def fingerprint_items(level: str, seed: int, overrides: dict[str, Any] | None = None) -> list[Item]:
    cfg = yaml.safe_load(Path("configs/fingerprints.yaml").read_text(encoding="utf-8"))
    if level not in {"smoke", "core", "full"}:
        raise ValueError("known_fingerprint_level_required")
    sizes = dict(cfg[level])
    overrides = dict(overrides or {})
    embedding = None
    embedding_cfg = None
    if isinstance(overrides.get("methods"), list) and "FP4" in overrides["methods"]:
        embedding_cfg = yaml.safe_load(
            Path("configs/fingerprints_embed.yaml").read_text(encoding="utf-8")
        )
        sizes.update(embedding_cfg[level])
        manifest = overrides.pop("embedding_model", None)
        embedding = (
            LocalEmbeddingSpec.model_validate(manifest)
            if manifest is not None
            else LocalEmbeddingSpec.model_validate_json(
                Path(embedding_cfg["model_manifest"]).read_bytes()
            )
        )
    if overrides and set(overrides) - set(sizes):
        raise ValueError("unknown_fingerprint_design_override")
    sizes.update(overrides or {})
    methods = sizes["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(m, str) for m in methods)
        or len(methods) != len(set(methods))
        or any(m not in {"FP1", "FP2", "FP3", "FP4"} for m in methods)
    ):
        raise ValueError("known_unique_fingerprint_methods_required")
    for key, limit in (
        ("fp1_cells", 40),
        ("fp2_cells", 25),
        ("fp1_samples", 100000),
        ("fp2_samples", 100000),
        ("fp3_samples", 100000),
    ):
        if type(sizes[key]) is not int or not 2 <= sizes[key] <= limit:
            raise ValueError("bounded_positive_fingerprint_design_required")
    if embedding is not None and any(
        type(sizes[k]) is not int or not 2 <= sizes[k] <= limit
        for k, limit in [("fp4_cells", 25), ("fp4_samples", 100000)]
    ):
        raise ValueError("bounded_positive_embedding_design_required")
    design = {
        "version": cfg["version"],
        "seed": seed,
        "level": level,
        "sizes": sizes,
        "output_cap": cfg["output_cap"],
        "fp2_length": cfg["fp2_length"],
        "top_logprobs": cfg["top_logprobs"],
        "anchor_tokens": cfg["anchor_tokens"],
        "minimum_independent_groups": cfg["minimum_independent_groups"],
        "minimum_anchor_tokens": cfg["minimum_anchor_tokens"],
        "prompt_hash": content_hash([CHOICES, INSTRUCTIONS, LANGUAGES, TOPICS]),
    }
    result = []
    if embedding is not None:
        assert embedding_cfg is not None
        design.update(
            embedding_model=embedding.model_dump(mode="json"),
            embedding_version=embedding_cfg["version"],
        )
    for method in methods:
        count = (
            sizes["fp1_cells"]
            if method == "FP1"
            else sizes["fp2_cells"]
            if method == "FP2"
            else sizes["fp4_cells"]
            if method == "FP4"
            else 1
        )
        for index in range(count):
            choices = list(CHOICES[(index // 4) % len(CHOICES)] if method == "FP1" else CHOICES[0])
            prompt = (
                INSTRUCTIONS[index % 4] + ", ".join(choices) + "."
                if method == "FP1"
                else (
                    f"Write a short original description of {TOPICS[index]}. "
                    "Return only the description."
                )
                if method in {"FP2", "FP4"}
                else INSTRUCTIONS[0] + ", ".join(choices) + "."
            )
            params = {
                "temperature": 1.0,
                "max_tokens": cfg["output_cap"],
                "reasoning_effort": "low",
            }
            if method == "FP3":
                params.update(logprobs=True, top_logprobs=cfg["top_logprobs"])
            item = probe(
                method,
                index,
                "fingerprint",
                prompt,
                {"type": "sample", "choices": choices},
                params=params,
                samples=sizes[method.lower() + "_samples"],
                extra={
                    "design": design,
                    "design_hash": content_hash(design),
                    "language": LANGUAGES[index % 4] if method == "FP1" else "en",
                    "fixed_length": cfg["fp2_length"] if method == "FP2" else None,
                    "requires": ["logprobs"] if method == "FP3" else [],
                },
            )
            item.suite_id, item.version = "S6", cfg["version"]
            if method == "FP4":
                item.version = design["embedding_version"]
            result.append(item)
    return result
