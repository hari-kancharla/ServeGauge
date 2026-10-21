"""Standalone finite experiment arguments shared by both remote receivers."""

from __future__ import annotations

from typing import Any

STUDY_PROFILE = "qwen3-0.6b-study"
STUDY_COLLECTION = "qwen3-study-collection-v1"
STUDY_MODEL = "Qwen/Qwen3-0.6B"
STUDY_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
STUDY_ENGINES = {
    "0.31.0": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    "0.32.0": "5df63a7203cdb9328d824a27b682e77c7b9928be",
}
# Original deliberately incorrect renderer, not a copied maker template.
STUDY_WRONG_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}"
    "\\n{% endfor %}assistant: "
)


def study_engine_args(config: dict[str, Any]) -> list[str]:
    """Finite configured interventions; these arguments do not attest numeric kernels."""
    version = config.get("engine_version")
    precision, template = config.get("precision"), config.get("template")
    if (
        config.get("version") != "selfhost-study-config-v1"
        or config.get("purpose") != "perturbation-study"
        or config.get("profile") != STUDY_PROFILE
        or config.get("model_repository") != STUDY_MODEL
        or config.get("model_revision") != STUDY_REVISION
        or not isinstance(version, str)
        or version not in STUDY_ENGINES
        or config.get("engine_revision") != STUDY_ENGINES[version]
        or config.get("gpu") != "H100"
        or precision not in {"bf16", "fp8", "mxfp4-w4a16"}
        or template not in {"official", "wrong-v1"}
        or (template == "wrong-v1" and (precision != "bf16" or version != "0.32.0"))
        or (version == "0.31.0" and precision != "bf16")
    ):
        raise ValueError("selfhost_finite_study_recipe_required")
    args = [
        "vllm",
        "serve",
        STUDY_MODEL,
        "--revision",
        STUDY_REVISION,
        "--tokenizer-revision",
        STUDY_REVISION,
        "--served-model-name",
        STUDY_PROFILE,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--max-model-len",
        "8192",
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if precision == "fp8":
        args.extend(["--quantization", "fp8_per_tensor"])
    elif precision == "mxfp4-w4a16":
        args.extend(["--quantization", "mxfp4", "--linear-backend", "marlin"])
    if template == "wrong-v1":
        args.extend(["--chat-template", STUDY_WRONG_TEMPLATE])
    return args
