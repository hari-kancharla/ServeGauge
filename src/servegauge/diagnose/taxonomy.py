"""Observed failure hypotheses; causal alternatives and falsifiers remain explicit."""

from typing import Literal

from servegauge.core.models import FailureClass

type Code = Literal[
    "PARAM_IGNORED",
    "PARSER_LEAK",
    "PARSER_DROP",
    "TEMPLATE_MISMATCH",
    "GUIDED_DECODING_ABSENT",
    "CONTEXT_TRUNCATION",
    "ACCOUNTING_MISMATCH",
    "LONG_OUTPUT_FAULT",
    "CACHE_ANOMALY",
    "INFRA_UNRELIABLE",
    "ROUTING_INSTABILITY",
    "DIFFERENT_MODEL_LIKE",
    "QUANTIZATION_LIKE_DRIFT",
]

CLASSES: dict[str, FailureClass] = {
    code: "API_INCOMPATIBILITY"
    for code in (
        "PARAM_IGNORED",
        "PARSER_LEAK",
        "PARSER_DROP",
        "TEMPLATE_MISMATCH",
        "GUIDED_DECODING_ABSENT",
        "CONTEXT_TRUNCATION",
    )
}
CLASSES["ACCOUNTING_MISMATCH"] = "ACCOUNTING"
CLASSES.update(
    LONG_OUTPUT_FAULT="MODEL_BEHAVIOR",
    CACHE_ANOMALY="ROUTING_OR_INFRA",
    INFRA_UNRELIABLE="ROUTING_OR_INFRA",
    ROUTING_INSTABILITY="ROUTING_OR_INFRA",
    DIFFERENT_MODEL_LIKE="MODEL_BEHAVIOR",
    QUANTIZATION_LIKE_DRIFT="MODEL_BEHAVIOR",
)

ALTERNATIVES: dict[str, list[str]] = {
    "PARAM_IGNORED": ["parameter translation or parser mismatch", "response metadata error"],
    "PARSER_LEAK": ["model emitted literal formatting tokens", "response normalization error"],
    "PARSER_DROP": ["model did not generate the expected call", "stream assembly mismatch"],
    "TEMPLATE_MISMATCH": [
        "ordinary model accuracy difference",
        "unsupported tool/history contract",
    ],
    "GUIDED_DECODING_ABSENT": ["unsupported schema feature", "schema translation mismatch"],
    "CONTEXT_TRUNCATION": ["long-context retrieval error", "template/tokenizer overhead mismatch"],
    "ACCOUNTING_MISMATCH": [
        "different native tokenizer or template",
        "usage normalization or caching rules",
        "reasoning/output-channel counting convention",
        "price changed after discovery or account fee/discount rules differ",
    ],
    "LONG_OUTPUT_FAULT": [
        "output budget censoring",
        "ordinary structured-generation error",
        "response normalization",
    ],
    "CACHE_ANOMALY": [
        "deterministic model outputs",
        "sampling translation",
        "dependent request scheduling",
    ],
    "INFRA_UNRELIABLE": [
        "client or network failure",
        "transient rate limiting",
        "upstream service errors",
    ],
    "ROUTING_INSTABILITY": [
        "provider metadata naming or discovery-label mismatch",
        "gateway routing configuration or upstream metadata error",
        "exact serving tag and successful fallback sequence remain unverified",
    ],
    "DIFFERENT_MODEL_LIKE": [
        "different sampling or template",
        "reference misspecification",
        "serving or tokenizer change",
        "mixed routing or correlated serving behavior; no weight or hardware proof",
    ],
    "QUANTIZATION_LIKE_DRIFT": [
        "mixed routing or a model mixture with the same measured signature",
        "reference misspecification or correlated serving behavior",
        "sampling change",
        "template change",
        "engine change; no weight or hardware proof",
    ],
}
FALSIFIERS: dict[str, list[str]] = {
    "PARAM_IGNORED": ["matched replay demonstrates the requested parameter's specified effect"],
    "PARSER_LEAK": ["raw generated text and independent parsing show no marker leakage"],
    "PARSER_DROP": ["raw generated calls match correctly assembled response calls"],
    "TEMPLATE_MISMATCH": ["the same measured deficit persists with the official raw template"],
    "GUIDED_DECODING_ABSENT": [
        "the declared supported schema is enforced on matched repeat trials"
    ],
    "CONTEXT_TRUNCATION": ["raw prompt token IDs prove both canaries reached the model intact"],
    "ACCOUNTING_MISMATCH": [
        "documented native rendering/counting explains the calibrated residual"
    ],
    "LONG_OUTPUT_FAULT": [
        "uncensored long outputs meet the same verified rules on matched repeat items"
    ],
    "CACHE_ANOMALY": ["independent fresh/warm requests have matched repetition distributions"],
    "INFRA_UNRELIABLE": [
        "matched capture shows the apparent service errors originated exclusively in the client"
    ],
    "ROUTING_INSTABILITY": [
        "generation-id-linked replay metadata consistently matches the requested provider",
        "documented provider naming equivalence explains the apparent contradiction",
    ],
    "DIFFERENT_MODEL_LIKE": [
        "qualified matched-configuration references place the endpoint in the same-model bands"
    ],
    "QUANTIZATION_LIKE_DRIFT": [
        "matched sampling, templates and engines explain the observed fingerprint displacement"
    ],
}
