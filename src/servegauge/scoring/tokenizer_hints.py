"""B05 claimed-tokenizer prompt-count hints, calibrated like B01, never identity proof."""

from __future__ import annotations

from collections import defaultdict

from servegauge.core.models import Exchange, Item, ModelProfile
from servegauge.scoring.accounting_scoring import accounting_samples
from servegauge.scoring.common import Samples
from servegauge.suites.s7_accounting import TOKENIZER_CONTRACT, tokenizer_contract
from servegauge.tokencount.profile import counter_kind


def tokenizer_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint: str,
    profile: ModelProfile,
) -> dict[str, Samples]:
    contract = tokenizer_contract(items)
    if contract is None:
        return {}
    probes = [item for item in items if item.version == TOKENIZER_CONTRACT["version"]]
    captures: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint:
            captures[record.item_id].append(record)
    result = Samples(check_id="B05", check_class="comparative", binary=False, two_sided=True)
    if counter_kind(profile) is None:
        result.mark("INCONCLUSIVE", "verified pinned claimed-tokenizer counter unavailable")
    else:
        # Reuse the exact verified template/count projection, preserving B01's historical scorer.
        counted = [
            item.model_copy(deep=True, update={"expected": {**item.expected, "check_id": "B01"}})
            for item in probes
        ]
        result = accounting_samples(counted, records, endpoint, profile, cfg=contract)["B01"]
        result.check_id = "B05"
    result.details.update(
        tokenizer_contract=contract,
        claimed_tokenizer=profile.tokenizer,
        model_identity_proof=False,
        scope="tokenizer-sensitive count residual; native template/normalization may explain it",
        diagnosis_requires="two distinct same-run fingerprint failures in different-model bands",
        interpretation="count hint only; no alternative tokenizer, weights or hardware identified",
    )
    for item in probes:
        available = sorted(captures[item.item_id], key=lambda r: r.sample_idx)
        result.evidence.extend(
            r.exchange_id for r in available if r.exchange_id not in result.evidence
        )
        if [r.sample_idx for r in available] != list(range(int(item.expected.get("samples", 1)))):
            result.values.clear()
            result.repeats.clear()
            result.mark("ERROR", "tokenizer hint capture missing or duplicated")
        elif any(r.status != 200 or r.metadata.get("request_not_sent") for r in available):
            result.mark("INCONCLUSIVE", "tokenizer hint group has unavailable/rejected captures")
    return {"B05": result}
