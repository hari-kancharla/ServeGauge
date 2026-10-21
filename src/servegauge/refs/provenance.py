"""Pure linkage for existing grouped scorer keys; licenses remain Item declarations."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from servegauge.core.models import Exchange, Item, ReferencePack
from servegauge.scoring.common import Samples


class OutcomeProvenanceError(ValueError):
    def __init__(self) -> None:
        super().__init__("outcome_provenance_unverified")


@dataclass(frozen=True)
class AggregateLink:
    run_id: str
    endpoint_id: str
    check_id: str
    outcome_key: str
    item_ids: tuple[str, ...]
    exchange_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeItems:
    item_ids: frozenset[str]
    aggregate_links: tuple[AggregateLink, ...]


@dataclass(frozen=True)
class _Contract:
    suite: str
    source_check: str
    versions: frozenset[str]
    variants: frozenset[str] | None


def _contract(check: str) -> _Contract | None:
    if check in {"J01", "J02.semantic", "J03"}:
        return _Contract(
            "S3", "J02" if check == "J02.semantic" else check, frozenset({"json-ladder-v1"}), None
        )
    variants: dict[str, frozenset[str]] = {
        "P03": frozenset({"low", "high"}),
        "P04.top_p": frozenset({"low", "high"}),
        "P04.top_k": frozenset({"low", "high"}),
        "P07": frozenset({"low", "high"}),
        "P13": frozenset({"required", "optional"}),
        "P06.calibration": frozenset({"logprobs"}),
    }
    if check not in variants:
        return None
    versions = {"protocol-v2"}
    if check == "P07":
        versions.add("protocol-effort-math-v1")
    return _Contract(
        "S1",
        "P06.structure" if check == "P06.calibration" else check,
        frozenset(versions),
        variants[check],
    )


def _trial(item: Item) -> str | None:
    value = item.expected.get("trial")
    if item.suite_id == "S1" and type(value) is int and value >= 0:
        return str(value)
    if item.suite_id == "S3" and isinstance(value, str) and value:
        try:
            return value if str(int(value)) == value else None
        except ValueError:
            return None
    return None


def _group(
    contract: _Contract,
    key: str,
    items: list[Item],
) -> list[Item]:
    candidates = [
        item
        for item in items
        if item.suite_id == contract.suite
        and item.expected.get("check_id") == contract.source_check
    ]
    if any(_trial(item) is None or item.version not in contract.versions for item in candidates):
        raise OutcomeProvenanceError()
    group = [item for item in candidates if _trial(item) == key]
    if not group:
        raise OutcomeProvenanceError()
    if contract.variants is None:
        if len(group) != 1:
            raise OutcomeProvenanceError()
    elif (
        len(group) != len(contract.variants)
        or {item.expected.get("variant") for item in group} != contract.variants
    ):
        raise OutcomeProvenanceError()
    return sorted(group, key=lambda item: item.item_id)


def outcome_items(
    pack: ReferencePack,
    items: list[Item],
    records: list[Exchange],
    samples: dict[str, dict[str, Samples]],
) -> OutcomeItems:
    """Resolve grouped inputs after raw reconstruction and exact numeric validation.

    This does not rescore, grant rights, or change a sampling unit/calibration. The
    caller validates pack outcomes against reconstructed scores separately. Direct
    Item keys retain their original binding; recognized grouped checks always use
    their scorer contract, even when an unrelated Item has a matching numeric ID.
    """
    by_id = {item.item_id: item for item in items}
    by_exchange = {record.exchange_id: record for record in records}
    if len(by_id) != len(items) or len(by_exchange) != len(records):
        raise OutcomeProvenanceError()
    run = pack.manifest.get("source_run")
    captures: dict[tuple[str, str], list[Exchange]] = defaultdict(list)
    for record in records:
        if record.run_id == run:
            captures[record.endpoint_id, record.item_id].append(record)
    selected: set[str] = set()
    required: set[tuple[str, str, str]] = set()
    for checks in pack.outcomes.values():
        for check, view in checks.items():
            for key in view["item_ids"]:
                if not isinstance(key, str) or not key:
                    raise OutcomeProvenanceError()
                if _contract(check) is None:
                    # A numeric label cannot borrow an unrelated direct Item's rights.
                    try:
                        numeric_label = str(int(key)) == key
                    except ValueError:
                        numeric_label = False
                    if key not in by_id or numeric_label:
                        raise OutcomeProvenanceError()
                    selected.add(key)
                    continue
                peers = view["peer_ids"]
                if (
                    not isinstance(peers, list)
                    or not peers
                    or any(not isinstance(peer, str) or not peer for peer in peers)
                    or len(set(peers)) != len(peers)
                ):
                    raise OutcomeProvenanceError()
                required.update((peer, check, key) for peer in peers)
    links = []
    for peer, check, key in sorted(required):
        if not isinstance(run, str) or not run:
            raise OutcomeProvenanceError()
        contract = _contract(check)
        assert contract is not None
        sample = samples.get(peer, {}).get(check)
        if (
            sample is None
            or sample.check_id != check
            or sample.check_class != "comparative"
            or sample.status is not None
            or key not in sample.values
            or not math.isfinite(sample.values[key])
        ):
            raise OutcomeProvenanceError()
        evidence = set(sample.evidence)
        if len(evidence) != len(sample.evidence) or any(
            identifier not in by_exchange
            or by_exchange[identifier].endpoint_id != peer
            or by_exchange[identifier].run_id != run
            or by_exchange[identifier].item_id not in by_id
            for identifier in evidence
        ):
            raise OutcomeProvenanceError()
        group = _group(contract, key, items)
        identifiers: set[str] = set()
        for item in group:
            count: Any = item.expected.get("samples", 1)
            observed = captures[peer, item.item_id]
            if (
                type(count) is not int
                or count < 1
                or len(observed) != count
                or {record.sample_idx for record in observed} != set(range(count))
                or any(
                    record.exchange_id not in evidence
                    or record.suite_id != item.suite_id
                    or record.error is not None
                    for record in observed
                )
            ):
                raise OutcomeProvenanceError()
            identifiers.update(record.exchange_id for record in observed)
        selected.update(item.item_id for item in group)
        links.append(
            AggregateLink(
                run,
                peer,
                check,
                key,
                tuple(item.item_id for item in group),
                tuple(sorted(identifiers)),
            )
        )
    return OutcomeItems(frozenset(selected), tuple(links))
