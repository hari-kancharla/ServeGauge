"""Captured reference/control packs for matched independent fingerprint components.

Calibration uses reference observations only. Different-model bands describe the
captured controls, not an identification guarantee for arbitrary unseen models.
"""

from __future__ import annotations

import copy
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, ReferencePack
from servegauge.embeddings.runtime import get_encoder
from servegauge.refs.fingerprint_qualification import qualify_data, target_binding
from servegauge.scoring.common import Samples
from servegauge.scoring.fingerprint_comparison import collection_cells, signature
from servegauge.stats.fingerprint_energy import (
    EnergyCell,
    calibrate_energy_cells,
    compare_energy_cells,
)
from servegauge.stats.fingerprint_strata import Cell, Comparison, calibrate_cells, compare_cells
from servegauge.verdicts.config import DecisionPolicy

if TYPE_CHECKING:
    from servegauge.refs.pack import ReferenceRun

VERSION = "captured-components-v1"
METRICS = ("FP1", "FP2", "FP3-mean", "FP3-dispersion", "FP4")


def band_position(
    delta: float,
    noise: float,
    different: list[float],
    *,
    separated: bool,
    split_beyond: bool = False,
) -> str:
    """Locate a signed contrast; a negative excursion is not between positive bands."""
    try:
        valid = (
            len(different) == 2
            and bool(np.all(np.isfinite([delta, noise, *different])))
            and noise >= 0
            and different[0] <= different[1]
            and type(separated) is bool
            and separated == (different[0] > noise)
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("finite_ordered_fingerprint_bands_required")
    if not separated:
        return "calibration-overlap"
    if delta < -noise:
        return "below-same-model-band"
    if delta <= noise:
        return "same-model-band"
    if delta < different[0]:
        return "between-bands"
    if not split_beyond:
        return "different-model-band-or-beyond"
    return "beyond-different-model-band" if delta > different[1] else "different-model-band"


def control_positions(
    metric: dict[str, Any], controls: list[dict[str, Any]], delta: float
) -> list[dict[str, Any]]:
    """Join each captured control's identity and band without changing the comparison."""
    bands, noise = metric["control_bands"], metric["noise_band"]
    if not controls or len(controls) != len(bands):
        raise ValueError("fingerprint_control_band_provenance_required")
    result = []
    for control, band in zip(controls, bands, strict=True):
        fields = ("run_id", "endpoint_id", "model_identity", "capture_hash")
        if any(not isinstance(control.get(k), str) or not control[k] for k in fields):
            raise ValueError("fingerprint_control_band_provenance_required")
        interval = band.get("interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("finite_ordered_fingerprint_bands_required")
        try:
            separated = interval[0] > noise
            position = band_position(delta, noise, interval, separated=separated, split_beyond=True)
        except (ValueError, TypeError) as error:
            raise ValueError("finite_ordered_fingerprint_bands_required") from error
        result.append(
            {
                "control_run_id": control["run_id"],
                "control_endpoint_id": control["endpoint_id"],
                "model_identity": control["model_identity"],
                "capture_hash": control["capture_hash"],
                "different_model_band": list(interval),
                "bands_separated": separated,
                "position": position,
            }
        )
    return result


def _base(method: str) -> str:
    return "FP3" if method.startswith("FP3-") else method


def _descriptor(source: ReferenceRun, endpoint: str) -> EndpointSpec:
    declarations = source.manifest.get("endpoint_metadata", [])
    matches = [d for d in declarations if d.get("id") == endpoint]
    if len(matches) != 1:
        raise ValueError("captured_model_declaration_required")
    declaration = EndpointSpec.model_validate(matches[0])
    descriptors = [e for e in source.endpoints if e.id == endpoint]
    if len(descriptors) != 1 or (
        descriptors[0].kind != declaration.kind
        or descriptors[0].profile != declaration.claimed_profile
    ):
        raise ValueError("captured_model_descriptor_mismatch")
    if declaration.kind == "mock":
        mock = declaration.metadata.get("mock_declaration", {})
        if (
            type(mock.get("seed")) is not int
            or mock["seed"] != descriptors[0].seed
            or mock.get("clean") is not descriptors[0].clean
        ):
            raise ValueError("captured_mock_descriptor_mismatch")
    return declaration


def _identity(source: ReferenceRun, endpoint: str) -> str:
    declaration = _descriptor(source, endpoint)
    if declaration.kind == "mock":
        backend = declaration.metadata.get("mock_declaration", {}).get("backend")
        if not isinstance(backend, str) or not backend:
            raise ValueError("captured_model_declaration_required")
        return "mock:" + backend
    model = declaration.model
    if not isinstance(model, str) or not model:
        raise ValueError("captured_model_declaration_required")
    return "declared-model:" + model


def captured_controls(source: ReferenceRun, tier: str) -> list[ReferenceRun]:
    """Select actual captured mock control backends, never fault labels or invented data."""
    if tier != "MOCK":
        return []
    result = []
    selected = set()
    for endpoint in sorted(source.endpoints, key=lambda e: e.id):
        if endpoint.kind != "mock" or endpoint.clean or endpoint.id not in source.fingerprints:
            continue
        declaration = _descriptor(source, endpoint.id)
        backend = declaration.metadata.get("mock_declaration", {}).get("backend")
        if backend not in {"other:control-a", "other:control-b"} or backend in selected:
            continue
        selected.add(backend)
        result.append(
            source.model_copy(
                deep=True,
                update={
                    "endpoints": [endpoint.model_copy(deep=True)],
                    "fingerprints": {
                        endpoint.id: {
                            m: s.model_copy(deep=True)
                            for m, s in source.fingerprints[endpoint.id].items()
                        }
                    },
                    "checks": {},
                    "inputs": {},
                    "baseline_endpoint_id": None,
                },
            )
        )
    return result


def _trusted_endpoint(source: ReferenceRun, baseline: EndpointSpec) -> str:
    if source.baseline_endpoint_id is not None:
        endpoint = source.baseline_endpoint_id
    elif any(e.id == baseline.id for e in source.endpoints):
        endpoint = baseline.id
    elif len(source.endpoints) == 1:
        endpoint = source.endpoints[0].id
    else:
        raise ValueError("unambiguous_trusted_reference_endpoint_required")
    captured = _descriptor(source, endpoint)
    fields = ("kind", "model", "claimed_profile", "openrouter_pin", "defaults")
    if captured.base_url.rstrip("/") != baseline.base_url.rstrip("/") or any(
        getattr(captured, k) != getattr(baseline, k) for k in fields
    ):
        raise ValueError("additional_capture_requires_same_trusted_endpoint")
    return endpoint


def _verified(samples: dict[str, Samples]) -> set[str]:
    identities: list[str] = []
    for method, sample in samples.items():
        if method not in {"FP1", "FP2", "FP3", "FP4"} or sample.check_id != method:
            raise ValueError("known_captured_fingerprint_method_required")
        observations = sample.details.get("observations", [])
        if not observations or sample.details.get("observation_hash") != content_hash(observations):
            raise ValueError("captured_fingerprint_observation_hash_mismatch")
        ids = [row.get("exchange_id") for row in observations]
        if any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("complete_captured_fingerprint_identity_required")
        identities.extend(ids)
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("distinct_captured_fingerprint_observations_required")
    return set(identities)


def _minimum(cells: list[Cell], sample: Samples) -> int:
    minimum = sample.details["design"].get("minimum_independent_groups")
    if type(minimum) is not int or minimum < 2:
        raise ValueError("declared_minimum_independent_groups_required")
    n = sum(len(cell.a) for cell in cells)
    if n < minimum:
        raise ValueError("insufficient_independent_fingerprint_components")
    return n


def _orientations(peers: list[str]) -> list[list[str]]:
    if len(peers) < 3:
        raise ValueError("three_independent_reference_captures_required")
    # Balanced, predetermined cyclic triples; no data-based reference selection.
    return [[peers[(i + j) % len(peers)] for j in range(3)] for i in range(len(peers))]


def _comparison(
    cells: list[Cell], *, method: str, seed: int, permutations: int, resamples: int, alpha: float
) -> Comparison:
    if method == "FP4":
        return compare_energy_cells(
            cast(list[EnergyCell], cells),
            seed=seed,
            permutations=permutations,
            resamples=resamples,
            alpha=alpha,
        )
    return compare_cells(
        cells, seed=seed, permutations=permutations, resamples=resamples, alpha=alpha
    )


def _metric(
    references: dict[str, dict[str, Samples]],
    peers: list[str],
    controls: list[dict[str, Samples]],
    *,
    method: str,
    policy: DecisionPolicy,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    base = _base(method)
    encoder = get_encoder(references[peers[0]][base]) if method == "FP4" else None
    orientations = _orientations(peers)
    calibration: list[dict[str, Any]] = []
    control_results: list[list[dict[str, Any]]] = [[] for _ in controls]
    for index, triple in enumerate(orientations):
        samples = [references[peer][base] for peer in triple]
        named_seed = int(content_hash([seed, method, index])[:16], 16)
        for c, control in enumerate(controls):
            cells = collection_cells(
                [control[base], *samples], method=method, policy=policy, encoder=encoder
            )
            _minimum(cells, control[base])
            if c == 0:
                calibration.append(
                    calibrate_energy_cells(
                        cast(list[EnergyCell], cells),
                        seed=named_seed,
                        resamples=resamples,
                        alpha=policy.alpha,
                    )
                    if method == "FP4"
                    else calibrate_cells(
                        cells, seed=named_seed, resamples=resamples, alpha=policy.alpha
                    )
                )
            contrast = _comparison(
                cells,
                method=method,
                seed=named_seed,
                permutations=policy.permutation_resamples,
                resamples=resamples,
                alpha=policy.alpha / len(orientations),
            )
            control_results[c].append(
                {"delta": contrast.delta, "interval": [contrast.ci_low, contrast.ci_high]}
            )
    sizes = {v["n"] for v in calibration}
    if len(sizes) != 1:
        raise ValueError("matched_reference_component_counts_required")
    noise = max(float(v["noise_band"]) for v in calibration)
    bands: list[dict[str, Any]] = [
        {
            "delta": float(np.median([v["delta"] for v in rows])),
            "interval": [
                min(v["interval"][0] for v in rows),
                max(v["interval"][1] for v in rows),
            ],
        }
        for rows in control_results
    ]
    different = [min(v["interval"][0] for v in bands), max(v["interval"][1] for v in bands)]
    result = {
        "n": calibration[0]["n"],
        "signature": signature(references[peers[0]][base]),
        "orientations": orientations,
        "calibration": calibration,
        "noise_band": noise,
        "same_model_band": [-noise, noise],
        "control_bands": bands,
        "different_model_band": different,
        "bands_separated": different[0] > noise,
        "interval_method": "stratified component bootstrap hull; Bonferroni across orientations",
        "equivalence_qualified": False,
        "limitations": "Conditional empirical bands; interval coverage and equivalence power "
        "require separate qualification. Declarations cannot establish hidden independence.",
    }
    if encoder is not None:
        result["embedding_identity"] = copy.deepcopy(encoder.identity)
    return result


def build_fingerprints(
    source: ReferenceRun,
    views: dict[str, list[str]],
    controls: list[ReferenceRun],
    *,
    tier: str,
    seed: int,
    resamples: int,
    reference_runs: list[ReferenceRun] | None = None,
    qualification: str | None = None,
) -> dict[str, Any]:
    policy = source.fingerprint_policy
    if policy is None:
        raise ValueError("captured_fingerprint_policy_required")
    if len(controls) < 2:
        raise ValueError("two_captured_different_model_controls_required")
    control_samples, control_manifest, identities = [], [], []
    all_peers = sorted(set(chain.from_iterable(views.values())))
    if any(peer not in source.fingerprints for peer in all_peers):
        raise ValueError("reference_fingerprint_captures_missing")
    captures = {peer: (source, peer) for peer in all_peers}
    view_peers = {target: list(peers) for target, peers in views.items()}
    if reference_runs:
        if tier != "BASELINE" or len(all_peers) != 1:
            raise ValueError("additional_reference_runs_require_trusted_BASELINE")
        baseline = _descriptor(source, all_peers[0])
        used_runs = {source.run_id}
        for run in reference_runs:
            if (
                run.run_id in used_runs
                or run.fingerprint_policy != policy
                or run.profile != source.profile
            ):
                raise ValueError("distinct_matched_trusted_capture_runs_required")
            endpoint = _trusted_endpoint(run, baseline)
            if endpoint not in run.fingerprints:
                raise ValueError("reference_fingerprint_captures_missing")
            key = run.run_id + "/" + endpoint
            captures[key] = (run, endpoint)
            view_peers["*"].append(key)
            used_runs.add(run.run_id)
    references = {key: run.fingerprints[ep] for key, (run, ep) in captures.items()}
    reference_models = {_identity(run, ep) for run, ep in captures.values()}
    if len(reference_models) != 1:
        raise ValueError("same_model_reference_captures_required")
    seen: set[str] = set()
    mock_seeds: set[int] = set()
    for run, ep in [
        *captures.values(),
        *((c, c.endpoints[0].id) for c in controls if len(c.endpoints) == 1),
    ]:
        declaration = _descriptor(run, ep)
        if declaration.kind == "mock":
            mock_seed = int(declaration.metadata["mock_declaration"]["seed"])
            if mock_seed in mock_seeds:
                raise ValueError("distinct_mock_capture_seeds_required")
            mock_seeds.add(mock_seed)
    for samples in references.values():
        current = _verified(samples)
        if seen & current:
            raise ValueError("independent_reference_capture_evidence_required")
        seen |= current
    for control in controls:
        if control.fingerprint_policy != policy or len(control.endpoints) != 1:
            raise ValueError("single_matched_captured_model_control_required")
        endpoint = control.endpoints[0].id
        identity = _identity(control, endpoint)
        if identity in reference_models or identity in identities:
            raise ValueError("two_distinct_different_model_controls_required")
        samples = control.fingerprints.get(endpoint, {})
        current = _verified(samples)
        if seen & current:
            raise ValueError("independent_control_capture_evidence_required")
        seen |= current
        identities.append(identity)
        control_samples.append(samples)
        control_manifest.append(
            {
                "run_id": control.run_id,
                "endpoint_id": endpoint,
                "model_identity": identity,
                "capture_hash": content_hash(sorted(current)),
                "observation_hashes": {
                    m: s.details["observation_hash"] for m, s in samples.items()
                },
                "declarations": control.manifest["endpoint_metadata"],
            }
        )
    output: dict[str, Any] = {}
    for target, peers in view_peers.items():
        if tier == "R3" and target in peers:
            raise ValueError("R3_target_leakage")
        if len(peers) < 3:
            raise ValueError("three_independent_reference_captures_required")
        metrics, unavailable = {}, {}
        for method in METRICS:
            base = _base(method)
            if not any(base in references[p] for p in peers):
                continue
            try:
                metrics[method] = _metric(
                    references,
                    peers,
                    control_samples,
                    method=method,
                    policy=policy,
                    seed=int(content_hash([seed, target])[:16], 16),
                    resamples=resamples,
                )
            except KeyError:
                unavailable[method] = "matched_method_missing_in_reference_or_controls"
            except ValueError as error:
                unavailable[method] = str(error)
        output[target] = {
            "references": {
                p: {m: s.model_dump(mode="json") for m, s in references[p].items()} for p in peers
            },
            "metrics": metrics,
            "unavailable": unavailable,
        }
    result = {
        "version": VERSION,
        "policy": policy.model_dump(mode="json"),
        "controls": control_manifest,
        "views": output,
        "source_run": source.run_id,
    }
    if reference_runs:
        result["reference_sources"] = {
            key: {
                "run_id": run.run_id,
                "endpoint_id": ep,
                "declaration": _descriptor(run, ep).model_dump(mode="json"),
            }
            for key, (run, ep) in captures.items()
        }
    if qualification is not None:
        declarations = [_descriptor(run, ep) for run, ep in captures.values()] + [
            _descriptor(run, run.endpoints[0].id) for run in controls
        ]
        result = qualify_data(
            result,
            list(references.values()),
            control_samples,
            declarations,
            tier=tier,
            name=qualification,
        )
    return result


def _mock_seed(declaration: dict[str, Any]) -> int | None:
    metadata = declaration.get("metadata", {})
    recipe = metadata.get("mock_declaration", {}) if isinstance(metadata, dict) else {}
    seed = recipe.get("seed") if isinstance(recipe, dict) else None
    return seed if declaration.get("kind") == "mock" and type(seed) is int else None


def _known_mock_independence(
    pack: ReferencePack, data: dict[str, Any], view: dict[str, Any], endpoint: EndpointSpec | None
) -> None:
    if endpoint is None or endpoint.kind != "mock":
        return
    seed = _mock_seed(endpoint.model_dump(mode="json"))
    if seed is None:
        return
    peers = set(view["references"])
    source = pack.manifest.get("source_manifest", {})
    declarations = [d for d in source.get("endpoint_metadata", []) if d.get("id") in peers]
    declarations.extend(
        value["declaration"]
        for key, value in data.get("reference_sources", {}).items()
        if key in peers
    )
    declarations.extend(
        d
        for control in data["controls"]
        for d in control["declarations"]
        if d.get("id") == control["endpoint_id"]
    )
    if any(_mock_seed(d) == seed for d in declarations):
        raise ValueError("mock_target_seed_must_be_independent")


def compare_fingerprint(
    pack: ReferencePack,
    sample: Samples,
    *,
    target: str,
    method: str,
    policy: DecisionPolicy,
    seed: int,
    endpoint: EndpointSpec | None = None,
) -> dict[str, Any]:
    """Pre-family statistical evidence; never declares a provider verdict."""
    data = pack.fingerprints
    if data.get("version") != VERSION or data.get("policy") != policy.model_dump(mode="json"):
        raise ValueError("fingerprint_pack_policy_or_version_mismatch")
    view = data["views"][target if pack.tier == "R3" else "*"]
    metric = view["metrics"].get(method)
    if metric is None:
        raise ValueError("qualified_fingerprint_metric_unavailable")
    if sample.check_id != _base(method) or signature(sample) != metric["signature"]:
        raise ValueError("fingerprint_configuration_mismatch; recalibrate")
    _known_mock_independence(pack, data, view, endpoint)
    qualification = None
    if method in {"FP1", "FP2"} and data.get("qualification"):
        if pack.tier != "MOCK" or endpoint is None or endpoint.id != target:
            raise ValueError("qualification_mock_target_context_required")
        if metric.get("equivalence_qualified") is not True:
            raise ValueError("qualification_metric_binding_unavailable")
        qualification = target_binding(data, sample, endpoint, policy)
    encoder = get_encoder(sample) if method == "FP4" else None
    if encoder is not None and encoder.identity != metric.get("embedding_identity"):
        raise ValueError("embedding_runtime_mismatch; recalibrate")
    references = {
        peer: Samples.model_validate(samples[_base(method)])
        for peer, samples in view["references"].items()
    }
    _verified({sample.check_id: sample})
    contrasts = []
    for i, triple in enumerate(metric["orientations"]):
        if pack.tier == "R3" and target in triple:
            raise ValueError("R3_target_leakage")
        cells = collection_cells(
            [sample, *(references[p] for p in triple)],
            method=method,
            policy=policy,
            encoder=encoder,
        )
        n = _minimum(cells, sample)
        if n != metric["n"]:
            raise ValueError("fingerprint_requires_matched_components_and_n; recalibrate")
        contrasts.append(
            _comparison(
                cells,
                method=method,
                seed=int(content_hash([seed, method, i])[:16], 16),
                permutations=policy.permutation_resamples,
                resamples=policy.bootstrap_resamples,
                alpha=policy.alpha / len(metric["orientations"]),
            )
        )
    delta = float(np.median([v.delta for v in contrasts]))
    noise = metric["noise_band"]
    different = metric["different_model_band"]
    position = band_position(delta, noise, different, separated=metric["bands_separated"])
    return {
        "n": metric["n"],
        "estimate": float(np.median([v.estimate for v in contrasts])),
        "reference_estimate": float(np.median([v.reference_estimate for v in contrasts])),
        "ci_low": min(v.estimate_interval[0] for v in contrasts),
        "ci_high": max(v.estimate_interval[1] for v in contrasts),
        "reference_interval": [
            min(v.reference_interval[0] for v in contrasts),
            max(v.reference_interval[1] for v in contrasts),
        ],
        "delta": delta,
        "delta_ci_low": min(v.ci_low for v in contrasts),
        "delta_ci_high": max(v.ci_high for v in contrasts),
        # Intersection of predetermined orientation tests; no smallest-p selection.
        "p_value": max(v.p_value for v in contrasts),
        "median_absolute_deviation": float(np.median([abs(v.delta - delta) for v in contrasts])),
        "noise_band": noise,
        "same_model_band": metric["same_model_band"],
        "different_model_band": different,
        "position": position,
        "control_positions": control_positions(metric, data["controls"], delta),
        "equivalence_qualified": qualification is not None,
        "limitations": metric["limitations"],
        "interval_method": metric["interval_method"],
        "orientation_p_values": [v.p_value for v in contrasts],
        "groups_per_cell": metric["calibration"][0]["groups_per_cell"],
        **({"qualification": qualification} if qualification is not None else {}),
        **(
            {"embedding_identity": copy.deepcopy(metric["embedding_identity"])}
            if encoder is not None
            else {}
        ),
    }
