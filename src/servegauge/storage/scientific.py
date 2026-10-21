"""Offline command workflows. Stored outcomes are outputs, never scoring inputs."""

from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.refs.fingerprint import compare_fingerprint
from servegauge.refs.pack import load_pack, reference_for
from servegauge.runner.evaluate import evaluation_configuration, exit_code
from servegauge.runner.scheduler import _write_json
from servegauge.storage.analysis import Analysis, export_outcomes, reconstruct
from servegauge.storage.check_report import write_check_report
from servegauge.verdicts.config import DecisionPolicy


def score_run(
    root: Path,
    run_id: str,
    *,
    out: Path | None = None,
    redact_content: bool = False,
    reference: str | None = None,
) -> tuple[Analysis, list[dict[str, Any]]]:
    if reference:
        pack = load_pack(reference, root / "refs")
        value = {"pack": pack.model_dump(mode="json")}
        path = root / "data/raw" / safe_filename(run_id) / "reference.json"
        encoded = canonical_json({"value": value, "hash": content_hash(value)}) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError("immutable_reference_attachment_conflict")
        original = reconstruct(root / "data/raw", run_id)
        if pack.profile != original.plan.profile:
            raise ValueError("reference_profile_mismatch")
        matched = False
        configuration = evaluation_configuration(
            original.plan,
            original.items,
            DecisionPolicy.model_validate(original.metadata["policy"]),
        )
        policy = DecisionPolicy.model_validate(original.metadata["policy"])
        for endpoint, samples in original.samples.items():
            for check, sample in samples.items():
                if sample.metric == "fingerprint":
                    try:
                        compare_fingerprint(
                            pack,
                            sample,
                            target=endpoint,
                            method="FP3-mean" if check == "FP3" else check,
                            policy=policy,
                            seed=original.plan.seed,
                        )
                        matched = True
                    except (LookupError, ValueError, TypeError):
                        pass
                    continue
                if sample.check_class != "comparative" or sample.status or not sample.values:
                    continue
                try:
                    reference_for(
                        pack,
                        target=endpoint,
                        check=check,
                        item_ids=sorted(sample.values),
                        configuration=configuration,
                        input_hashes=sample.details.get("input_hashes"),
                    )
                    matched = True
                except (LookupError, ValueError):
                    continue
        if not matched:
            raise ValueError("reference_has_no_matching_qualified_views")
        path.write_bytes(encoded)
    analysis = reconstruct(root / "data/raw", run_id)
    analysis.metadata["exit_code"] = exit_code(
        analysis.checks,
        set(
            analysis.metadata.get("required_checks", [check.check_id for check in analysis.checks])
        ),
    )
    directory = root / "data/runs" / safe_filename(run_id)
    _write_json(directory / "outcomes.json", export_outcomes(analysis))
    found = diagnoses(analysis.checks)
    for diagnosis in found:
        diagnosis.bundle_path = str(
            write_bundle(analysis, diagnosis, root=root, redact_content=redact_content)
        )
    serialized = [diagnosis.model_dump(mode="json") for diagnosis in found]
    _write_json(directory / "diagnoses.json", serialized)
    metadata = {**analysis.metadata, "diagnoses": serialized}
    write_check_report(directory, analysis.checks, metadata=metadata)
    if out is not None:
        write_check_report(out, analysis.checks, metadata=metadata)
    return analysis, serialized
