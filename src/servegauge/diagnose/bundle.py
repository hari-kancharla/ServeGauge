"""Sanitized local evidence bundles, including irreversible content-only redaction."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json
from servegauge.core.ids import safe_filename
from servegauge.core.models import Diagnosis
from servegauge.core.redaction import Redactor
from servegauge.diagnose.issue import issue_markdown, markdown_text
from servegauge.diagnose.replay_expectations import expected_context
from servegauge.diagnose.summary import content_summary as content_summary
from servegauge.storage.analysis import Analysis


def write_bundle(
    analysis: Analysis, diagnosis: Diagnosis, *, root: Path, redact_content: bool = False
) -> Path:
    expectations = expected_context(diagnosis, analysis.checks)
    suffix = "-redacted-v2" if redact_content else "-v2"
    directory = root / "bundles" / (safe_filename(diagnosis.diagnosis_id) + suffix)
    if (
        redact_content
        and directory.exists()
        and any(
            path.name not in {"bundle.json", "README.md", "replay.py", "issue.md"}
            for path in directory.iterdir()
        )
    ):
        raise ValueError("redacted_bundle_directory_contains_unexpected_content")
    directory.mkdir(parents=True, exist_ok=True)
    records = [record for record in analysis.records if record.exchange_id in diagnosis.evidence]
    supporting = [
        check
        for check in analysis.checks
        if check.check_id in diagnosis.check_ids
        and check.endpoint_id == diagnosis.endpoint_id
        and check.run_id == diagnosis.run_id
    ]
    selected_items = {record.item_id for record in records}
    data = {
        "version": "bundle-v2",
        "redact_content": redact_content,
        "diagnosis": diagnosis.model_dump(mode="json"),
        "replay_expectations": expectations,
        "expected_vs_observed": [check.model_dump(mode="json") for check in supporting],
        "manifest": analysis.manifest,
        "configuration": analysis.plan.model_dump(mode="json"),
        "items": [
            item.model_dump(mode="json")
            for item in analysis.items
            if item.item_id in selected_items
        ],
        "exchanges": [record.model_dump(mode="json") for record in records],
        "analysis_metadata": analysis.metadata,
        "reference_pack": analysis.reference_pack.model_dump(mode="json")
        if analysis.reference_pack
        else None,
        "limitation": "observed behavior and competing explanations; "
        "no hardware or model-weight attestation",
    }
    if redact_content:
        data = {
            "version": "bundle-v2",
            "redact_content": True,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "evidence_hashes_and_structure": content_summary(data),
            "replay": "requires original private raw store; "
            "redacted bundle has no recoverable payload",
        }
    scrubber = Redactor.from_environment()
    (directory / "bundle.json").write_bytes(canonical_json(scrubber.redact(data)) + b"\n")
    replay_script = (
        '"""Use the installed reserved runner; pass --yes explicitly."""\n'
        "import subprocess\nimport sys\nfrom pathlib import Path\n\n"
        'raise SystemExit(subprocess.run([sys.executable, "-m", "servegauge.cli", "replay", '
        "str(Path(__file__).resolve().parent), *sys.argv[1:]], check=False).returncode)\n"
    )
    (directory / "replay.py").write_text(replay_script, encoding="utf-8")
    (directory / "issue.md").write_text(
        issue_markdown(
            diagnosis, supporting, redact_content=redact_content, sanitize=scrubber.text
        ),
        encoding="utf-8",
    )
    lines = [
        "# ServeGauge evidence",
        "",
        markdown_text(scrubber.text(diagnosis.summary)),
        "",
        "Confidence: " + diagnosis.confidence,
        "",
        "Alternatives: "
        + "; ".join(markdown_text(scrubber.text(v)) for v in diagnosis.alternatives),
        "Falsifiers: " + "; ".join(markdown_text(scrubber.text(v)) for v in diagnosis.falsifiers),
        "",
        "[Local issue draft](issue.md). This generated draft is refreshed by diagnose; "
        "copy it before editing.",
        "",
        "Replay through reservations: `uv run python replay.py --endpoint URL --yes`.",
        "Run from the original project so the raw evidence and budget ledger are available.",
        "Redacted bundles contain hashes and structure only; their content cannot be replayed.",
    ]
    if not redact_content and records:
        endpoint = next(
            endpoint for endpoint in analysis.plan.endpoints if endpoint.id == diagnosis.endpoint_id
        )
        replay_plan = analysis.plan.model_copy(update={"endpoints": [endpoint]})
        import yaml

        (directory / "proxy-plan.yaml").write_text(
            yaml.safe_dump(scrubber.redact(replay_plan.model_dump(mode="json"))), encoding="utf-8"
        )
        (directory / "request.json").write_bytes(
            canonical_json(scrubber.redact(records[0].request)) + b"\n"
        )
        key = endpoint.api_key_env or "SERVEGAUGE_REPLAY_KEY"
        backend = scrubber.text("proxy:" + endpoint.base_url)
        proxy_path = str(directory / "proxy-plan.yaml")
        posix = shlex.quote(backend) + " --proxy-plan " + shlex.quote(proxy_path)
        powershell = (
            "'"
            + backend.replace("'", "''")
            + "' --proxy-plan '"
            + proxy_path.replace("'", "''")
            + "'"
        )
        lines.extend(
            [
                "",
                "The curl example targets a local proxy that reserves every forwarded call.",
                "Start it from the project root with the bundle's proxy-plan.yaml:",
                "macOS:",
                "```",
                f"uv run servegauge mock serve --port 8019 --backend {posix} --yes",
                "```",
                "PowerShell:",
                "```",
                f"uv run servegauge mock serve --port 8019 --backend {powershell} --yes",
                "```",
                "",
                "macOS curl (run from this bundle directory):",
                "```",
                "curl http://127.0.0.1:8019/v1/chat/completions "
                '-H "Content-Type: application/json" '
                + f'-H "Authorization: Bearer ${key}" --data-binary @request.json',
                "```",
                "PowerShell curl.exe:",
                "```",
                "curl.exe http://127.0.0.1:8019/v1/chat/completions "
                '-H "Content-Type: application/json" '
                + f'-H "Authorization: Bearer $env:{key}" --data-binary @request.json',
                "```",
            ]
        )
    (directory / "README.md").write_text(scrubber.text("\n".join(lines) + "\n"), encoding="utf-8")
    return directory


def read_bundle(identifier: str, root: Path) -> dict[str, Any]:
    path = Path(identifier)
    if not path.is_file():
        if path.is_dir():
            path = path / "bundle.json"
        else:
            name = safe_filename(identifier)
            current = root / "bundles" / (name + "-v2") / "bundle.json"
            path = current if current.is_file() else root / "bundles" / name / "bundle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") not in {"bundle-v1", "bundle-v2"}:
        raise ValueError("unsupported_bundle_version")
    if data.get("redact_content"):
        raise ValueError("content_redacted_bundle_requires_original_private_evidence")
    Diagnosis.model_validate(data["diagnosis"])
    return dict(data)
