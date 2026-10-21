"""Cross-platform command line interface. Unbuilt commands fail closed."""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, NoReturn

import httpx
import typer
import uvicorn
import yaml
from pydantic import ValidationError

from servegauge.adapters.openrouter import (
    discover_model,
    fetch_catalog,
    fetch_external,
    pick_second,
)
from servegauge.config import load_budget
from servegauge.core.errors import SafeError
from servegauge.core.ids import safe_filename
from servegauge.core.models import ReferencePack
from servegauge.core.redaction import Redactor
from servegauge.core.secrets import scan_repository
from servegauge.diagnose.repro import replay_bundle
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.profiles import load_profile
from servegauge.refs.pack import ReferenceRun, build_pack, load_pack, save_pack
from servegauge.report.endpoint_report import write_endpoint_report
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.check import _mock_metadata, run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import Requirements, RunPlan, load_plan, load_requirements
from servegauge.runner.proxy import reserved_forwarder
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.scientific import score_run
from servegauge.storage.tables import rebuild as rebuild_tables
from servegauge.tokencount.cache import cache_dir, setup_cache

app = typer.Typer(
    no_args_is_help=True,
    help="Check inference endpoint requirements with evidence.",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


def not_implemented() -> None:
    typer.echo("not implemented yet")
    raise typer.Exit(3)


@app.command("init")
def init_workspace(path: Path = Path("."), evidence_root: Path | None = None) -> None:
    """Create offline defaults without replacing files; live and GPU caps start at zero."""
    from servegauge.storage.workspace import initialize

    try:
        result = initialize(path, evidence_root=evidence_root)
    except (OSError, ValueError, KeyError, TypeError):
        typer.echo(
            "Workspace initialization refused; check destination conflicts or evidence.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("setup-cache")
def setup_cache_command(profile: str | None = None) -> None:
    """Download and checksum public tokenizer data for an explicit profile (setup only)."""
    if profile not in {None, "gpt-oss-120b", "gpt-oss-20b", "glm-5.3"}:
        typer.echo("Tokenizer profile unsupported.", err=True)
        raise typer.Exit(3)
    try:
        if profile == "glm-5.3":
            from servegauge.tokencount.glm_assets import setup_glm_cache

            result = setup_glm_cache()
        else:
            result = setup_cache()
    except (OSError, ValueError, SafeError, httpx.HTTPError):
        typer.echo("Tokenizer setup failed; check network access and retry.", err=True)
        raise typer.Exit(5) from None
    typer.echo(f"Tokenizer vocabulary ready: {result}")


@app.command()
def doctor(profiles: bool = False, scan_secrets: bool = False) -> None:
    """Print safe readiness checks; never print credential values."""
    path = Path("budget/dev_spend.json")
    spend = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"spent_usd": 0}
    typer.echo(f"Development spend: ${float(spend['spent_usd']):.2f}")
    cached = (cache_dir() / "o200k_base.tiktoken").exists()
    typer.echo(f"Tokenizer cache: {'ready' if cached else 'missing; run setup-cache'}")
    typer.echo("Live calls: disabled without runner reservations and explicit --yes")
    if profiles:
        try:
            files = sorted(Path("profiles").glob("*.yaml"))
            if not files:
                raise ValueError("profiles_missing")
            for file in files:
                profile = load_profile(file.stem)
                typer.echo(
                    f"Profile {profile.id}: valid; unverified: "
                    f"{', '.join(profile.unverified) or 'none'}"
                )
        except (OSError, ValueError):
            typer.echo("Profile validation failed; check schema and field source URLs.", err=True)
            raise typer.Exit(3) from None
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and not scan_secrets and not profiles:
        try:
            response = httpx.get(
                "https://openrouter.ai/api/v1/key",
                headers={"authorization": "Bearer " + key},
                timeout=10,
                follow_redirects=False,
                trust_env=False,
            )
            data = response.json()["data"] if response.status_code == 200 else None
            if not isinstance(data, dict):
                raise ValueError("metadata unavailable")
            remaining = data.get("limit_remaining")
            if data.get("limit") is None or remaining is None:
                typer.echo("Key credit limit: missing; development live calls refused.")
                raise typer.Exit(4)
            credit = Decimal(str(remaining))
            limit = Decimal(str(data["limit"]))
            if not credit.is_finite() or not limit.is_finite() or limit < 0:
                raise ValueError("invalid remaining credit")
            if credit <= 0:
                typer.echo("Key credit exhausted; live calls refused.")
                raise typer.Exit(4)
            typer.echo(f"Key spending limit: configured; remaining ${credit:.2f}")
            typer.echo("Account balance: unverified; the key limit is not purchased credit.")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, InvalidOperation):
            typer.echo("Key credit metadata: unavailable; live preflight will refuse.")
            raise typer.Exit(5) from None
    if scan_secrets:
        findings = scan_repository(Path.cwd())
        for finding in findings:
            typer.echo(finding, err=True)
        if findings:
            raise typer.Exit(3)
        typer.echo("Secret scan: clean")


@app.command()
def rebuild(raw: Path = Path("data/raw"), out: Path = Path("data/tables")) -> None:
    """Rebuild Parquet tables and DuckDB catalog from sanitized append-only captures."""
    try:
        counts = rebuild_tables(raw, out)
    except (ValueError, OSError):
        typer.echo("Invalid or unreadable raw store; rebuild stopped.", err=True)
        raise typer.Exit(5) from None
    typer.echo(json.dumps(counts, sort_keys=True))


def _check_configuration(
    *,
    endpoint: str | None,
    model: str | None,
    profile: str | None,
    requires: Path | None,
    level: str,
    baseline: str | None,
    baseline_model: str | None,
    baseline_api_key_env: str | None,
    api_key_env: str | None,
    max_usd: float | None,
) -> tuple[RunPlan, Requirements | None, str | None, ReferencePack | None]:
    if endpoint is None or model is None or profile is None:
        raise ValueError("endpoint_model_profile_required")
    target = resolve_endpoint(endpoint, model, profile, api_key_env=api_key_env)
    endpoints = [target]
    baseline_id, pack = None, None
    if baseline:
        if baseline.startswith(("http://", "https://")):
            trusted = resolve_endpoint(
                baseline,
                baseline_model or model,
                profile,
                api_key_env=baseline_api_key_env,
                identifier="baseline",
            )
            endpoints.append(trusted)
            baseline_id = trusted.id
        else:
            pack = load_pack(baseline)
    budget = load_budget(Path("."))
    cap = float(budget.dev_live_usd) if max_usd is None else max_usd
    if all(e.kind == "mock" for e in endpoints) and max_usd is None:
        cap = 0
    config = RunPlan.model_validate(
        dict(
            plan_id="requirements-check",
            profile=profile,
            level=level,
            seed=20261018,
            reference={
                "tier": pack.tier if pack else "BASELINE" if baseline else "MOCK",
                "pack": pack.pack_id if pack else None,
            },
            endpoints=endpoints,
            suites=["S0", "S1"],
            max_usd=cap,
            requests_per_minute=60000 if all(e.kind == "mock" for e in endpoints) else 60,
        )
    )
    return config, load_requirements(requires) if requires else None, baseline_id, pack


@app.command("estimate")
def estimate_command(
    plan: Annotated[Path | None, typer.Option("--plan")] = None,
    check_args: Annotated[bool, typer.Option("--check-args")] = False,
    endpoint: str | None = None,
    model: str | None = None,
    profile: str | None = None,
    requires: Path | None = None,
    tests: str | None = None,
    level: str | None = None,
    baseline: str | None = None,
    baseline_model: str | None = None,
    baseline_api_key_env: str | None = None,
    api_key_env: str | None = None,
    max_usd: float | None = None,
) -> None:
    """Estimate either a saved plan or ordinary check arguments; no dispatch or writes."""
    try:
        supplied = (
            endpoint,
            model,
            profile,
            requires,
            tests,
            level,
            baseline,
            baseline_model,
            baseline_api_key_env,
            api_key_env,
            max_usd,
        )
        if plan is not None:
            if check_args or any(value is not None for value in supplied):
                raise ValueError("choose_one_estimate_mode")
            estimate = estimate_plan(load_plan(plan))
        else:
            if not check_args:
                raise ValueError("estimate_mode_required")
            config, requirement, baseline_id, pack = _check_configuration(
                endpoint=endpoint,
                model=model,
                profile=profile,
                requires=requires,
                level=level if level is not None else "smoke",
                baseline=baseline,
                baseline_model=baseline_model,
                baseline_api_key_env=baseline_api_key_env,
                api_key_env=api_key_env,
                max_usd=max_usd,
            )
            estimate = asyncio.run(
                run_check(
                    config,
                    tests=tests,
                    requirements=requirement,
                    baseline_id=baseline_id,
                    reference_pack=pack,
                    dry_run=True,
                )
            ).estimate
    except (ValueError, OSError):
        typer.echo("Invalid plan, inventory or price snapshot; no calls made.", err=True)
        raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(estimate, indent=2)))


@app.command("discover")
def discover_command(
    model: str = "openai/gpt-oss-120b",
    pick_second_model: Annotated[bool, typer.Option("--pick-second")] = False,
) -> None:
    """Fetch unpaid router metadata; retain status/capabilities and write an endpoint snapshot."""

    async def discovery() -> tuple[dict[str, Any], dict[str, Any] | None]:
        headers = {}
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            headers["authorization"] = "Bearer " + key
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            choice = (
                await pick_second(client, await fetch_catalog(client, headers), headers=headers)
                if pick_second_model
                else None
            )
            snapshot = (
                choice["snapshot"]
                if choice
                else await discover_model(client, model, headers=headers)
            )
            try:
                snapshot["external_signals"] = {
                    "status": "available",
                    "rows": await fetch_external(client, headers=headers),
                    "drives_verdict": False,
                }
            except SafeError:
                snapshot["external_signals"] = {
                    "status": "unavailable; authenticated capture pending",
                    "rows": [],
                    "drives_verdict": False,
                }
            return snapshot, choice

    try:
        snapshot, choice = asyncio.run(discovery())
        output = Path("configs/endpoints") / (safe_filename(snapshot["model"]) + ".yaml")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(Redactor.from_environment().redact(snapshot), sort_keys=False),
            encoding="utf-8",
        )
    except (ValueError, OSError, SafeError):
        typer.echo("Discovery unavailable or unverified metadata; no paid calls made.", err=True)
        raise typer.Exit(5) from None
    typer.echo(
        json.dumps(
            {
                "model": snapshot["model"],
                "snapshot": str(output),
                "endpoints": len(snapshot["endpoints"]),
                "active": sum(e["metadata"].get("status") == 0 for e in snapshot["endpoints"]),
                "selection": {k: v for k, v in choice.items() if k != "snapshot"}
                if choice
                else None,
            },
            indent=2,
        )
    )


def _budget_exception(error: BaseException) -> BudgetExceeded | None:
    if isinstance(error, BudgetExceeded):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            found = _budget_exception(child)
            if found:
                return found
    return None


@app.command("run")
def run_command(
    plan: Annotated[Path, typer.Option("--plan")],
    yes: bool = False,
    dry_run: bool = False,
    max_usd: float | None = None,
    resume: str | None = None,
) -> None:
    """Execute a validated plan only after an estimate and explicit --yes."""
    try:
        config = load_plan(plan)
        if max_usd is not None:
            if max_usd > config.max_usd:
                raise BudgetExceeded("max_usd_exceeds_plan")
            config = type(config).model_validate({**config.model_dump(), "max_usd": max_usd})
        if yes and not dry_run and config.suites != ["S0"]:
            asyncio.run(_mock_metadata(config, {}))
        typer.echo(json.dumps(estimate_plan(config), indent=2))
        if yes and dry_run:
            raise ValueError("choose_yes_or_dry_run")
        if dry_run:
            typer.echo("Dry run: no dispatch, reservations or fabricated results.")
            return
        if not yes:
            raise BudgetExceeded("explicit_yes_required")
        result = asyncio.run(
            execute_plan(config, run_id=resume, resume=resume is not None, approved=yes)
        )
    except BudgetExceeded as error:
        typer.echo(error.code, err=True)
        raise typer.Exit(4) from None
    except ExceptionGroup as error:
        budget_error = _budget_exception(error)
        typer.echo(
            budget_error.code
            if budget_error
            else "run_harness_failure; inspect sanitized captures",
            err=True,
        )
        raise typer.Exit(4 if budget_error else 5) from None
    except (OSError, ValueError):
        typer.echo(
            "Invalid plan or incompatible resume configuration; no further calls made.", err=True
        )
        raise typer.Exit(3) from None
    except SafeError as error:
        typer.echo(error.code, err=True)
        raise typer.Exit(5) from None
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "completed": result.completed,
                "skipped": result.skipped,
                "errors": result.errors,
                "spent_usd": str(result.spent_usd),
                "unmeasured": result.unmeasured,
            }
        )
    )
    if result.errors:
        raise typer.Exit(5)
    if result.unmeasured:
        raise typer.Exit(2)


@app.command("check")
def check_command(
    endpoint: str | None = None,
    model: str | None = None,
    profile: str | None = None,
    requires: Path | None = None,
    tests: str | None = None,
    level: str = "smoke",
    baseline: str | None = None,
    baseline_model: str | None = None,
    baseline_api_key_env: str | None = None,
    api_key_env: str | None = None,
    yes: bool = False,
    dry_run: bool = False,
    max_usd: float | None = None,
    redact_content: bool = False,
    out: Path | None = None,
) -> None:
    """Measure requirements with a paired baseline and replayed invariant evidence."""
    try:
        config, requirement, baseline_id, pack = _check_configuration(
            endpoint=endpoint,
            model=model,
            profile=profile,
            requires=requires,
            level=level,
            baseline=baseline,
            baseline_model=baseline_model,
            baseline_api_key_env=baseline_api_key_env,
            api_key_env=api_key_env,
            max_usd=max_usd,
        )
        result = asyncio.run(
            run_check(
                config,
                tests=tests,
                requirements=requirement,
                baseline_id=baseline_id,
                reference_pack=pack,
                approved=yes,
                dry_run=dry_run,
                out=out,
                on_estimate=lambda value: typer.echo(json.dumps(value, indent=2)),
            )
        )
    except BudgetExceeded as error:
        typer.echo(error.code, err=True)
        raise typer.Exit(4) from None
    except ExceptionGroup as error:
        budget_error = _budget_exception(error)
        typer.echo(
            budget_error.code
            if budget_error
            else "check_harness_failure; inspect sanitized captures",
            err=True,
        )
        raise typer.Exit(4 if budget_error else 5) from None
    except (ValueError, OSError):
        typer.echo(
            "Invalid check configuration, unsupported test selection or missing verified "
            "endpoint prices; see docs/usage.md.",
            err=True,
        )
        raise typer.Exit(3) from None
    except SafeError as error:
        typer.echo(error.code, err=True)
        raise typer.Exit(5) from None
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "exit_code": result.code,
                "report_dir": str(result.report_dir) if result.report_dir else None,
                "checks": [
                    {
                        "check_id": r.check_id,
                        "verdict": r.verdict,
                        "n": r.n,
                        "estimate": r.estimate,
                        "interval": [r.ci_low, r.ci_high],
                        "reference_tier": r.reference_tier,
                        "reference_estimate": r.reference_estimate,
                        "reason": r.reason,
                        "metric_units": r.details.get("units", "fraction or declared metric units"),
                        "geometric_mean_ratio": r.details.get("geometric_mean_ratio"),
                        "ratio_interval": r.details.get("ratio_interval"),
                        "reference_geometric_mean_ratio": r.details.get(
                            "reference_geometric_mean_ratio"
                        ),
                    }
                    for r in result.checks
                ],
            },
            indent=2,
        )
    )
    if result.report_dir is not None:
        score_run(Path.cwd(), result.run_id, out=result.report_dir, redact_content=redact_content)
    raise typer.Exit(result.code)


def _offline_workflow(
    run: str, out: Path | None, redact_content: bool, reference: str | None = None
) -> None:
    try:
        analysis, found = score_run(
            Path.cwd(), run, out=out, redact_content=redact_content, reference=reference
        )
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Scientific context or raw evidence unavailable/incompatible; scoring refused.",
            err=True,
        )
        raise typer.Exit(5) from None
    typer.echo(
        json.dumps(
            {
                "run_id": run,
                "checks": len(analysis.checks),
                "diagnoses": found,
                "out": str(out) if out else None,
            },
            indent=2,
        )
    )


@app.command("score")
def score_command(run: Annotated[str, typer.Option("--run")], reference: str | None = None) -> None:
    """Recompute scores from sealed raw inputs; optionally attach one qualified reference."""
    _offline_workflow(run, None, False, reference)


@app.command("diagnose")
def diagnose_command(
    run: Annotated[str, typer.Option("--run")], redact_content: bool = False
) -> None:
    """Recompute scoped diagnoses, local bundles and issue drafts; no external messages."""
    _offline_workflow(run, None, redact_content)


@app.command("report")
def report_command(
    run: Annotated[str, typer.Option("--run")],
    out: Path | None = None,
    redact_content: bool = False,
) -> None:
    """Render local HTML, Markdown and JSON from sealed raw observations."""
    destination = out or Path("reports") / (
        safe_filename(run) + "-report-v1" + ("-redacted" if redact_content else "")
    )
    methods = Path("docs/METHODS.md")
    try:
        analysis = reconstruct(Path("data/raw"), run)
        result = write_endpoint_report(
            analysis,
            destination,
            redact_content=redact_content,
            methods_path=methods if methods.is_file() else None,
        )
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Report refused: sealed inputs unavailable/incompatible or output already contains "
            "different files. Use a fresh --out directory.",
            err=True,
        )
        raise typer.Exit(5) from None
    typer.echo(json.dumps(result, indent=2))


@app.command("replay")
def replay_command(bundle: str, endpoint: str | None = None, yes: bool = False) -> None:
    """Repeat supporting checks under the original spending scope and compare outcomes."""
    try:
        value = asyncio.run(
            replay_bundle(
                bundle,
                root=Path.cwd(),
                endpoint=endpoint,
                approved=yes,
                on_estimate=lambda estimate: typer.echo(json.dumps(estimate, indent=2)),
            )
        )
    except BudgetExceeded as error:
        typer.echo(error.code, err=True)
        raise typer.Exit(4) from None
    except ExceptionGroup as error:
        budget_error = _budget_exception(error)
        typer.echo(budget_error.code if budget_error else "replay_harness_failure", err=True)
        raise typer.Exit(4 if budget_error else 5) from None
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Replay evidence or endpoint unavailable; no unreserved calls allowed.", err=True
        )
        raise typer.Exit(5) from None
    score_run(Path.cwd(), value["run_id"])
    typer.echo(json.dumps(value, indent=2))
    raise typer.Exit(0 if value["reproduced"] else 2)


mock_app = typer.Typer(help="Seeded synthetic and cassette replay HTTP service.")
app.add_typer(mock_app, name="mock")


@mock_app.command("serve")
def mock_serve(
    port: int = 8011,
    seed: int = 1,
    faults: Path | None = None,
    backend: str = "scripted",
    chunk_bytes: int = 0,
    proxy_plan: Path | None = None,
    yes: bool = False,
    context_length: int = 131072,
    max_completion_tokens: int = 32768,
    sampling_version: str = "v1",
    healthy_logit_jitter: float = 0.0,
) -> None:
    """Serve on loopback; remote proxy forwarding requires a reserved runner."""
    try:
        config = FaultConfig.model_validate(
            yaml.safe_load(faults.read_text(encoding="utf-8")) if faults else {}
        )
        forward = None
        if backend.startswith("proxy:"):
            if proxy_plan is None or not yes:
                typer.echo(
                    "Proxy requires --proxy-plan and --yes; no forwarding enabled.", err=True
                )
                raise typer.Exit(4)
            run_config = load_plan(proxy_plan)
            typer.echo(json.dumps(estimate_plan(run_config), indent=2))
            if len(run_config.endpoints) != 1 or backend[6:].rstrip("/") != run_config.endpoints[
                0
            ].base_url.rstrip("/"):
                raise ValueError("proxy plan endpoint mismatch")
            forward = reserved_forwarder(run_config)
        service = create_app(
            seed=seed,
            faults=config,
            backend=backend,
            chunk_bytes=chunk_bytes,
            reserved_forward=forward,
            context_length=context_length,
            max_completion_tokens=max_completion_tokens,
            sampling_version=sampling_version,
            healthy_logit_jitter=healthy_logit_jitter,
        )
    except (OSError, ValueError, ValidationError):
        typer.echo("Invalid mock configuration or cassette.", err=True)
        raise typer.Exit(3) from None
    uvicorn.run(service, host="127.0.0.1", port=port, access_log=False, log_level="warning")


refs_app = typer.Typer(help="Build matched-item reference packs; publication is owner-only.")
app.add_typer(refs_app, name="refs")


def _reference_environment() -> dict[str, str]:
    return dict(os.environ)


@refs_app.command("template")
def refs_template(
    profile: Annotated[str, typer.Option("--profile")],
    tier: Annotated[str, typer.Option("--tier")],
    endpoint: Annotated[Path, typer.Option("--endpoint")],
    maker: str | None = None,
) -> None:
    """Print an incomplete R1/R2 review form; no capture, key or owner activation."""
    from servegauge.core.models import EndpointSpec
    from servegauge.refs.owner import qualification_template
    from servegauge.storage.owner_reference import read_review

    try:
        declaration = EndpointSpec.model_validate(read_review(Path.cwd(), endpoint))
        result = qualification_template(tier, load_profile(profile), declaration, maker=maker)
    except (OSError, ValueError, LookupError, TypeError):
        typer.echo(
            "Reference template refused: owned endpoint JSON/profile "
            "and known tier/creator required.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(json.dumps(result, indent=2))


@refs_app.command("prepare")
def refs_prepare(
    profile: Annotated[str, typer.Option("--profile")],
    endpoint: Annotated[Path, typer.Option("--endpoint")],
    qualification: Annotated[Path, typer.Option("--qualification")],
) -> None:
    """Validate local origin review and print the endpoint to put in a future plan."""
    from servegauge.core.hashing import content_hash
    from servegauge.core.models import EndpointSpec
    from servegauge.refs.owner import Qualification, bind_endpoint
    from servegauge.storage.owner_reference import read_review

    try:
        root = Path.cwd()
        request = Qualification.model_validate(read_review(root, qualification))
        declaration = EndpointSpec.model_validate(read_review(root, endpoint))
        bound = bind_endpoint(request, load_profile(profile), declaration)
    except (OSError, ValueError, LookupError, TypeError):
        typer.echo(
            "Reference preparation refused: complete reviewed origin/profile/endpoint required.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "status": "prepared_only",
                "qualification_hash": content_hash(request.model_dump(mode="json")),
                "endpoint": bound.model_dump(mode="json"),
                "model_api_calls": 0,
                "reference_tier_granted": False,
            },
            indent=2,
        )
    )


@refs_app.command("build")
def refs_build(
    profile: Annotated[str, typer.Option("--profile")],
    tier: Annotated[str, typer.Option("--tier")],
    from_run: Annotated[str, typer.Option("--from-run")],
    seed: int = 20261018,
    control_run: Annotated[list[str] | None, typer.Option("--control-run")] = None,
    reference_run: Annotated[list[str] | None, typer.Option("--reference-run")] = None,
    baseline_endpoint: Annotated[str | None, typer.Option("--baseline-endpoint")] = None,
    fingerprint_qualification: Annotated[
        str | None, typer.Option("--fingerprint-qualification")
    ] = None,
    owner_qualification: Annotated[Path | None, typer.Option("--owner-qualification")] = None,
    confirm: bool = False,
) -> None:
    """Build from sealed captures; R1/R2 require pre-capture review and owner confirmation."""
    try:
        loaded_profile = load_profile(profile)
        if tier not in {"MOCK", "BASELINE", "R3", "R1", "R2"}:
            raise ValueError("tier_requires_owner_qualification")
        if tier in {"R1", "R2"}:
            from servegauge.refs.owner import Qualification, build_owner_pack
            from servegauge.storage.owner_reference import captured_reference, read_review
            from servegauge.storage.reference_export import _guard_capture

            environment = _reference_environment()
            if (
                not confirm
                or owner_qualification is None
                or environment.get("SERVEGAUGE_OWNER_RUN", environment.get("TRUEWEIGHTS_OWNER_RUN"))
                != "1"
                or baseline_endpoint is not None
                or fingerprint_qualification is not None
            ):
                raise ValueError("reference_explicit_owner_qualification_required")
            root = Path.cwd()
            request = Qualification.model_validate(read_review(root, owner_qualification))
            if request.tier != tier:
                raise ValueError("reference_reviewed_tier_changed")
            capture = captured_reference(root, from_run, loaded_profile, request)
            owner_controls = []
            for name in control_run or []:
                _guard_capture(root / "data/raw", name)
                owner_controls.append(
                    ReferenceRun.model_validate(
                        export_outcomes(reconstruct(root / "data/raw", name))
                    )
                )
            pack = build_owner_pack(
                capture.source,
                loaded_profile,
                request,
                capture.records,
                seed=seed,
                environment=environment,
                session=capture.session,
                controls=owner_controls if control_run is not None else None,
                references=[
                    captured_reference(root, name, loaded_profile, request)
                    for name in reference_run or []
                ],
            )
            destination = save_pack(pack)
            typer.echo(
                json.dumps(
                    {
                        "pack_id": pack.pack_id,
                        "tier": tier,
                        "path": str(destination),
                        "n_per_check": pack.manifest["n_per_check"],
                    },
                    indent=2,
                )
            )
            return
        if owner_qualification is not None or confirm:
            raise ValueError("reference_owner_options_require_R1_or_R2")
        source = ReferenceRun.model_validate(
            export_outcomes(reconstruct(Path("data/raw"), from_run))
        )
        controls = [
            ReferenceRun.model_validate(export_outcomes(reconstruct(Path("data/raw"), name)))
            for name in control_run or []
        ]
        references = [
            ReferenceRun.model_validate(export_outcomes(reconstruct(Path("data/raw"), name)))
            for name in reference_run or []
        ]
        if source.profile != profile or source.run_id != from_run:
            raise ValueError("reference_source_mismatch")
        if baseline_endpoint is not None:
            if tier != "BASELINE" or baseline_endpoint not in {e.id for e in source.endpoints}:
                raise ValueError("trusted_baseline_endpoint_required")
            source.baseline_endpoint_id = baseline_endpoint
        # The branch validates the Literal before invoking the pure builder.
        pack = build_pack(
            source,
            "MOCK" if tier == "MOCK" else "BASELINE" if tier == "BASELINE" else "R3",
            seed=seed,
            controls=controls if control_run is not None else None,
            reference_runs=references,
            fingerprint_qualification=fingerprint_qualification,
        )
        destination = save_pack(pack)
    except (OSError, ValueError, LookupError, TypeError, ArithmeticError):
        typer.echo(
            "Reference pack refused: sealed raw evidence, matching profile/items/config "
            "and qualified peers/controls are required.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "pack_id": pack.pack_id,
                "tier": tier,
                "path": str(destination),
                "n_per_check": pack.manifest["n_per_check"],
            },
            indent=2,
        )
    )


@refs_app.command("publish")
def refs_publish(
    pack_id: str,
    out: Path | None = None,
    config: Path = Path("configs/observatory.yaml"),
    dry_run: bool = False,
    confirm: bool = False,
) -> None:
    """Stage a reusable reference offline; owner confirmation rechecks exact bytes."""
    from servegauge.observatory.schedule import load_observatory
    from servegauge.storage.notices import guard_path
    from servegauge.storage.reference_export import reference_preflight, stage_reference

    destination = out or Path("reports/reference-publication") / safe_filename(pack_id)
    try:
        if dry_run and confirm:
            raise ValueError("choose_dry_run_or_confirm")
        if not confirm:
            value = stage_reference(Path.cwd(), pack_id, destination)
        else:
            if not any(
                os.environ.get(name) == "1"
                for name in ["SERVEGAUGE_ALLOW_PUBLISH", "TRUEWEIGHTS_ALLOW_PUBLISH"]
            ):
                raise ValueError("owner_publish_variable_required")
            guard_path(config)
            settings = load_observatory(config)
            target = settings.publishing
            if not target.enabled or settings.mode != "owner" or not target.repo_id:
                raise ValueError("enabled_owner_destination_required")
            credential = os.environ.get(target.credential_env)
            if not credential:
                raise ValueError("owner_destination_credential_required")
            payloads = reference_preflight(Path.cwd(), pack_id, destination)
            from servegauge.adapters.hub import upload_dataset

            commit = upload_dataset(
                target.repo_id,
                credential,
                payloads,
                commit_message="Add reviewed ServeGauge reusable reference",
            )
            value = {
                "published": True,
                "upload_commits": 1,
                "commit_oid": commit,
                "repo_id": target.repo_id,
                "source_pack_id": pack_id,
                "model_api_calls": 0,
            }
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Reference publication refused: check saved pack, source rights/notices, "
            "reviewed bytes and owner destination settings.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(json.dumps(value, indent=2))


validate_app = typer.Typer(help="Socket-free controlled validation; never sends model requests.")


@validate_app.command("operational-health")
def validate_operational_health(
    out: Annotated[Path, typer.Option()] = Path("docs/report/operational-health-v1"),
) -> None:
    """Validate default core procedure alarms on clean and original F-ERR controls."""
    from servegauge.validation.operational_health import health_validation

    try:
        result = health_validation(out)
    except (ValueError, OSError, LookupError, TypeError, RuntimeError) as error:
        typer.echo(
            "Operational health study failed before acceptance ("
            + type(error).__name__
            + "); check configuration and use a fresh output directory.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "output": str(out / "health.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("fingerprint-product-intervals")
def validate_fingerprint_product_intervals(
    out: Annotated[Path, typer.Option()] = Path("docs/report/fingerprint-product-v1"),
) -> None:
    """Calibrate/evaluate text intervals on disjoint known-truth categorical grids."""
    from servegauge.validation.fingerprint_product import interval_study

    try:
        result = interval_study(out)
    except (ValueError, OSError, LookupError, TypeError):
        typer.echo(
            "Fingerprint interval study refused: use intact evidence or a new output directory.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "output": str(out / "intervals.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
                "scope": result["scope"],
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("fingerprint-product-null")
def validate_fingerprint_product_null(
    out: Annotated[Path, typer.Option()] = Path("docs/report/fingerprint-product-null-v1"),
) -> None:
    """Measure candidate-size null/family error on declared iid components."""
    from servegauge.validation.fingerprint_product_null import null_study

    try:
        result = null_study(out)
    except (ValueError, OSError, LookupError, TypeError):
        typer.echo(
            "Fingerprint null study refused: intact evidence or a new directory required.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "scope": result["scope"],
                "output": str(out / "null.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("fingerprint-wire-power")
def validate_fingerprint_wire_power(
    out: Annotated[Path, typer.Option()] = Path("docs/report/fingerprint-wire-v1"),
) -> None:
    """Measure held-out scripted capture power for three independent reference packs."""
    from servegauge.validation.fingerprint_wire import wire_study

    try:
        result = wire_study(out)
    except (ValueError, OSError, LookupError, TypeError):
        typer.echo(
            "Fingerprint wire study refused: intact prerequisites and source required.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "scope": result["scope"],
                "output": str(out / "wire.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("faults")
def validate_faults(
    replicates: Annotated[int | None, typer.Option(min=30)] = None,
    out: Annotated[Path, typer.Option()] = Path("docs/report"),
) -> None:
    from servegauge.validation.fault_matrix import fault_matrix

    result = fault_matrix(replicates=replicates, out=out)
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "output": str(out / "faults.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("power")
def validate_power(
    replicates: Annotated[int | None, typer.Option(min=30)] = None,
    out: Annotated[Path, typer.Option()] = Path("docs/report"),
) -> None:
    from servegauge.validation.power_study import power_study

    result = power_study(replicates=replicates, out=out)
    typer.echo(
        json.dumps(
            {
                "complete": result["complete"],
                "rows": len(result["rows"]),
                "model_api_calls": 0,
                "output": str(out / "power.md"),
            }
        )
    )


@validate_app.command("monitor-transport")
def validate_monitor_transport(
    out: Annotated[Path, typer.Option()],
    replicates: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Construct full-family/readiness evidence offline; does not arm a monitor."""
    from servegauge.validation.transport_study import transport_study

    try:
        result = transport_study(out, replicates=replicates)
    except (ValueError, OSError, KeyError, TypeError):
        typer.echo(
            "Transport study refused: use a fresh output and the declared local plan.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "eligible_for_operating_review": result["eligible_for_operating_review"],
                "operating_qualification_accepted": False,
                "model_api_calls": 0,
                "new_spend_usd": 0,
                "output": str(out / "study.json"),
            }
        )
    )
    if not result["eligible_for_operating_review"]:
        raise typer.Exit(2)


@validate_app.command("changepoint")
def validate_changepoint(
    replicates: Annotated[int | None, typer.Option(min=60)] = None,
    out: Annotated[Path, typer.Option()] = Path("docs/report"),
) -> None:
    from servegauge.validation.changepoint_sim import changepoint_study

    result = changepoint_study(replicates=replicates, out=out)
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "model_api_calls": 0,
                "output": str(out / "changepoint.md"),
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("noise-v2")
def validate_noise_v2(
    out: Annotated[Path, typer.Option()] = Path("docs/report/detection-v2"),
) -> None:
    """Evaluate truthful top-k noise sensitivity; preserves original power study."""
    from servegauge.validation.detection_study import noise_study

    result = noise_study(out=out)
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "output": str(out / "noise.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


@validate_app.command("changepoint-v2")
def validate_changepoint_v2(
    out: Annotated[Path, typer.Option()] = Path("docs/report/detection-v2"),
) -> None:
    """Calibrate replacement methods, then compare on fresh matched trajectories."""
    from servegauge.validation.detection_study import change_study

    result = change_study(out=out)
    typer.echo(
        json.dumps(
            {
                "accepted": result["accepted"],
                "output": str(out / "changepoint.json"),
                "model_api_calls": 0,
                "spent_usd": 0,
            }
        )
    )
    if not result["accepted"]:
        raise typer.Exit(1)


app.add_typer(validate_app, name="validate")

fingerprint_app = typer.Typer(
    help="Offline fingerprint evidence diagnostics; never sends requests."
)


@fingerprint_app.command("analyze")
def analyze_fingerprint(
    file: Path,
    out: Annotated[Path | None, typer.Option()] = None,
    calibration: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Analyze paired session evidence; incomplete evidence exits INCONCLUSIVE (2)."""
    from servegauge.scoring.fingerprint_analysis import analyze_evidence
    from servegauge.scoring.fingerprint_calibration import SignalCalibration
    from servegauge.scoring.fingerprint_evidence import FingerprintEvidence
    from servegauge.validation.common import ROOT, offline_only, source_hash, write_json
    from servegauge.verdicts.config import load_policy

    try:
        with offline_only():
            raw = file.read_bytes()
            evidence = FingerprintEvidence.model_validate_json(raw)
            cfg = yaml.safe_load(
                (ROOT / "configs/robustness_validation.yaml").read_text(encoding="utf-8")
            )
            result = analyze_evidence(
                evidence,
                policy=load_policy(),
                minimum_groups=int(cfg["minimum_independent_groups"]),
                minimum_anchor_tokens=int(cfg["minimum_anchor_tokens"]),
                signal_calibration=SignalCalibration.model_validate_json(calibration.read_bytes())
                if calibration is not None
                else None,
                current_source_hash=source_hash() if calibration is not None else None,
            )
            if out is not None:
                if out.resolve() == file.resolve() or (out.exists() and out.samefile(file)):
                    raise ValueError("analysis_output_cannot_replace_input")
                if calibration is not None and (
                    out.resolve() == calibration.resolve()
                    or (out.exists() and out.samefile(calibration))
                ):
                    raise ValueError("analysis_output_cannot_replace_calibration")
                write_json(out, result)
    except (OSError, ValueError, TypeError, KeyError):
        typer.echo('{"error":"invalid_or_unavailable_fingerprint_evidence"}')
        raise typer.Exit(3) from None
    typer.echo(json.dumps(result))
    if result["status"] == "INCONCLUSIVE":
        raise typer.Exit(2)


app.add_typer(fingerprint_app, name="fingerprint")


@validate_app.command("robustness")
def validate_robustness(
    out: Annotated[Path, typer.Option()] = Path("docs/report/robustness-v3-1"),
) -> None:
    """Evaluate session dependence, response-cache bursts and incomplete logprobs offline."""
    from servegauge.validation.robustness_study import robustness_study

    result = robustness_study(out=out)
    typer.echo(json.dumps({"accepted": result["accepted"], "model_api_calls": 0, "spent_usd": 0}))
    if not result["accepted"]:
        raise typer.Exit(1)


site_app = typer.Typer(help="Build content-removed local sites; preview only on loopback.")


@site_app.command("build")
def site_build_command(
    run: Annotated[list[str] | None, typer.Option("--run")] = None,
    out: Path = Path("reports/site"),
) -> None:
    """Select sealed run IDs explicitly; no selection builds an honest empty index."""
    from servegauge.report.site.builder import build_site

    try:
        result = build_site(Path.cwd(), out, runs=run)
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Site build refused: inputs unavailable or destination differs; use fresh --out.",
            err=True,
        )
        raise typer.Exit(5) from None
    typer.echo(json.dumps(result, indent=2))


@site_app.command("serve")
def site_serve_command(
    out: Path = Path("reports/site"),
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8040,
    dry_run: bool = False,
) -> None:
    """Preview only verified files at 127.0.0.1; --dry-run validates without sockets."""
    from servegauge.report.site.builder import verify_site
    from servegauge.report.site.server import serve_site

    try:
        manifest = verify_site(out)
        if dry_run:
            typer.echo(
                json.dumps(
                    {
                        "url": f"http://127.0.0.1:{port}/",
                        "files": len(manifest["files"]),
                        "dry_run": True,
                    }
                )
            )
            return
        serve_site(
            out,
            port,
            on_ready=lambda chosen: typer.echo(
                f"Local preview: http://127.0.0.1:{chosen}/ (Ctrl-C to stop)"
            ),
        )
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, KeyError, TypeError):
        typer.echo("Site preview refused: artifact invalid or local port unavailable.", err=True)
        raise typer.Exit(5) from None


@site_app.command("verify")
def site_verify_command(
    out: Path = Path("reports/site"),
    for_publication: bool = False,
    confirm: bool = False,
) -> None:
    """Inspect the sealed inventory; publication preflight never performs a network write."""
    from servegauge.report.site.builder import publication_preflight, verify_site

    if for_publication and (
        not confirm
        or not any(
            os.environ.get(name) == "1"
            for name in ["SERVEGAUGE_ALLOW_PUBLISH", "TRUEWEIGHTS_ALLOW_PUBLISH"]
        )
    ):
        typer.echo(
            "Publication preflight requires --confirm and the owner allow-publish variable.",
            err=True,
        )
        raise typer.Exit(4)
    try:
        value = publication_preflight(out) if for_publication else verify_site(out)
    except (OSError, ValueError, KeyError, TypeError):
        typer.echo(
            "Site verification refused: invalid inventory or pending provider notice/reply "
            "workflow.",
            err=True,
        )
        raise typer.Exit(5) from None
    typer.echo(json.dumps(value, indent=2))


app.add_typer(site_app, name="site")


monitor_app = typer.Typer(help="Reserved UTC monitoring windows; disabled unless configured.")


@monitor_app.command("tick")
def monitor_tick_command(
    cadence: Annotated[str, typer.Option("--cadence")],
    config: Annotated[Path, typer.Option("--config")],
    root: Path = Path("."),
    resume: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    evaluate: bool = False,
    proof_root: Path = Path("."),
) -> None:
    """Execute or inspect an explicit cadence; never publish or contact providers."""
    from servegauge.observatory.schedule import load_observatory
    from servegauge.observatory.tick import tick

    try:
        selected = load_observatory(config)
        result = asyncio.run(
            tick(
                selected,
                cadence,
                root=root,
                resume=resume,
                dry_run=dry_run,
                approved=yes,
            )
        )
    except (ValueError, OSError, SafeError, KeyError, TypeError) as error:
        typer.echo(
            "Monitor refused: check configuration, owner gates or preserved window context; "
            "use --resume only for an unfinished window.",
            err=True,
        )
        raise typer.Exit(4 if isinstance(error, BudgetExceeded) else 3) from None
    if result["status"] == "dry_run":
        result = {
            **result,
            "jobs": [
                {"run_id": j["run_id"], "model_id": j["model_id"], "estimate": j["estimate"]}
                for j in result["jobs"]
            ],
        }
    if evaluate and result["status"] in {"completed", "already_completed"}:
        from filelock import Timeout

        from servegauge.observatory.evaluation import evaluate_history

        try:
            result["monitoring"] = evaluate_history(root, selected.id, proof_root=proof_root)
        except (ValueError, OSError, KeyError, TypeError, Timeout):
            typer.echo(
                "Monitor evaluation refused: inspect preserved history and qualification.", err=True
            )
            raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(result, indent=2)))
    raise typer.Exit(
        result.get("measurement_exit_code", 0) if result["status"] == "completed" else 0
    )


@monitor_app.command("inspect")
def monitor_inspect_command(
    observatory_id: Annotated[str, typer.Option("--observatory-id")],
    root: Path = Path("."),
    read_only: bool = False,
) -> None:
    """Replay sealed transport candidates locally; never collect or arm an alert."""
    from filelock import Timeout

    from servegauge.observatory.history import inspect_history

    try:
        result = inspect_history(root, observatory_id, persist=not read_only)
    except (ValueError, OSError, KeyError, TypeError, Timeout):
        typer.echo(
            "Monitor inspection refused: missing, conflicting or changed sealed history.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(result, indent=2)))


@monitor_app.command("evaluate")
def monitor_evaluate_command(
    observatory_id: Annotated[str, typer.Option("--observatory-id")],
    root: Path = Path("."),
    proof_root: Path = Path("."),
    read_only: bool = False,
) -> None:
    """Evaluate qualified sealed local evidence; no new inference or external notice."""
    from filelock import Timeout

    from servegauge.observatory.evaluation import evaluate_history

    try:
        result = evaluate_history(
            root, observatory_id, proof_root=proof_root, persist=not read_only
        )
    except (ValueError, OSError, KeyError, TypeError, Timeout):
        typer.echo(
            "Monitor evaluation refused: inspect preserved history and qualification.", err=True
        )
        raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(result, indent=2)))


app.add_typer(monitor_app, name="monitor")

notice_app = typer.Typer(
    help="Local owner assertions only; no message is sent or delivery verified."
)


@notice_app.command("prepare")
def notice_prepare_command(
    run: Annotated[str, typer.Option("--run")],
    endpoint: Annotated[str, typer.Option("--endpoint")],
    check: Annotated[str, typer.Option("--check")],
    bundle: Annotated[str, typer.Option("--bundle")],
    root: Path = Path("."),
) -> None:
    from filelock import Timeout

    from servegauge.observatory.notice_workflow import prepare_notice

    try:
        value = prepare_notice(root, run, endpoint, check, bundle)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, Timeout):
        typer.echo(
            "Notice preparation refused: sealed FAIL and matching full bundle required.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(value, indent=2)))


@notice_app.command("record")
def notice_record_command(
    notice_id: Annotated[str, typer.Option("--notice-id")],
    kind: Annotated[str, typer.Option("--kind")],
    at: Annotated[str, typer.Option("--at")],
    reference_sha256: Annotated[str, typer.Option("--reference-sha256")],
    root: Path = Path("."),
    owner_assertion: bool = False,
    channel: str | None = None,
) -> None:
    from datetime import datetime

    from filelock import Timeout

    from servegauge.storage.notices import NoticeJournal

    try:
        journal = NoticeJournal(root / "data/raw", notice_id)
        recorded = journal.record(
            kind,
            at=datetime.fromisoformat(at),
            reference_sha256=reference_sha256,
            owner_assertion=owner_assertion,
            channel=channel,
        )
    except (OSError, ValueError, KeyError, TypeError, Timeout):
        typer.echo(
            "Notice assertion refused: check owner assertion, UTC time, identity and hash.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(
        json.dumps(
            {
                "status": "recorded" if recorded else "unchanged",
                "delivery_verified": False,
                "network_writes": 0,
            },
            indent=2,
        )
    )


@notice_app.command("status")
def notice_status_command(
    run: Annotated[str, typer.Option("--run")],
    endpoint: Annotated[str, typer.Option("--endpoint")],
    check: Annotated[str, typer.Option("--check")],
    root: Path = Path("."),
) -> None:
    from servegauge.observatory.notice_workflow import publication_status
    from servegauge.observatory.notices import failure_subject

    try:
        subject = failure_subject(reconstruct(root / "data/raw", run), endpoint, check)
        value = publication_status(root, subject)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        typer.echo("Notice status unavailable: current sealed failure evidence required.", err=True)
        raise typer.Exit(3) from None
    typer.echo(Redactor.from_environment().text(json.dumps(value, indent=2)))


app.add_typer(notice_app, name="notice")


publish_app = typer.Typer(help="Stage explicit numeric exports; default is offline dry-run.")


@publish_app.command("hf")
def publish_hf_command(
    run: Annotated[list[str] | None, typer.Option("--run")] = None,
    out: Path = Path("reports/hf-export"),
    config: Path = Path("configs/observatory.yaml"),
    dry_run: bool = False,
    confirm: bool = False,
) -> None:
    """Prepare a dataset/card without network; --confirm uses the reviewed local selection."""
    from servegauge.observatory.schedule import load_observatory
    from servegauge.storage.export import publication_preflight, stage_export
    from servegauge.storage.notices import guard_path

    try:
        if dry_run and confirm:
            raise ValueError("choose_dry_run_or_confirm")
        if not confirm:
            value = stage_export(Path.cwd(), out, runs=run)
        else:
            if not any(
                os.environ.get(name) == "1"
                for name in ["SERVEGAUGE_ALLOW_PUBLISH", "TRUEWEIGHTS_ALLOW_PUBLISH"]
            ):
                raise ValueError("owner_publish_variable_required")
            guard_path(config)
            settings = load_observatory(config)
            destination = settings.publishing
            if not destination.enabled or settings.mode != "owner" or not destination.repo_id:
                raise ValueError("enabled_owner_destination_required")
            credential = os.environ.get(destination.credential_env)
            if not credential:
                raise ValueError("owner_destination_credential_required")
            if run is not None:
                stage_export(Path.cwd(), out, runs=run)
            payloads = publication_preflight(Path.cwd(), out)
            from servegauge.adapters.hub import upload_dataset

            commit = upload_dataset(destination.repo_id, credential, payloads)
            value = {
                "published": True,
                "upload_commits": 1,
                "commit_oid": commit,
                "repo_id": destination.repo_id,
                "model_api_calls": 0,
                "network_writes": "SDK-managed; physical write requests not counted",
            }
    except (OSError, ValueError, KeyError, TypeError, SafeError):
        typer.echo(
            "Dataset operation refused or upload outcome unverified; "
            "inspect local inputs and owner destination gates.",
            err=True,
        )
        raise typer.Exit(5) from None
    typer.echo(json.dumps(value, indent=2))


app.add_typer(publish_app, name="publish")


selfhost_app = typer.Typer(help="Local self-host preparation and explicit owner session controls.")


def _selfhost_refused(code: int = 3) -> NoReturn:
    typer.echo(
        "Self-host operation refused: check reviewed configuration, caps, session identity, "
        "owner authorization and credentials. No automatic launch retry is performed.",
        err=True,
    )
    raise typer.Exit(code)


def _selfhost_environment() -> dict[str, str]:
    # One snapshot binds authorization and adapter credentials to the same inputs.
    return dict(os.environ)


def _selfhost_emit(value: dict[str, Any], *, launch: bool = False) -> None:
    typer.echo(json.dumps(value, indent=2))
    if value.get("state") == "outcome_unverified":
        raise typer.Exit(5)
    if launch and value.get("state") not in {"prepared", "running"}:
        raise typer.Exit(5)


@selfhost_app.command("template")
def selfhost_template(
    backend: str = "ssh",
    profile: str = "gpt-oss-120b",
    out: Path = Path("configs/selfhost/owner.json"),
    variant: str | None = None,
) -> None:
    """Write an incomplete owner template with no guessed image, revision or price."""
    from servegauge.selfhost.plan import config_template
    from servegauge.storage.selfhost import write_template

    try:
        write_template(Path.cwd(), out, config_template(backend, profile, variant=variant))
    except (OSError, ValueError, TypeError, KeyError):
        _selfhost_refused()
    typer.echo(
        json.dumps(
            {
                "incomplete_owner_template": True,
                "backend": backend,
                "model_api_calls": 0,
                "network_writes": 0,
            }
        )
    )


@selfhost_app.command("study-prepare")
def selfhost_study_prepare(
    design: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    backend: str | None = None,
) -> None:
    """Prepare a private, unfunded matrix; never launches an engine or collects data."""
    from servegauge.core.hashing import content_hash
    from servegauge.selfhost.study_design import StudyDesign
    from servegauge.storage.gpu_study import _paths, prepare_study

    try:
        _paths(design, out)
        from servegauge.selfhost.study_protocol import CollectionSpec
        from servegauge.storage.comparison import _private_json
        from servegauge.storage.gpu_collection import prepare_collection_study

        raw = _private_json(design.read_bytes())
        if isinstance(raw, dict) and raw.get("version") == "gpu-study-collection-v1":
            full = CollectionSpec.model_validate(raw)
            if backend is not None and backend != full.backend:
                raise ValueError("study_collection_backend_changed")
            value = prepare_collection_study(Path.cwd(), full, out=out)
        else:
            settings = StudyDesign.model_validate(raw)
            value = prepare_study(Path.cwd(), settings, out=out, backend=backend or "ssh")
    except (OSError, ValueError, TypeError, KeyError, SafeError):
        _selfhost_refused()
    typer.echo(json.dumps({"kit_hash": content_hash(value), "manifest": value}, indent=2))


@selfhost_app.command("study-verify")
def selfhost_study_verify(
    kit: Annotated[Path, typer.Option()],
    kit_hash: Annotated[str, typer.Option()],
) -> None:
    """Verify prepared matrix bytes against its independently retained hash."""
    from servegauge.storage.gpu_study import verify_study

    try:
        value = verify_study(kit, kit_hash=kit_hash)
    except (OSError, ValueError, TypeError, KeyError):
        _selfhost_refused()
    typer.echo(json.dumps({"verified": True, "manifest": value}, indent=2))


@selfhost_app.command("study-template")
def selfhost_study_template(
    out: Path = Path("configs/selfhost/study-collection.json"),
    backend: str = "ssh",
    scope: str = "owner_gpu",
) -> None:
    """Write the complete study design with unreviewed sampling assumptions."""
    from servegauge.selfhost.study_protocol import collection_template
    from servegauge.storage.selfhost import write_template

    try:
        write_template(Path.cwd(), out, collection_template(backend, scope))
    except (OSError, ValueError, TypeError, KeyError):
        _selfhost_refused()
    typer.echo(
        json.dumps(
            {"state": "incomplete_owner_design", "model_api_calls": 0, "gpu_sessions_started": 0}
        )
    )


@selfhost_app.command("study-slot")
def selfhost_study_slot(
    kit: Annotated[Path, typer.Option()],
    kit_hash: Annotated[str, typer.Option()],
    slot: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    """Copy a slot's incomplete engine recipe and emit its exclusive session/run IDs."""
    from servegauge.selfhost.study_protocol import collection_owner_template
    from servegauge.storage.gpu_capture import study_context
    from servegauge.storage.selfhost import write_template

    try:
        context = study_context(Path.cwd(), kit, kit_hash)
        selected = context.slot(slot)
        write_template(Path.cwd(), out, collection_owner_template(context.spec, selected["recipe"]))
    except (OSError, ValueError, TypeError, KeyError, SafeError):
        _selfhost_refused()
    typer.echo(
        json.dumps(
            {
                "slot": selected,
                "incomplete_owner_recipe": True,
                "model_api_calls": 0,
                "gpu_sessions_started": 0,
            },
            indent=2,
        )
    )


@selfhost_app.command("study-collect")
def selfhost_study_collect(
    kit: Annotated[Path, typer.Option()],
    kit_hash: Annotated[str, typer.Option()],
    slot: Annotated[str, typer.Option()],
    yes: Annotated[bool, typer.Option("--yes")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    resume: bool = False,
) -> None:
    """Collect one bound slot through normal reservations; never launches a GPU."""
    from servegauge.storage.gpu_capture import collect_slot

    try:
        value = asyncio.run(
            collect_slot(
                Path.cwd(),
                kit,
                kit_hash,
                slot,
                approved=yes,
                dry_run=dry_run,
                resume=resume,
                environment=_selfhost_environment(),
            )
        )
    except BudgetExceeded:
        _selfhost_refused(4)
    except (OSError, ValueError, TypeError, KeyError, SafeError):
        _selfhost_refused()
    typer.echo(json.dumps(value, indent=2))
    if value["state"] == "captured_with_missing_evidence":
        raise typer.Exit(2)


@selfhost_app.command("study-status")
def selfhost_study_status(
    kit: Annotated[Path, typer.Option()],
    kit_hash: Annotated[str, typer.Option()],
) -> None:
    """Read current bound slots and pending intents; never probes a remote endpoint."""
    from servegauge.storage.gpu_capture import study_status

    try:
        value = study_status(Path.cwd(), kit, kit_hash)
    except (OSError, ValueError, TypeError, KeyError, SafeError):
        _selfhost_refused()
    typer.echo(json.dumps(value, indent=2))


@selfhost_app.command("study-analyze")
def selfhost_study_analyze(
    kit: Annotated[Path, typer.Option()],
    kit_hash: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    """Reconstruct selected raw quartets and write a private descriptive ROC report."""
    from servegauge.storage.gpu_results import analyze_study, write_study_report

    try:
        value = analyze_study(Path.cwd(), kit, kit_hash)
        report = write_study_report(Path.cwd(), value, out=out)
    except (OSError, ValueError, TypeError, KeyError, SafeError):
        _selfhost_refused()
    typer.echo(
        json.dumps(
            {
                "state": "report_written",
                "scope": value["scope"],
                "status": value["status"],
                "capture_summary": value["capture_summary"],
                "report": report,
                "model_api_calls": 0,
                "gpu_sessions_started": 0,
                "physical_precision_verified": False,
            },
            indent=2,
        )
    )


@selfhost_app.command("up")
def selfhost_up(
    profile: str | None = None,
    config: Path = Path("configs/selfhost/owner.json"),
    max_usd: str | None = None,
    shutdown_minutes: int | None = None,
    session_id: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    backend: str | None = None,
) -> None:
    """Stage a bounded session offline; --yes requires the previously reviewed session ID."""
    from uuid import uuid4

    from servegauge.adapters.selfhosted import driver_for
    from servegauge.selfhost.controller import _authorize, launch_session
    from servegauge.selfhost.plan import build_plan, parse_session_config
    from servegauge.storage.notices import guard_path
    from servegauge.storage.selfhost import SessionJournal, prepare_session

    try:
        if dry_run == yes or profile is None or max_usd is None or shutdown_minutes is None:
            raise ValueError("selfhost_explicit_mode_profile_and_caps_required")
        if (
            ".." in config.parts
            or config.suffix != ".json"
            or any(p.casefold() == ".git" or p.casefold().startswith(".env") for p in config.parts)
        ):
            raise ValueError("selfhost_reviewed_json_configuration_required")
        guard_path(config)
        settings = parse_session_config(json.loads(config.read_text()))
        if settings.profile != profile or (backend is not None and settings.backend != backend):
            raise ValueError("selfhost_reviewed_profile_changed")
        root = Path.cwd()
        guard_path(root / "configs/budgets.yaml")
        policy = load_budget(root)
        plan = build_plan(
            settings, policy, max_usd=Decimal(max_usd), shutdown_minutes=shutdown_minutes
        )
        if dry_run:
            identifier = session_id or "selfhost-" + uuid4().hex
            value = prepare_session(root, identifier, plan).status()
        else:
            if session_id is None:
                raise ValueError("selfhost_previously_reviewed_session_required")
            journal = SessionJournal(root, session_id)
            if journal.status()["plan"] != plan:
                raise ValueError("selfhost_reviewed_session_inputs_changed")
            environment = _selfhost_environment()
            _authorize(plan, environment, launch=True)
            value = launch_session(
                root, session_id, policy, driver=driver_for(root, plan), environment=environment
            )
    except BudgetExceeded:
        _selfhost_refused(4)
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, SafeError, TimeoutError):
        _selfhost_refused()
    _selfhost_emit(value, launch=True)


@selfhost_app.command("status")
def selfhost_status(
    session_id: str | None = None,
    profile: str | None = None,
    refresh: bool = False,
    backend: str | None = None,
) -> None:
    """Read local state; --refresh additionally requires owner-authorized remote inspection."""
    from servegauge.adapters.selfhosted import driver_for
    from servegauge.selfhost.controller import _authorize, refresh_session
    from servegauge.storage.selfhost import SessionJournal

    try:
        if session_id is None:
            raise ValueError("selfhost_session_id_required")
        root = Path.cwd()
        value = SessionJournal(root, session_id).status()
        if backend is not None and value["plan"]["configuration"]["backend"] != backend:
            raise ValueError("selfhost_reviewed_backend_changed")
        if profile is not None and value["plan"]["profile"] != profile:
            raise ValueError("selfhost_reviewed_profile_changed")
        if refresh:
            environment = _selfhost_environment()
            _authorize(value["plan"], environment, launch=False, remote=False)
            value = refresh_session(
                root, session_id, driver=driver_for(root, value["plan"]), environment=environment
            )
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, SafeError, TimeoutError):
        _selfhost_refused()
    _selfhost_emit(value)


@selfhost_app.command("down")
def selfhost_down(
    session_id: str | None = None,
    profile: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    backend: str | None = None,
) -> None:
    """Preview locally or stop only the saved owner resource; cleanup never reserves again."""
    from servegauge.adapters.selfhosted import driver_for
    from servegauge.selfhost.controller import _authorize, stop_session
    from servegauge.storage.selfhost import SessionJournal

    try:
        if session_id is None or dry_run == yes:
            raise ValueError("selfhost_explicit_cleanup_mode_and_session_required")
        root = Path.cwd()
        value = SessionJournal(root, session_id).status()
        if backend is not None and value["plan"]["configuration"]["backend"] != backend:
            raise ValueError("selfhost_reviewed_backend_changed")
        if profile is not None and value["plan"]["profile"] != profile:
            raise ValueError("selfhost_reviewed_profile_changed")
        if dry_run:
            value = {"preview": value, "network_writes": 0, "model_api_calls": 0}
        else:
            environment = _selfhost_environment()
            _authorize(value["plan"], environment, launch=False, remote=False)
            value = stop_session(
                root, session_id, driver=driver_for(root, value["plan"]), environment=environment
            )
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, SafeError, TimeoutError):
        _selfhost_refused()
    _selfhost_emit(value)


app.add_typer(selfhost_app, name="selfhost")


def _usage_errors() -> tuple[type[Exception], ...]:
    # Follow Typer's public exception module across vendored/external Click releases.
    try:
        error = getattr(import_module(typer.BadParameter.__module__), "UsageError", None)
    except ImportError:
        return ()
    return (error,) if isinstance(error, type) and issubclass(error, Exception) else ()


def main() -> None:
    """Unexpected failures must not dump request headers, environment values or locals."""
    try:
        from servegauge.storage.workspace import configure_workspace

        configure_workspace(Path.cwd())
        result = app(standalone_mode=False)
    except _usage_errors():
        # Parser messages may contain arbitrary argument values, including secrets.
        typer.echo("Invalid command usage; run servegauge --help or COMMAND --help.", err=True)
        raise SystemExit(3) from None
    except Exception:
        typer.echo(
            "Unexpected harness failure; inspect sanitized captures and docs/usage.md.", err=True
        )
        raise SystemExit(5) from None
    raise SystemExit(result)


if __name__ == "__main__":
    main()
