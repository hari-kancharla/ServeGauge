# Using ServeGauge

Run commands from the checkout after `uv run python scripts/dev.py setup`.
For a wheel installation, start in an empty directory and run `servegauge init`,
then `servegauge setup-cache`. Initialization creates offline configuration with
zero live and GPU budgets. It does not install empirical qualification.

## Estimate first

These commands do not send requests or create reservations:

```text
uv run --offline servegauge estimate --plan configs/plans/mock_smoke.yaml
uv run --offline servegauge estimate --check-args --endpoint http://127.0.0.1:8011/v1 --model mock --profile gpt-oss-120b --tests P01 --level core
```

Choose one estimate mode. An estimate describes the configured design and price
assumptions. It does not discover current prices or guarantee a final invoice.

## Check requirements

The [README demo](../README.md#try-it-locally) measures one local output-cap
requirement. Use `--tests protocol`, `--tests tool-calling`, or `--tests operations`
to select other groups. See `servegauge check --help` for the complete interface.
`configs/requirements/example_app.yaml` shows an application requirements file;
pass it with `--requires` when preparing a compatible check.

A smoke run can be too small to establish a strict tolerance. Increasing the
sample size can improve precision, but cannot replace a missing reference or
unsupported qualification. Do not change a margin to obtain a passing result.

## Inspect saved evidence

Replace `RUN_ID` with the ID printed by a completed check:

```text
uv run --offline servegauge score --run RUN_ID
uv run --offline servegauge diagnose --run RUN_ID --redact-content
uv run --offline servegauge report --run RUN_ID --out reports/example --redact-content
uv run --offline servegauge site build --run RUN_ID --out reports/example-site
uv run --offline servegauge site serve --out reports/example-site
```

The site is served at `http://127.0.0.1:8040/`. These commands do not publish it.
Reports preserve earlier outputs and reject conflicting destinations. Use a
fresh directory for another report. Redacted bundles remove prompt and response
content while retaining structure and hashes. Review any artifact before sharing.

Diagnosis lists supporting checks, alternative explanations, and falsifiers.
`servegauge replay DIAGNOSIS_ID --yes` dispatches the supporting checks again,
including required controls. Replay uses reservations and the original budget;
it is a new run, unlike rebuilding a report from stored evidence.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Command succeeded, or the required gate is accepted. |
| 1 | At least one required measured check failed. |
| 2 | No failure, but a required check is inconclusive or unsupported. |
| 3 | Invalid command usage or configuration. |
| 4 | Budget or admission refusal, including missing `--yes` for a run. |
| 5 | Harness or transport failure. |

## Real endpoints

Live checks need a compatible endpoint configuration, a reviewed price snapshot,
an explicit budget, and a credential supplied through its named environment
variable. Never put a key in a configuration file or command argument. All paid
requests must use the runner and its reservations. No live check is enabled by
cloning this repository or running the local demo.

The source contains public endpoint snapshots used by configuration tests.
They are historical fixtures. Refresh discovery and prices before relying on
them for a new live run.

## Troubleshooting

Run `servegauge doctor --profiles` to check configuration and cached assets.
Run `servegauge setup-cache` if the tokenizer cache is missing. That command
downloads public tokenizer files, not model inference.

A refused fingerprint qualification usually means the runtime, source, evidence,
or sampling design does not match its recorded scope. Do not bypass the verifier.
See [reference packs](references.md). For a harness failure, inspect the sanitized
capture and the failing command's output before attempting a new run. Never
include credentials in an issue or diagnostic attachment.
