# ServeGauge

Test what an LLM endpoint actually delivers.

ServeGauge checks an inference endpoint against your application's requirements
and, when available, a compatible baseline or reference. It records the requests,
measures the responses, and produces results you can inspect and replay.

Use it to investigate output limits, tool calling, structured output, context
handling, response distributions, accounting, and operational behavior. Each
decision includes the evidence behind it. When the evidence is insufficient,
the result is **INCONCLUSIVE**.

**Status:** development release, `0.1.0.dev0`. The offline harness and scripted
fault tests are implemented. General provider qualification, physical GPU
validation, and production monitoring remain open. A local PASS is limited to
the check, reference, and conditions that produced it.

## Try it locally

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). The commands work in
PowerShell and macOS or Linux terminals. Initial setup downloads dependencies
and checksum-verified tokenizer files. The demo uses no API key or paid credits.

```text
git clone https://github.com/hari-kancharla/ServeGauge.git servegauge
cd servegauge
uv run python scripts/dev.py setup
uv run --offline servegauge --help
```

Start a local scripted endpoint:

```text
uv run --offline servegauge mock serve --port 8011 --seed 1304
```

In a second terminal, from the same directory:

```text
uv run --offline servegauge check --endpoint http://127.0.0.1:8011/v1 --model mock --profile gpt-oss-120b --tests P01 --level core --yes
```

This checks the output-cap requirement with 400 local trials. Copy the printed
run ID in place of `RUN_ID` to build a report:

```text
uv run --offline servegauge report --run RUN_ID --out reports/demo --redact-content
```

Open the HTML file in `reports/demo`. Keep the run ID: scoring, diagnosis, and
report generation can reconstruct results from the saved evidence. Stop the mock
with Ctrl-C when finished. Use a new output directory for each report.

## What you get

- **Requirement checks:** protocol behavior, output caps, tool calls, structured
  responses, context handling, and generated reasoning tasks.
- **Comparisons:** distribution fingerprints and matched baseline checks, with
  explicit reference tiers and uncertainty.
- **Evidence:** sanitized captures, local HTML and Markdown reports, diagnostic
  bundles, and reserved replay of supporting checks.
- **Offline investigation:** scripted faults, missing-data and dependence tests,
  change-detection studies, and local monitoring workflows.
- **Spending controls:** estimates, reservations, reconciliation, and hard caps.
  This checkout starts with every live and GPU budget set to zero.

ServeGauge can detect a difference in observed behavior. That difference alone
cannot establish which weights, quantization, hardware, or internal parser a
provider uses. It also cannot establish intent. Diagnoses retain plausible
alternatives and ways to test them.

## Understand the result

| Result | Meaning |
| --- | --- |
| PASS | The measured requirement meets its declared rule and evidence requirements. |
| FAIL | The measured requirement violates its declared rule with sufficient evidence. |
| INCONCLUSIVE | The data, reference, sample size, or qualification cannot support a decision. |
| INFO | A diagnostic observation without a qualified pass/fail interpretation. |
| NOT_SUPPORTED / NOT_APPLICABLE | The requested check is unavailable or outside the endpoint's applicable scope. |

The check command returns `0` for an accepted gate, `1` for a measured failure,
and `2` for unresolved required checks. See [usage](docs/usage.md) for the other
exit codes, estimates, requirements files, and reports.

## Documentation

- [Usage and troubleshooting](docs/usage.md)
- [Methods and limitations](docs/METHODS.md)
- [Reference packs and fingerprint qualification](docs/references.md)
- [Local monitoring](docs/monitoring.md)
- [Development and verification](docs/development.md)

`configs/endpoints/` holds three OpenRouter endpoint discovery snapshots taken on
2026-10-18: `openai/gpt-oss-120b` (24 endpoints), `deepseek/deepseek-v4-pro-0813`
(23), and `z-ai/glm-5.3` (46). They are inputs for owner runs, not published
measurement results. This repository does not include results from a live
multi-endpoint study.

The retained laboratory files support reproducible tests. They describe scripted
MOCK processes, not a ranking or certification of real providers. Private
captures, account state, development instructions, and live pilot reports are
not part of this repository.

## Development

```text
uv run python scripts/dev.py check
uv run --offline pytest --no-cov -m slow tests/integration/test_distribution_boundary.py
```

The main check runs formatting, lint, strict type checking, secret scanning,
offline tests, and coverage gates. Default tests block sockets. Slow studies and
tests that deliberately start local servers are separate. Empirical fingerprint
qualification is tied to its recorded runtime; other platforms must refuse that
qualification until independently validated.

## Related work

[Model Equality Testing](https://arxiv.org/abs/2410.20247) studies statistical
comparison of API and reference output distributions.
[Promptfoo](https://www.promptfoo.dev/docs/intro/) provides application evaluations
and red teaming. ServeGauge focuses on explicit endpoint requirements, qualified
comparisons, and replayable evidence. A direct comparative quality claim would
require a separate matched evaluation.

## License

Copyright 2026 Hari Krishna Kancharla. Licensed under [Apache 2.0](LICENSE).
Citation metadata is in [CITATION.cff](CITATION.cff).
