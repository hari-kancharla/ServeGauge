# Development and verification

ServeGauge uses Python 3.12, uv, a `src` layout, Ruff, strict mypy, and pytest.
The same Python task runner is used on Windows, macOS, and Linux.

```text
uv run python scripts/dev.py setup
uv run python scripts/dev.py check
```

Setup installs locked dependencies and downloads checksum-verified public
tokenizer assets. It makes no paid model calls. The check runs format validation,
lint, strict types, a tracked-file secret scan, and socket-blocked offline tests.
Each of core, statistics, verdicts, diagnosis, runner, and scoring must reach
85% combined statement and branch coverage.

## Additional checks

```text
uv run --offline pytest --no-cov -m slow tests/integration/test_distribution_boundary.py
uv run --offline pytest --no-cov -m slow tests/integration/test_monitor_operating_workflow.py
uv build --offline
```

Distribution tests build real wheels and source archives with private canary
files present. The canaries must be absent from both artifacts and unchanged in
the checkout. The monitoring integration test exercises actual local producers,
sealed raw reconstruction, admission, repeat evaluation, and report/export views.
Its receipt is test-only and cannot qualify a deployed monitor.

Slow statistical studies can take substantially longer than the default tests.
Use `servegauge validate --help` to choose a study and a fresh output directory.
Smaller diagnostic runs do not inherit a full study's acceptance.

## Qualification and portability

The pinned text-equivalence study was measured on macOS arm64 with CPython
3.12.15, NumPy 1.26.4, and SciPy 1.17.1. Tests that require admission run on that
runtime. Other runtimes must reject the qualification; ordinary collection,
scoring, and refusal tests still apply. Passing general tests on another platform
does not extend this empirical qualification.

The repository retains the frozen inputs needed by these tests and their
qualification verifiers. This includes two captured mock calibration packs and
the transport laboratory's sufficient-count arrays. They account for most of
the checkout's size. Their hashes and original observations are preserved.
They contain no live provider captures or account credentials.

## Repository boundary

Keep runtime code in `src`, tests in `tests`, public configuration in `configs`,
model descriptions in `profiles`, and user documentation in `docs`.
Private instructions, local budgets and ledgers, caches, raw captures, and
generated reports stay outside Git. The source archive also excludes these
private inputs. The wheel contains the package and a small offline workspace
template; it does not include the large laboratory corpus.

Changes to scoring or qualification must retain uncertainty, matched references,
and declared margins. Never reinterpret a missing measurement as a pass or
increase a tolerance to make a test succeed. Keep side effects in adapters,
the runner, and storage. Keep scoring and statistical decisions pure.
