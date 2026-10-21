# Local monitoring

The repository includes a local mock observatory and a disabled owner template.
The example at `configs/workflows/observatory.yml.example` is outside
`.github/workflows` and cannot register a schedule. Cloning this repository does
not start services, spend credits, contact providers, or publish observations.

Preview a mock window:

```text
uv run --offline servegauge monitor tick --cadence daily --config configs/observatory.mock.yaml --dry-run
```

Remove `--dry-run` to execute the in-process mock workflow. It uses the ordinary
runner and reservations at zero cost. Repeating a completed window checks its
saved evidence without making another set of requests. Interrupted windows
require explicit resume; existing captures are preserved.

Operational observations and admitted change events are separate. The public
transport admission registry is empty. Stored observations do not become a
qualified production alert merely because a threshold is crossed.

An explicit reviewed proof root can be evaluated against existing raw captures:

```text
uv run --offline servegauge monitor evaluate --observatory-id OBSERVATORY_ID --root . --proof-root PATH --read-only
```

The verifier requires matching source, policies, complete evidence, lock,
runtime, and admitted producers. It cannot borrow the proof root's claimed
runtime in place of the runtime actually executing. An absent or incompatible
receipt must remain unqualified. This command does not dispatch requests.

The retained transport laboratory assumes a declared independent mock process.
Its counts, missing-window behavior, intervals, and source pins are evidence
for that experiment. They do not validate arbitrary provider dependence,
cache bursts, nonstationarity, production clocks, or indefinite monitoring.

Use `servegauge site build` for a local view of selected captures and
`servegauge publish hf --dry-run --run RUN_ID --out reports/export` to stage
numeric tables without uploading. Scripted mock observations remain blocked
from public dataset upload. Provider results require separate review and
publication prerequisites.

The Pages example is also stored as `configs/workflows/pages.yml.example`. It
is a manual preparation template, not an installed publishing workflow. Enabling
either example requires a separate review of its configuration and permissions.
