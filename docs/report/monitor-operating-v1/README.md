# Full transport-family and readiness laboratory

This is the M14b3b1 **offline laboratory prerequisite**, not operating admission.
No operating registry is installed and no alert is armed. All data are constructed
under a declared independent-binomial mock model, reference tier `MOCK`. Native
captures separately verify the actual local generator/request path. These results
do not qualify providers, hidden dependence, weights, hardware or production timing.

## Fixed design

Each null case uses 2,000 independent simulated families, each with 100 streams,
400 potential first-attempt units per update, 120 priming and 120 evaluation updates.
Six-hour logical windows make the evaluation interval 30 days. History retains at
most 121 batches and resets after an alert or an unavailable window. All stream
and update limits are checked by the complete public controller in conformance tests.

Three complete nulls use success probabilities .50, .75 and .95. A fourth uses .50
with basic preflight readiness `1-(1-p)^3`, representing two retries. Unavailable
windows stay missing; their potential generated counts cannot vote in detection.
The namespace, shapes and acceptance settings were fixed before evaluation in
[plan.json](plan.json). No numerical cut, policy, margin or old default was tuned.

The unchanged v2 allocation uses summable suffix weights and a fixed-family bound
of **3/61**, at most 0.049180327868852465 after upward float conversion, below the
configured 0.05 interval budget and expected false-event target of 1. The derivation
is in the [continuous-history report](../monitor-rolling-v1/README.md). It assumes
independent bounded potential observations and predeclared cadence/stream scope.

## Observed null families

| Case | Families | Any-event families | Rate [individual 95% Wilson] | Available fraction, mean [95% family bootstrap] |
|---|---:|---:|---|---|
| null-half | 2000 | 0 | 0.000000 [0.000000, 0.001917] | 1.000000 [1.000000, 1.000000] |
| null-three-quarters | 2000 | 0 | 0.000000 [0.000000, 0.001917] | 1.000000 [1.000000, 1.000000] |
| null-high | 2000 | 0 | 0.000000 [0.000000, 0.001917] | 1.000000 [1.000000, 1.000000] |
| null-readiness-half | 2000 | 0 | 0.000000 [0.000000, 0.001917] | 0.875134 [0.875004, 0.875264] |

Zero observed alerts is not zero risk. Restricted run lengths are right-censored
at 120 evaluation updates. A zero-variance bootstrap interval is descriptive; it
is not an upper confidence bound of zero operating risk. The finite-family
calculation and empirical observations have separate roles. Per-case intervals
are individual intervals, not a simultaneous confidence statement about all cases.

## Readiness and detection

Each strong case has 2,000 trajectories and the same basic-preflight readiness
process. Daily abrupt/gradual drops are .95 to .75 after 30 stable observations;
the gradual ramp lasts 30 windows. Monthly full uses 800 units/window, 12 stable
windows and a 12-window ramp, with 36 total. Daily followup is 90 windows.

| Case | Detected/n | Power [95% Wilson] | Conditional mean delay [95% bootstrap], windows | Censored |
|---|---:|---|---|---:|
| abrupt-core | 1951/2000 | 0.97550 [0.96776, 0.98142] | 0.3906 [0.3701, 0.4126], n=1951 | 49 |
| gradual-core | 1972/2000 | 0.98600 [0.97984, 0.99030] | 15.0827 [14.9589, 15.2262], n=1972 | 28 |
| gradual-monthly | 1987/2000 | 0.99350 [0.98891, 0.99620] | 5.7383 [5.7071, 5.7710], n=1987 | 13 |

All strong-case power lower bounds exceed .8. Delay excludes missed trajectories;
censoring remains visible. Delay zero means the first changed observation, not
instantaneous wall-clock detection. Bootstrap resampling uses 2,000 independent
family/trajectory resamples. Readiness followup spans multiple 30-day families;
it does not acquire a whole-followup simultaneous .05 guarantee.

The earlier [mild monthly and 15% missingness failures](../monitor-rolling-v1/README.md)
remain unchanged. A complete outage may fail basic preflight and leave this P01
channel unavailable. Native acceptance checks that this produces missing evidence,
rather than 400 imputed failures or a service-up claim.

## Conformance and provenance

- 120 held-out paths compare count replay with complete raw-bit replay at both
  canonical sizes, including gaps, resets and reviewed measurement identities.
- Two additional full-policy paths use 100 streams and 240 updates, exercise the
  121-batch history cap, readiness, a producer-version seam and a gradual change.
  The 198 selected event contrasts agree in counts, inputs, direction and bounds.
- Actual isolated producers differ by an unused Python inventory leaf while
  retaining the identical reviewed generator and runtime. Actual source hashes
  remain visible. Native core and full P01, retry extraction and all-error preflight
  are tested through reservations/sealed raw records without sockets or credits.
- Runtime evidence records interpreter/build/platform, dependency versions, lock
  hash and RNG code images. The host RNG is built into the interpreter/shared
  library, so an absent extension file is not mistaken for absent provenance.
- A static local import closure pins the generator. An unrelated unimported leaf
  can change the package inventory without claiming a changed generator; imported
  changes, unknown dynamic imports and absent runtime cannot borrow approval.

[study.json](study.json) contains hashes, all rates, bootstrap intervals, coverage
and censoring. Each NPZ chunk retains counts, availability, probabilities and
individual family/trajectory seeds; matching JSON retains every emitted cut.
Only sufficient counts are simulated, not billions of actual HTTP requests.
Every native captured request remains separate from these constructed experiments.

## Reproduction

From the pinned source checkout, with a fresh output:

    uv run servegauge validate monitor-transport --out .cache/evidence/transport-study-fresh

A smaller `--replicates` run is diagnostic and cannot qualify. Outputs refuse
overwrite, source changes invalidate a run, and sockets are blocked during the
experiment. Component/runtime/plan/recipe hashes and archived code identify the
executed procedure. `eligible_for_operating_review=true` is not an operating receipt.

## Limits and remaining work

The probability model is an IID mock model backed by source/native checks and
observed seed experiments. Python pseudorandomness is not a cryptographic or
machine-security attestation. Provider/correlated/cache processes remain outside
scope. The confidence scope is a fixed declared cadence family, not all future
years, conditional-on-alarm coverage, arbitrary backfill/lateness or a production
clock guarantee. Production scheduling/telemetry requires separate owner validation.

Finish source-bound event admission/storage and local CLI integration before any
operating event is enabled, then notice/reply records, table/site projections,
export dry runs and release checks. Publishing/contact/schedule activation stay
owner actions. New provider spending in this work is $0.
