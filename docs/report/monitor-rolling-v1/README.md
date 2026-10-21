# Continuous transport candidate: offline study

This accepts a numerical/replay prerequisite for M14b2. It does **not** arm an
observatory alert, qualify providers, or establish model correctness. All candidate
outputs remain `qualified=false`, `INCONCLUSIVE`, reference tier `MOCK`.

## What changed

The preserved [epoch counterexample](epoch-counterexample.json) gradually reduces
monthly success from 95% to 75% at 800 trials/window. Resetting history at each
epoch produced zero candidate events across 36 months. The replacement retains
up to 121 complete continuous batches, including across monthly boundaries.
After a candidate alarm it resets to the latest complete window. Gaps, incomplete
or unsupported evidence, repeated seeds/captures and measurement changes cannot
bridge into a comparison. Original inactive v1 settings and old studies stay intact.

Each bit means first-attempt HTTP 200 versus declared mock HTTP 503 on canonical
P01 requests. Retries do not replace failed first attempts. This is a transport
observation, not a statement that the model answered correctly.

## Proposed scope and arithmetic

For any **fixed** rolling 30-day interval, at most S=100 predeclared measurement
variants and U=120 updates per variant are admitted. Source, endpoint, request
design or batch-size changes consume a new variant. Conflicting inputs or rolling
cap violations refuse before replay. Historical records are never deleted.

With family alpha=0.05 and expected-event target F=1, b=min(alpha,F)/(S U). For
a suffix of l complete batches, delta_l=b*2/(l*(l+1)). The weights over l>=2 sum
to one, covering every possible adaptive suffix start. A fixed cut with n0 and n1
independent bounded bits uses radius sqrt((1/n0+1/n1)/2 * log(4*(n0+n1)/delta_l)).
The two-sided Hoeffding tail per cut is at most delta_l/(2*(n0+n1)); fewer than
n0+n1 cuts give a bound below delta_l. Union over starts, updates and variants
gives at most S U b <= 0.05 for that fixed family. Since at most one event is
emitted per update, the same sum bounds expected false-event count under the null.

This is a project derivation under a predeclared independent potential-observation
process. It is not a published theorem about ServeGauge or an all-future-years
simultaneous guarantee. The [original ADWIN paper](https://www.cs.upc.edu/~gavalda/papers/adwin06.pdf)
provides the bounded-observation cut starting point. Adaptive suffix allocation
and rolling accounting are additional policy. Arbitrary missingness/dependence,
post-hoc stream selection and provider traffic cannot inherit this scope.

Allocations use exact fractions rounded down; radii use directed decimal
operations and upward conversion; decisions compare integer cross-products.
Effect endpoints round outward and cover the selected **input expected contrast**
within the declared family scope. They are not confidence bounds on a change time
or a root cause, and do not assert conditional coverage given selection. See
[Python Decimal](https://docs.python.org/3.12/library/decimal.html).

## Observed evidence

- All 12,000 supported S/U budget shapes round down tightly.
- All 14,520 canonical 400/800-unit cut shapes through 121 batches pass the
  binary lattice audit; 38 public frozen-detector boundary comparisons agree.
- Independent Decimal-120 count replay agrees with the complete public raw-bit
  controller on 120 held-out trajectories (45 windows each, both batch sizes,
  resets and missing windows).
- Native runner capture retains 800 logical units over two windows; first
  attempts agree with the HTTP spy, retries and final outcomes stay distinct.
- Nine fixed cases use 2,000 trajectories each, with canonical per-window sizes.
  No parameter was selected or tuned from these evaluation results.

| Case | Windows/trajectory | Detection or any-event count | Rate [95% Wilson] | Conditional mean delay [95% bootstrap], windows | Misses |
|---|---:|---:|---|---|---:|
| null-half | 150 | 0/2000 | 0.0000 [0.0000, 0.0019] | not applicable | not applicable |
| null-three-quarters | 150 | 0/2000 | 0.0000 [0.0000, 0.0019] | not applicable | not applicable |
| null-high | 150 | 0/2000 | 0.0000 [0.0000, 0.0019] | not applicable | not applicable |
| abrupt-core | 120 | 2000/2000 | 1.0000 [0.9981, 1.0000] | 0.395 [0.374, 0.416], n=2000 | 0 |
| gradual-core | 120 | 2000/2000 | 1.0000 [0.9981, 1.0000] | 14.754 [14.713, 14.797], n=2000 | 0 |
| gradual-monthly | 36 | 2000/2000 | 1.0000 [0.9981, 1.0000] | 5.743 [5.723, 5.764], n=2000 | 0 |
| mild-core | 120 | 1984/2000 | 0.9920 [0.9870, 0.9951] | 59.862 [59.473, 60.262], n=1984 | 16 |
| mild-monthly | 36 | 247/2000 | 0.1235 [0.1098, 0.1386] | 21.530 [21.344, 21.713], n=247 | 1753 |
| missing-gradual | 120 | 149/2000 | 0.0745 [0.0638, 0.0868] | 23.430 [22.409, 24.450], n=149 | 1851 |

The three null rows mean any candidate over 150 daily windows of one simulated
stream. They do not empirically certify the full 100-stream observatory. Zero
observed events is not zero risk. Bootstrap zero-variance intervals remain
descriptive; the Wilson upper bounds remain visible. Delay zero means detection
at the first changed observation, not instantaneous wall-clock detection.

Large drops are 0.95 to 0.75. Daily gradual drops span 30 windows after 30 stable
windows; monthly drops span 12 windows after 12 stable windows, with 36 total.
Mild cases drop only 0.05. Missing-gradual independently removes 15% of windows
and clears comparison history; 1,851/2,000 trajectories are censored. It is a
stress case, not an operating readiness guarantee. Small monthly shifts and
frequent missingness remain weak. No query budget, margin or old threshold changed.

[Interval audit](interval-audit.json) reports trajectory-family failures, including
vacuous no-event trajectories. [Full results](results/study.json) retain event
counts, seeds, conditional delays, censoring and effect coverage. NPZ files retain
all generated counts, availability masks and means; event JSON retains every cut.

## Reproduction and provenance

Run from the accepted source checkout, with a new output directory:

    uv run python docs/report/monitor-rolling-v1/study.py --out .cache/evidence/rolling-study-fresh

The recipe refuses a changed package hash or an existing output directory. It
blocks sockets during computation, uses no model APIs, and does not alter budgets.
Its [predeclared plan](plan.json) pins settings and a separate held-out namespace.
The source, plan, script and every result artifact have hashes in study.json.

The initial scratch study finished computation but failed final JSON serialization
of a NumPy integer. Its output/log/script remain in .cache/evidence under v1.
The corrected durable recipe casts the supplementary count to an integer, uses
a fresh output, and reproduces the same nine headline measurements; no parameter
was retuned. Epoch prototype and earlier rounding/cut counterexamples remain saved.

## Remaining admission work

Source/config-bound runtime admission, complete preflight/readiness studies,
multi-stream rolling-family validation, immutable event records and monitor CLI
integration remain. Provider/dependent/cache channels stay unqualified. Continue
local notice/reply records, table/site projection, export dry runs and release
checks. No contact, publication, schedule activation or provider spending occurred.
