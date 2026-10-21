# Measurement methods and limits

ServeGauge measures observable endpoint behavior against declared requirements.
An invariant can be tested directly. A comparative claim needs a compatible
baseline or reference, a matched sampling design, and enough independent evidence.
No measurement establishes a provider's intent or uniquely identifies its hardware.

## Evidence and decisions

The runner reserves cost before dispatch, records sanitized exchanges, and
reconciles usage against the ledger. Derived scores can be reconstructed from
sealed raw evidence. Reference provenance and sampling design travel with a
result; changing a derived file cannot replace its underlying capture.

Inferential decisions include sample counts, intervals, margins, and reference
tiers. The implementation uses Wilson intervals for binary rates, paired or
clustered comparisons where appropriate, and multiplicity correction for the
declared family. Practical margins come from `configs/margins.yaml`. Insufficient
data, missing references, or unsupported assumptions produce INCONCLUSIVE.

PASS means the declared rule was satisfied under the tested conditions. It does
not mean a zero failure rate or universal reliability. A diagnostic statistic
without a qualified decision rule remains INFO.

## Fingerprints

FP1 and FP2 compare response distributions. FP3 measures logprob behavior when
the required information is present. FP4 uses a separately installed local
embedding model with pinned assets. Missing logprobs cannot establish logprob
equivalence. Undeclared dependence or cache reuse cannot be treated as new
independent evidence.

Reference calibration uses independent same-model captures and different-model
controls at the same design and sample size. A baseline pack needs repeated
captures, not just one response file. Different-model bands help interpret a
distance; a shared band does not prove identical model weights.

The optional `captured-scripted-text-v1` qualification is limited to one frozen
MOCK process and numerical runtime. Its verifier checks evidence, code, runtime,
requests, independence, and policy. It cannot qualify real providers. See
[reference packs](references.md) for its scope and setup.

## Operations and diagnosis

Operational checks record client-observed timing, throughput, attempts, failures,
and simulated task action costs. Network buffering, retries, and unavailable
preflight can affect interpretation. Missing windows stay missing; they are not
converted into successful observations or invented failures.

Diagnoses combine measured symptoms with rules, alternative explanations, and
falsifiers. A tool-call failure may have several causes, including request
formatting, model behavior, serving templates, or parser behavior. Similar
fingerprint changes can result from mixtures, noise, or other unmeasured changes.
Replay repeats the supporting checks and necessary controls under reservations.

## What the offline evidence establishes

The retained fingerprint and transport studies use declared scripted MOCK
processes. Counts, uncertainty, seeds, and hash-bound artifacts make those
experiments inspectable. They test the implementation and its behavior under
specified assumptions. They do not establish production false-alarm rates,
arbitrary dependence handling, real quantization sensitivity, or provider quality.

The transport study keeps its original source identity. Its laboratory results
are prerequisites for review, not operating admission for this checkout. The
public monitoring registry starts empty. Physical GPU execution and real-provider
qualification require separate evidence.

## Sharing results

Reports and bundles remain local until explicitly shared. Content-removed
exports retain hashes and structure instead of private prompt/response text.
Unknown or restricted benchmark text is aggregate-only in numeric exports.
Publication preflight checks selected inputs and reply gates; scripted mock
captures are not approved for upload as provider observations.
