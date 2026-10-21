# Reference packs

A reference pack binds its source captures, sampling design, policies, and tier.
The tier records where the evidence came from; a filename or boolean cannot
upgrade an unqualified capture.

For existing captured MOCK runs:

```text
uv run --offline servegauge refs build --profile gpt-oss-120b --tier MOCK --from-run REFERENCE_RUN --control-run CONTROL_A_RUN --control-run CONTROL_B_RUN
uv run --offline servegauge score --run TARGET_RUN --reference PACK_ID
```

Replace the IDs with sealed captures. The source needs at least three independent
same-model reference captures and two different-model controls at matched design
and sample size. A combined mock capture can contain all three references and
both `other:control-a` and `other:control-b` backends. Reusing a capture or random
stream does not create independence.

For a trusted baseline, supply two additional independent captures from the
same trusted endpoint:

```text
uv run --offline servegauge refs build --profile gpt-oss-120b --tier BASELINE --from-run BASELINE_RUN --baseline-endpoint ENDPOINT_ID --reference-run REPEAT_A_RUN --reference-run REPEAT_B_RUN --control-run CONTROL_A_RUN --control-run CONTROL_B_RUN
```

These commands read stored captures. They do not send new requests. The source
URL, model, pin, defaults, design, and policy must match. Unknown sampling design,
missing fixed logprob anchors, or low effective sample size keeps the result
inconclusive. Known cache lineages and session dependence must be declared.

## Scoped text qualification

`--fingerprint-qualification captured-scripted-text-v1` opts into a measured
scripted-mock text-equivalence rule. It requires exactly three independent
references and two distinct controls, sampling version v2, healthy logit jitter
0.3, session size one, no shared cache groups, and the original requests/policy.
Its S6 design is:

```yaml
S6:
  methods: [FP1, FP2, FP3]
  fp1_cells: 40
  fp1_samples: 90
  fp2_cells: 25
  fp2_samples: 30
  fp3_samples: 400
```

Requests use temperature 1, low reasoning, and output cap 2048. The complete
FP1/FP2/FP3 family and original 0.03 text margins are required. Missing logprobs,
an extra method, a stricter margin, a reused seed, or a mismatched target prevents
qualified text PASS. FP3 and FP4 remain informational without an applicable
practical margin.

The registry `configs/fingerprint_qualification.json` pins the evidence and
implementation. Its measured runtime is CPython 3.12.15, NumPy 1.26.4, SciPy
1.17.1, macOS arm64. Other runtimes may collect and compare responses but must
refuse this qualification. No runtime field should be edited to bypass that rule.

Installed workspaces begin unqualified. `servegauge init --evidence-root PATH`
can explicitly bind a reviewed source corpus when its verifier accepts every
pin. It does not automatically grant monitoring admission or certify a provider.

## Local embeddings

FP4 is optional. Install the `embed` extra and supply local model files matching
`configs/embedding_minilm_v1.json`. Downloaded files must match the configured
pinned revision and checksums. Embedding diagnostics are separate from the scoped text
qualification above and from physical GPU validation.
