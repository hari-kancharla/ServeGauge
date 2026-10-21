# Report assets

The report package renders local HTML, Markdown, and JSON from reconstructed
evidence. See [usage](../../../docs/usage.md) for report and site commands, and
[methods](../../../docs/METHODS.md) for interpretation and limitations.

`templates/report.html.jinja` expects an explicitly constructed report context:
run/profile/capture metadata; outcome summaries; endpoint labels; check rows with
formatted estimates, intervals, reference tiers, margins and p-values; diagnoses
with alternatives/falsifiers; provenance rows; and an embedded chart payload.
Missing measurements stay unavailable rather than becoming zero.

Use Jinja `StrictUndefined` and autoescaping enabled for the `.jinja` extension.
The embedded payload uses `tojson`; no captured text becomes HTML or chart code.
Asset prefixes and optional methods/bundle links must be constructed from trusted
local paths by the renderer, never copied from response metadata. Relative paths
are validated before rendering; numeric values retain their recorded provenance.

`site/report.js` only filters pre-rendered rows and draws outcome **counts**.
The chart and its exact data table follow the same filters. It is not a provider
quality score. All checks and initial count rows are server-rendered, so the
evidence remains readable without scripts. Filters require JavaScript.

The pinned Vega browser bundle and its BSD-3-Clause license are in `vendor/`;
`manifest.json` records retrieval URLs, hashes and sizes. Copy the JS, CSS, Vega
bundle and license alongside a generated report. There is no runtime CDN fetch.
The fixed local Vega specification requires `unsafe-eval`; external chart loaders
are disabled and the report CSP blocks connections. Do not accept user-supplied
Vega specifications, expression strings, CSS colors or resource URLs.

