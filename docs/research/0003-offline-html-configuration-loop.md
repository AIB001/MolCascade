# Offline HTML → Next-Round Configuration Loop

Date: 2026-08-23

## User Workflow

Every stage, whether successful or failed, generates a self-contained `report.html`. The report embeds only aggregate metrics and a bounded sample, never a million-row structure table. In the browser, the user can:

1. Review input, pass, reject, warning, conservation, top-reason, distribution, and plugin-provenance information;
2. Review the current artifact/revision, as well as available backends and health checks for the next layer;
3. Edit only the “next-layer soft parameters/quotas” permitted by the report; identity policies and hard rules already executed remain read-only;
4. Validate locally in the browser and download `next-config.json`;
5. Run `molcascade validate next-config.json`, followed by a normal run or dry run;
6. Have the backend perform strict parsing, allowlist checks, and canonicalization again, producing a new revision ID.

Configurations produced by the browser are always untrusted input. HTML cannot directly mark a run as successful or modify an existing artifact.

## Single-File MVP Rather Than a Web Service

The default report is a single HTML file with embedded plain CSS, fixed JavaScript, and HTML-escaped JSON data. It works completely offline. The configuration is downloaded as JSON through a Blob. JSON itself is a MolCascade-supported configuration format, which avoids reimplementing an unsafe or semantically inconsistent YAML parser in the browser.

A future optional `molcascade report serve` command binds only to `127.0.0.1` and supports large, paginated samples; it is not a remote service. The default single-file report requires no FastAPI, Node, CDN, or browser extension.

## Report Contract

`report_summary/v1` contains at least:

```text
run_id, revision_id, stage_id, status
input_artifact_ids, output_artifact_id
plugin_id, plugin_version, backend_version
input_count, pass_count, reject_count, warn_count
conservation_ok, top_reason_codes
preaggregated_histograms, drift/scaffold/cluster/basket metrics
elapsed_seconds, peak_rss, cache_hit
next_stage_spec, allowed_editable_paths
base_config, schema_id, schema_hash
sampling_method, sampling_seed
```

The envelope downloaded by the browser is:

```json
{
  "schema_version": 1,
  "base_revision_id": "...",
  "source_artifact_id": "artifact:sha256:...",
  "allowed_editable_paths": ["stages.3.config.threshold"],
  "pipeline": {}
}
```

The CLI rejects mismatched base revisions/artifacts, unauthorized field modifications, duplicate keys, NaN/Infinity, unknown fields, and schema-hash mismatches.

## Security Requirements

- Make zero external network requests and use no CDN.
- Place user, molecule, and exception text only through `textContent`; never concatenate it into `innerHTML`.
- Do not use `eval`, `Function`, `unsafe-eval`, or a remote Vega loader.
- Escape `</script>`, `<>&`, U+2028, and U+2029 in embedded data; pin the fixed script with a CSP SHA-256 hash.
- Do not render arbitrary SVG/HTML; in the future, molecular diagrams must be generated as sanitized static PNG/SVG by a controlled renderer.
- Sanitize download filenames and revoke Blob URLs after use.
- The page may only propose a new configuration; it cannot write to artifact directories, run external commands, or recover a hard rejection.
- The default report-size budget is 10 MB, and the default sample is limited to 2,000 rows.

## Research on Optional UI Components

If the plain-JavaScript MVP becomes insufficient, the following components may be pinned, vendored, and recorded with checksums/licenses:

- Plotly: fully offline charts, but the bundle is approximately 3 MB; include only aggregate data.
- Tabulator: MIT-licensed virtual tables; the sample-row limit still applies.
- JSON Editor + Ajv: MIT-licensed JSON Schema forms and browser preflight checks; backend Pydantic validation remains authoritative.
- js-yaml: MIT-licensed; add only if YAML downloads are genuinely required, and continue to use the backend's strict loader for input.
- Vega-Lite: BSD-3-Clause; may replace charts declaratively, but remote URL loaders are prohibited.

The MVP starts as a single file with no third-party JavaScript, reducing supply-chain, CSP, and offline risks.

## HTML Conformance

- Playwright/browser smoke tests pass with the network completely blocked, with zero requests and zero CSP violations.
- A downloaded configuration can be read back by MolCascade's strict loader, and its canonical revision hash matches the preview.
- Injection cases include `</script><img ...>`, `javascript:`, malicious SVG, U+2028, and U+2029.
- Only allowlisted paths can be modified; identity fields and hard-gate rules already executed remain read-only.
- A million-row artifact contributes only statistics and a reproducible sample, never the full table.

References: [OWASP XSS Prevention](https://cheatsheetseries.owasp.org/cheatsheets/XSS_Prevention_Cheat_Sheet.html), [MDN CSP script-src](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/script-src), [Plotly write_html](https://plotly.com/python-api-reference/generated/plotly.io.write_html.html), [Tabulator](https://tabulator.info/), and [Ajv](https://ajv.js.org/).
