# Round 5: Independent node-canvas, runtime, and backend review

Review date: 2026-08-23  
Reviewer role: independent cross-check after the node-canvas V2 and ADMET-AI work  
Status: core graph/runtime behavior verified; two high-priority scope gaps remain

## Technical summary

The new builder is a real node-canvas interface rather than a reordered form. A
Windows Edge render of the generated 212,890-byte offline document showed a central
dotted canvas, draggable cards, explicit input/output ports, curved branch and join
connections, tier bands, pan/zoom/fit/arrange controls, a component palette, and a
node inspector. Research-only tools are visible but disabled. The current default is
a 14-node graph with four chemistry gates sharing the standardized parent population
and converging through one executable ALL policy node.

Parallel and serial are also different executable graphs, not visual labels. The
parallel regression runs both criteria over four parents and joins complete decision
streams; the serial graph gives the second criterion only the three survivors of the
first. Both produce the expected intersection for the selected deterministic
predicates. The default V2 graph compiled and every one of its 14 stages succeeded in
an end-to-end local run.

The ADMET-AI v2 adapter is carefully bounded and honest about its evidence contract.
It imports the optional backend lazily, requires an exact model-tree digest and an
explicit Torch-deserialization acknowledgement, snapshots regular symlink-free files,
runs bounded batches, emits typed long-form `prediction/v1`, fingerprints package and
model provenance, checks for mutation, cleans partial output, and declares
best-effort rather than deterministic execution.

This review nevertheless found two high-priority gaps against the requested product:

1. ADMET-AI and SA Score currently produce evidence but no registered gate consumes
   `prediction/v1` or `synthesis_score/v1`; those metrics cannot yet change the
   survivor set.
2. Most non-RDKit choices requested for each layer remain researched, disabled cards
   rather than executable adapters. The UI represents that limitation accurately,
   but the multi-backend implementation goal is not complete.

## Scope and method

The review covered compiler bindings, runtime behavior, default-pipeline execution,
decision joins, cache/provenance identity, large-input bounds, UI generation and CSP,
English-language output, optional-backend status, ADMET asset handling, and the
absence of docking/PRISM execution. It inspected implementation and tests, added a
real-runtime parallel/serial regression, generated the current HTML, rendered it in a
locally installed browser, and ran focused suites with no network dependency.

The companion backend report is
[`docs/research/0005-open-source-backend-landscape.md`](../research/0005-open-source-backend-landscape.md).
It distinguishes integrated, optional local, and research-only states and documents
the recommended placement of every requested metric.

## Severity-ranked findings

### High — score evidence cannot yet become a screening decision

`prediction.admet_ai_v2@0.1.0` emits `parent/v1` and `prediction/v1`.
`synthesis.rdkit_sa_score@0.1.0` emits `parent/v1` and
`synthesis_score/v1`. No registered plugin consumes either score contract and emits
`decision/v1`; the current policy join accepts parent rows plus decision streams only.
The builder's Parallel/Serial control consequently applies only to decision-emitting
chemistry and similarity gates.

Impact: a user can calculate ADMET and SA evidence, but cannot configure endpoint
thresholds, a maximum SA score, missing-value policy, or a multi-endpoint ALL/ANY
screen that changes survivors. The default SA branch is also independent of the
selection branch, so it cannot affect final selection.

Required correction: add typed evidence-to-decision adapters. An ADMET gate should
select exact endpoint and model IDs, threshold direction, units/label semantics,
missing/non-finite policy, and per-endpoint action before emitting complete decisions.
A synthesis-score gate should bind an exact method ID and threshold and continue to
state that the score is a proxy. These gates can then participate in the same explicit
parallel policy join as other criteria.

### High — executable alternative breadth is not complete

The UI catalogue includes ChEMBL Structure Pipeline, Lilly Medchem Rules,
MordredCommunity, Chemprop, RAscore, AiZynthFinder, FPSim2, ScaffoldGraph, BitBIRCH,
paretoset, and pymoo as reviewed options. The new research landscape additionally
reviews OpenADMET, Uni-Mol Tools, and Syntheseus, which are not yet catalogue entries.
Most of these projects do not have a registered plugin reference. A locally detected
module is still shown as `adapter pending` and cannot be dragged or exported, which is
correct and safe behavior.

Impact: the requirement for multiple usable implementations in each screening layer
is not yet met. In the current environment, scientific chemistry gates,
standardization, fingerprints, scaffolds, and the actionable property filters remain
primarily RDKit implementations. ADMET-AI and Open Babel have registered optional
adapters, but neither backend is available under the current local/default policy.

Required correction: prioritize ChEMBL standardization, FPSim2 indexed similarity,
Chemprop project checkpoints, and isolated AiZynthFinder route search as described in
the backend report. Each must remain disabled until its real adapter, assets,
contracts, and conformance tests exist.

### Medium — the ADMET package-code pin is recommended but not required

`expected_model_manifest_sha256` is mandatory, while
`expected_package_code_sha256` is optional. The adapter records and post-verifies the
actual installed package digest, and its best-effort determinism prevents normal
cross-run stage caching, so this is not a demonstrated cache-key collision. It is a
trust and reproducibility gap: a configuration can acknowledge executable checkpoint
deserialization without binding the installed ADMET package code to a previously
reviewed digest.

Required correction: make `expected_package_code_sha256` mandatory and test that a
missing or mismatched pin stops before importing ADMET-AI. Continue to explain that a
hash proves byte identity, not safety, and that trusted imported Python/model code is
not sandboxed or technically prevented from using the network.

Resolution (2026-08-24): completed. Both package-code and recursive model-tree
SHA-256 values are now mandatory configuration fields and are checked before the
dynamic ADMET-AI import. Regression tests cover missing and mismatched pins. The
integration guide and canvas help explicitly state that imported trusted code is not
sandboxed or network-blocked. The adapter also validates the official DataFrame SMILES
index position-by-position to prevent silent same-cardinality row reordering.

### Medium — optional-model configuration is not yet fully form-driven

The inspector renders scalar JSON-schema fields and offers an Advanced JSON editor.
ADMET endpoint bindings and per-checkpoint pins are arrays, so a future locally
available ADMET card would require manual JSON editing. Browser validation does not
currently implement the complete Pydantic schema for these complex fields; the CLI
will reject an invalid export safely, but that failure occurs later than a first-time
user expects.

Required correction: provide table/repeater controls for endpoint and checkpoint
bindings, run a bounded schema precheck before download, and keep server/CLI Pydantic
validation authoritative.

### Medium — older packaged engineering documents are not English

The Python source, generated HTML, and README contain no Han characters. Six older
files under `docs/research` and `docs/iterations` are still Chinese. This does not
change runtime behavior, but it conflicts with a literal reading of the requirement
that the whole software be English and makes the engineering record inconsistent.

Required correction: translate those documents without changing their technical
claims. The original Chinese documents in `plan/` are user-provided requirements and
should remain unmodified source material.

### Low — Windows reparse-point hardening is uneven

The million-file MOL2 source explicitly detects Windows reparse points as well as
symlinks. The ADMET asset walker rejects POSIX-style symlinks and non-regular entries
but has no equivalent explicit reparse/junction check. Its private snapshot and hashes
still bound the bytes ultimately loaded, so no output-integrity bypass was
demonstrated.

Recommended correction: reuse the source adapter's best-effort reparse-point check in
model and package asset traversal, and add a native-Windows junction test.

### Low — single-criterion Parallel labeling is semantically harmless but ambiguous

When a tier contains one decision criterion and is marked Parallel (ALL), the builder
uses the direct single-criterion path and creates no policy join. That graph is
scientifically equivalent to ALL over one criterion, and validation intentionally
checks joins only when at least two criteria exist, but a canvas label may imply a
join that is absent.

Recommended correction: display `Single criterion` until a second criterion exists,
or explain explicitly that a one-branch ALL join is elided.

## Verified behavior

### Real parallel versus serial execution

The added integration regression uses four actual molecules and the packaged source,
standardization, property-range, Rule-of-Five/QED, and policy plugins.

- Parallel: both gates consume the four standardized parents; one ALL join receives
  the parent port plus both complete decision ports.
- Serial: the Rule-of-Five/QED gate consumes the three property-gate survivors.
- Both retain exactly the expected two-parent intersection for these predicates.
- The compiled graphs have different bindings and revision identities.

This proves the user-facing modes alter runtime populations, not just node positions.

### Default V2 pipeline smoke

The generated default configuration was pointed to a temporary five-molecule CSV and
the final target was reduced to three. The compiler produced a 14-stage revision and
the local runner reported `SUCCEEDED` for:

`source`, `standardize`, `structural_gate`, `property_gate`, `drug_likeness`,
`structural_alerts`, `chemistry_policy`, `properties`, `fingerprint`,
`synthetic_accessibility`, `scaffolds`, `diversity_groups`, `final_select`, and
`shortlist`.

The final primary contract was `parent/v1`.

### Decision join, lineage, and bounded execution

- Parallel ALL uses exactly one `parent/v1` input and one or more `decision/v1`
  inputs with unique request ports.
- The native join uses disk-backed SQLite state and requires complete evidence for
  every parent/branch before publication. Missing branch evidence fails closed.
- Cache identities include request-port-qualified input artifact references,
  compiler bindings, full config, and the versioned plugin descriptor.
- The runner validates output contracts, producer, config, complete lineage, staging
  inventory, and hashes before commit or checkpoint reuse.
- Source plugins do not use unsafe cross-run config-only cache keys for mutable local
  files.
- CSV/SDF records have byte bounds, XLSX uses read-only iteration, Parquet uses batch
  scanning/snapshots, and MOL2 directory enumeration is disk-indexed instead of held
  in a Python list.

### UI and export safety

- Builder schema is V3 and persists node coordinates, viewport, tier modes, and
  explicit edges in `builder_layout` schema V2.
- The default hard-gate tier has four same-parent branches and one `all_required`
  policy node with five inputs.
- Serial transformation removes the policy join and rewires downstream parent inputs
  to the final gate survivor port.
- Disabled or unavailable nodes cannot be exported. Research alternatives have no
  draggable executable card and are labeled `cannot be added or exported`.
- The configuration is topologically sorted before export. Cycles, missing nodes,
  duplicate request ports, incompatible contracts, disabled dependencies, and
  malformed lead-reference selection are reported before download.
- The payload is HTML-escaped in a hidden textarea. Tests include a
  `</textarea><script>` payload and exact round-trip recovery.
- CSP allows only the exact inline CSS/JavaScript hashes; `connect-src`, objects,
  forms, frames, and base URLs are disabled. The implementation contains no `eval`,
  `new Function`, remote URL, inline event handler, or CDN dependency.

### ADMET contract and security checks

- Importing the builtin registry does not import ADMET-AI, Chemprop, Torch, Pandas,
  or RDKit through the adapter.
- Model inspection hashes a complete recursive regular-file manifest without loading
  a checkpoint and enforces the v2 ensemble-directory/`.pt` layout.
- Execution requires the model manifest and explicit executable-model trust flag,
  copies a private snapshot, verifies configured hashes before backend import, and
  rehashes package/model code after inference.
- Predictions are processed in configured batches and emitted in finite, numeric,
  long form keyed by parent, endpoint, and model. Missing columns, row-count changes,
  Boolean/non-finite values, load errors, and partial writes produce structured
  failures and cleanup.
- Uncertainty and calibration fields are null and metadata says unavailable instead
  of inventing confidence.

### Scope and language

No PRISM module, docking command, MD runner, or MMPBSA implementation is imported or
executed. Mentions in code and documentation are scope disclaimers only. Runtime code,
the current HTML, and the README are English.

## Verification record

| Check | Result |
|---|---:|
| `tests/ui/test_builder.py tests/cli/test_cli.py` | 56 passed |
| `tests/integration/test_parallel_serial_semantics.py tests/plugins/test_policy_join.py` | 7 passed |
| `tests/plugins/test_admet_ai.py` | 11 passed |
| artifact, atomic-write, and local-runner security suites | 89 passed |
| plugin-registry and source-ingestion security suites | 46 passed |
| backend, contract, config, and compiler suites | 64 passed |
| Total focused tests, with disjoint file groups | **273 passed** |
| Current default pipeline end-to-end smoke | 14/14 stages succeeded |
| `node --check` and `python -m compileall -q src` | passed |
| `git diff --check` | passed |
| English scan of `src`, generated HTML, and README | passed |

`ruff` and `mypy` are declared development dependencies but are not installed in the
current environment, so those two static checks were not executed. No package was
installed during the review.

## Browser QA and limitations

The current generated file was rendered with the locally installed Windows Edge in
headless mode at 1600 x 1000. The screenshot was written successfully and visually
confirmed the canvas, ports, branches, policy join, controls, palette, and inspector.
Edge emitted internal task-manager warnings, not a page-render failure.

Linux Playwright could not start its bundled Chromium because WSL lacks `libnspr4`,
`libnss3`, `libnssutil3`, and `libasound.so.2`. Windows Edge's debugging endpoint is
bound to Windows loopback and was not reachable from the WSL test process, so an
automated pointer-drag/export interaction was not added during this review. Compiler-
backed UI tests, JavaScript syntax checking, CSP checks, and a real browser render all
passed; touch, Firefox, Safari, and screen-reader behavior remain unverified.

## Release recommendation and next round

The node-canvas V2, true parallel/serial graph semantics, bounded local runner, and
honest research-option disabling are suitable foundations for continued use. Do not
describe the present package as completing ADMET/synthesizability filtering or as
having multiple executable tools in every layer until the two high-priority gaps are
closed.

The next implementation round should, in order:

1. add typed ADMET and synthesis-score threshold gates with complete decision output;
2. require the ADMET package-code pin (completed); add native-Windows reparse tests;
3. add first-class UI editors for endpoints/checkpoints and schema prevalidation;
4. implement ChEMBL and FPSim2 adapters, then Chemprop and isolated route search;
5. translate the six older engineering documents and run the complete test, ruff,
   mypy, native-Windows, and automated browser matrices.
