# Round 6 — Constrained canvas and actionable numeric evidence

Date: 2026-08-24

## Trigger

Browser review exposed two usability failures in the node canvas:

1. dashed tier frames could overlap after nodes were dragged, so tier ownership was not
   visually reliable; and
2. the connector accepted structurally confusing edges, including primary-flow bypasses
   and decision edges outside their policy tier.

The independent Round 5 audit also found that ADMET-AI and RDKit SA Score produced typed
evidence but no generic component converted that evidence into an explicit filtering
decision. A score producer alone must not be presented as a gate.

## Canvas constraints

Tier frames are now geometry-backed lanes. The browser measures every rendered node in a
tier, adds fixed horizontal and vertical padding, and shifts the complete tier to the right
until it has an 84-pixel gap from the previous occupied tier. Dragging a node recalculates
the affected lane positions and guide bounds. The allocator is a pure JavaScript function
with deterministic geometry tests.

Connection rules are also a pure, shared decision function used by connector highlighting,
manual connection, auto-wiring, rendering classification, and pre-export validation.

| Output contract | Allowed placement | Visual route |
|---|---|---|
| `parent/v1` | same tier only in Serial mode, or the next occupied tier | solid gray orthogonal flow |
| `decision/v1` | same-tier criterion to same-tier policy join only | dashed purple decision bus |
| numeric evidence for a prediction/synthesis gate | same tier and co-produced with the gate's parent input | dotted teal side evidence |
| another exact typed contract | only a target that declares that exact input contract | dotted teal side evidence |

Backward edges, cycles, incompatible contracts, primary-flow tier bypasses, cross-tier policy
decisions, and branch-survivor input to a Parallel policy join are rejected immediately with
an English reason. Existing or imported invalid bindings are checked by the same rules and
block export; they are not silently grandfathered.

Parallel primary fan-out and policy decision fan-in now use shared orthogonal buses. A policy
join's shared-parent route detours around the criterion stack. This removes overlapping
cubic curves while keeping primary, decision, and evidence semantics visually distinct.

## Actionable numeric evidence

Two native, disk-backed plugins were added:

- `prediction.numeric_evidence_gate@0.1.0` consumes one `parent/v1` population and one
  `prediction/v1` dataset from the same producer. It selects one exact endpoint and model,
  applies an inclusive numeric window, and emits filtered parents plus a complete
  `decision/v1` stream.
- `synthesis.numeric_evidence_gate@0.1.0` consumes one `parent/v1` population and one
  `synthesis_score/v1` dataset from the same producer. It binds one exact method, verifies
  the declared score direction, applies an inclusive window, and emits the same pair of
  outputs.

Both gates use a temporary SQLite database instead of a whole-library Python mapping. They
reject duplicate parents, duplicate evidence keys, unknown parent IDs, non-finite numbers,
ambiguous automatic model/method selection, and synthesis-direction conflicts. Missing
evidence produces an explicit `REJECT`; it never becomes an implicit pass. Partial Parquet
and SQLite files are removed after failure.

The default pipeline now places a synthesis-score gate immediately after the RDKit SA-score
producer. Its starter policy retains SA scores at or below 6 under the producer's
`HIGHER_HARDER` direction. This is a visible, editable project triage threshold—not a
universal cutoff and not evidence of a retrosynthesis route.

The optional prediction gate is available beside ADMET-AI. Its curated HIA probability
threshold is explicitly labelled illustrative and must be reviewed against the exact
endpoint semantics, model provenance, applicability, and project costs before execution.

## Verification

The verification set covers pure connection decisions, non-overlapping lane allocation,
same-producer multi-input wiring, imported-binding validation, compiler behavior, complete
decision output, missing evidence, inclusive bounds, identity ambiguity, direction
mismatch, unknown parents, duplicate evidence, non-finite values, the default pipeline, and
the existing Parallel-versus-Serial runtime semantics. Final suite and browser-render
results are recorded in the release handoff after all Round 6 files settle.
