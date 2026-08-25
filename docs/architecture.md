# MolCascade architecture v0.1

## Product boundary

MolCascade starts with a completed local molecular library. It accepts delimited
SMILES, XLSX/XLSM, SDF, raw-contract Parquet, or a directory containing many MOL2
files. It does not generate molecules and it does not execute docking, MD, MMPBSA, or
other target-structure simulations. A verified shortlist is the terminal MolCascade
artifact and can be handed to PRISM or another downstream application later.

No PRISM module is imported by MolCascade. This separation keeps screening revisions,
simulation workflows, environments, and failure domains independently reproducible.

## Configuration graph and execution order

The user edits a directed acyclic graph (DAG) in a self-contained offline HTML builder.
The file is a configuration editor, not an execution engine: it exports strict JSON or
YAML, and the Python CLI validates and runs that configuration locally.

Stages remain listed in topological order so a configuration can be reviewed as text.
Each input binding can retain a named output port from any earlier enabled stage. This
allows real fan-out and fan-in without copying side evidence through every stage:

```text
registered parents
       |
       +--> property gate -------- decisions --+
       +--> Rule of Five + QED ---- decisions --+--> ALL policy join --> survivors
       +--> structural alerts ----- decisions --+
```

The UI's two within-tier modes have data-plane meaning:

- **Parallel (ALL):** sibling criteria read the same parent artifact. Their complete
  decision artifacts are combined by an explicit `policy.native_decision_join` barrier.
  The survivors are the intersection; no sibling can hide evidence from another by
  filtering it first.
- **Serial:** the first criterion's retained-parent output feeds the next criterion.
  Each criterion therefore sees only the survivors of its predecessor.

An explicit join is itself a versioned plugin with an auditable policy. A hard
`REJECT` is never recoverable by a later score or ranking component.

## Replaceable scientific components

A pipeline stage binds a stable capability slot to an exact `plugin@version`. Plugins
declare exact input/output contracts and named output ports. The compiler rejects an
untrusted plugin, an incompatible slot, a forward edge, a missing port, or a contract
mismatch before execution.

The backend catalogue is deliberately broader than the executable plugin registry:

- an **executable adapter** has a reviewed `plugin_ref` and may be placed on the canvas;
- a **local optional dependency** is executable only after the user provisions and
  explicitly configures its local package/assets;
- a **research option** is informative and disabled in the canvas until an adapter is
  implemented and tested.

Backend recommendation, local availability, and adapter status are independent fields.
The package never presents an importable library as if it were automatically usable in
an executable MolCascade graph.

## Control plane and data plane

Pydantic models validate small control-plane objects: pipeline configurations, plugin
descriptors, artifact manifests, run state, and structured errors. Duplicate YAML keys,
unknown fields, non-finite values, and ambiguous identifiers are rejected.

Versioned PyArrow schemas and Parquet datasets form the data plane. Molecular rows are
processed in bounded batches rather than through per-row Pydantic models. Toolkit-native
molecule objects and pickle files never cross a public artifact boundary.

## Identity and evidence

Raw source structure text is immutable. Standardization creates registered parent
identities under an explicit identity-policy ID and retains a source-to-parent map.
Every calculator, model, rule set, reference collection, and selection policy records a
method identity derived from its configuration, software version, and relevant asset
hashes.

Properties, predictions, applicability, alerts, synthesis proxies, scaffolds, clusters,
and decisions remain separate typed evidence. In particular:

- Rule of Five is not QED;
- an ADMET prediction is not an applicability-domain result;
- PAINS/reactivity alerts are triage hypotheses, not experimental stability data;
- SA score is a proxy, not a route or retrosynthesis step count;
- similarity-to-leads and novelty are opposing project objectives, not one universal
  quality score.

## Artifact and runtime rules

Plugins receive immutable artifact references and may write only to a runner-assigned
staging directory. Before atomic publication the runner validates declared files,
relative paths, schemas, global keys/counts where required, checksums, and lineage.

- Cache keys identify a complete invocation, including request-port-qualified inputs.
- Artifact IDs identify verified output content and lineage.
- A directory without a complete `artifact.json` is not a committed artifact.
- Resume verifies run state, revision identity, input lineage, and artifact integrity.
- Source stages are not cache-reused because external input paths may be mutable.
- A worker/plugin boundary is fault isolation, not a security sandbox.

Large inputs are streamed with explicit record/file/buffer limits. MOL2 directory
inventory is stored in SQLite instead of a Python list. Exact lead similarity has
reference-count and total-comparison budgets; larger reference sets should use a future
indexed adapter such as FPSim2.

## Offline and trust boundary

Normal configuration and execution perform no install or download. Optional ML and
retrosynthesis backends must be provisioned separately, and local model/checkpoint,
stock, rule, and configuration assets must be pinned by cryptographic digest. Model
formats backed by Python or Torch deserialization require an explicit trust decision;
hashing an untrusted file makes it reproducible, not safe.

Published weights and rule tables are declared in an asset catalogue that performs no I/O.
Each entry fixes its files' SHA-256 digests, sizes, and upstream URLs, and separately
records whether loading the payload executes foreign code — a distinction a digest cannot
make. Acquisition and use are different commands with different permissions: only
`molcascade assets fetch` may reach the network, and a screening run may only locate,
verify, and read. A run that needs an absent asset stops and names the command that would
supply it.

Configurations name assets by `asset:<id>/<path>` reference rather than absolute path, so a
configuration means the same thing on every machine; resolution verifies the digest before
the path reaches a backend. A verification stamp records size, mtime, and digest so an
unchanged tree is not rehashed per run, and the stamp is treated as evidence rather than
authority — a file whose size or mtime moved is hashed again. Member paths that escape the
asset directory, symlinks, and non-regular files are refused regardless of their contents.

Path-race protections are strongest on Linux/WSL where descriptor-relative and
`O_NOFOLLOW` operations are available. Windows reparse-point handling is best effort;
production runs should use a trusted local workspace and WSL2 when strict path semantics
are required.
