# MolCascade

MolCascade is a local-first Python package for building and running modular,
hierarchical molecular-screening cascades. It starts from an existing molecular library,
applies explicit and replaceable screening components, and produces an auditable shortlist.

MolCascade does not generate molecules and does not run docking, MD, or MMPBSA. Those are
deliberate handoff boundaries for downstream software such as PRISM.

> Status: early alpha (`0.1.0a0`). Contracts are versioned, but public compatibility is not
> guaranteed before the first stable release.

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install .
molcascade generate config --output molcascade-builder.html
```

Open `molcascade-builder.html` in a modern browser. The builder is one offline file: it
loads no CDN resources, sends no network requests, and does not execute screening code.

You assemble a cascade by dragging blocks from the left tray into tiers. Each tier is one
level of the funnel and can hold several blocks, combined in series or put to a vote. Every
block names both the *question* it answers and the *tool* answering it, so swapping RDKit
for `medchem`, `rd_filters`, or your own model is a change of block, not a rebuild. Tools
whose package or weights are not installed here are shown greyed out, with the exact command
that would provide them.

Download the configuration from the builder, then screen a library with it:

```bash
molcascade validate cascade.json
molcascade screen --config cascade.json --library molecules.csv \
  --workspace .molcascade --run-id screen-001
```

A configuration is a reusable screening *policy*; the library is supplied per run, so one
cascade can screen many generated batches without being edited. The run prints its funnel —
what each tier removed, and whether the shortlist reached the size the cascade was
configured for.

Inspect or resume the durable run and materialize its outputs:

```bash
molcascade status screen-001 --workspace .molcascade --events
molcascade screen --config cascade.json --library molecules.csv \
  --workspace .molcascade --run-id screen-001 --resume
molcascade report screen-001 --workspace .molcascade --output screen-001.html
molcascade export screen-001 --workspace .molcascade --output shortlist
```

### External weights and rule tables

Some tools need data files that are not shipped in the package: they are large, and they
carry their own licences. They are fetched once, into a local folder, and verified against a
pinned SHA-256:

```bash
molcascade assets status          # what is needed, what is present
molcascade assets fetch --all     # download and verify everything missing
```

Downloads only ever happen here. A screening run never reaches the network, and it checks
every asset it will need *before* reading the first molecule rather than failing halfway
down the funnel. `molcascade doctor` reports the same readiness for backends and hardware.

### Authoring a flat pipeline directly

`molcascade generate config --pipeline` emits the older node-and-port graph editor, which
writes a schema-1 pipeline instead of a cascade. It exposes stage wiring the tier builder
deliberately hides — explicit port-to-port connections, side evidence channels, and manual
policy joins — and is the right surface when you need a graph the funnel shape cannot
express. `molcascade run` executes what it produces.

## Input libraries

`molcascade screen --library` supplies the molecules for one run, replacing any source path
stored in the configuration. (`molcascade run` spells the same thing `--molecules`, since a
flat pipeline names its source stage explicitly.) Automatic format detection supports:

| Input | CLI format | Large-library behavior |
|---|---|---|
| CSV, TSV, or `.smi` | `delimited` | streamed in bounded batches |
| XLSX | `xlsx` | read-only worksheet streaming |
| SDF | `sdf` | supplier-based iteration |
| Raw molecule Parquet | `parquet` | Arrow row-group batches |
| A directory containing many MOL2 files | `mol2-directory` | disk-backed SQLite inventory; files opened one at a time |

Pass `--format` when a path is ambiguous. Multi-million-file MOL2 collections
should live on a fast local filesystem; the inventory avoids holding every pathname or
molecule in Python memory, but filesystem traversal and structure parsing remain real
costs.

### Reading a library laid out for a human

A configuration is a reusable screening policy; the library changes every run, and so does
its layout. These flags describe the file without editing the configuration:

| Flag | Use |
|---|---|
| `--smiles-column` | Name of the structure column. Matched ignoring case, so `SMILES` and `smiles` both resolve; two spellings in one header are refused rather than guessed. |
| `--id-column` | Column (or SD property) naming each molecule. Carried through to the exported shortlist. |
| `--sheet` | Worksheet to read. Without it the first *visible* sheet is used, which in a curated workbook is often a README. |
| `--skip-rows` | Leading rows to discard before the header, for exports that open with a title or a merged section banner. |

```bash
molcascade screen --config cascade.json --library binders.xlsx \
  --sheet 2_Master_Compounds --skip-rows 1 --id-column Compound_Name \
  --workspace .molcascade
```

`--id-column` is what makes the shortlist joinable back to the library it came from. When
it is supplied, `molcascade export` adds a `source_id` column to the SMILES output and a
`MOLCASCADE_SOURCE_ID` property to the SDF output; without it the output is unchanged.
Standardization deduplicates before export, so a structure that appeared under several
names is exported once, under one of them.

## Screening model

Plugins exchange immutable, typed Arrow/Parquet datasets. They never exchange pickled
toolkit objects. Every stage has a pinned plugin identity, exact input/output contracts,
validated configuration, determinism declaration, and content-addressed artifact record.

The shipped default cascade demonstrates:

1. streamed ingestion;
2. parent registration, normalization, and exact deduplication;
3. parallel chemistry gates for structural validity, physicochemical ranges, Rule of Five
   and QED, plus PAINS/BRENK/NIH/ZINC alerts;
4. an executable **ALL-required** policy join;
5. physicochemical descriptors and molecular fingerprints;
6. RDKit synthetic-accessibility evidence followed by an explicit, configurable score
   threshold gate, joined by a reaction-trained SCScore arm once its weights are fetched;
7. Murcko scaffolds and diversity groups;
8. deterministic scaffold-aware budget selection; and
9. a verified SMILES shortlist handoff.

Prediction and synthesis modules intentionally separate evidence from policy. For example,
ADMET-AI emits endpoint/model prediction rows; a prediction threshold gate turns one
reviewed endpoint into a complete PASS/REJECT decision stream. Missing evidence is rejected,
not silently accepted. The same pattern applies to synthetic-accessibility scores.

### Parallel and Serial criteria

Several criteria may occupy one tier.

- **Parallel (ALL)** gives every criterion the same parent population. Their complete
  decision streams converge at one policy-join node, and a parent survives only when all
  required criteria pass.
- **Serial** sends the survivors of one criterion into the next criterion in that tier.

These modes compile to materially different DAGs and different pipeline revision IDs.
Changing the canvas layout alone is metadata; changing a connection, tool, threshold, or
mode changes the executable revision.

## Local tools and scientific scope

MolCascade currently includes executable adapters for native Python/Arrow components,
RDKit chemistry, the ChEMBL Structure Pipeline, Mordred descriptors, Lilly Medchem Rules
and rd_filters alert sets, SCScore synthetic complexity, FPSim2 similarity search, optional
Open Babel descriptors, an optional ADMET-AI v2 endpoint panel, and two bring-your-own-model
paths. The builder and `molcascade doctor` distinguish three states:

- **available executable adapter** — registered and runnable under current local policy;
- **optional local adapter** — implemented but unavailable until its dependency or trusted
  assets are present;
- **research option only** — reviewed software with no MolCascade adapter yet; visible for
  planning but impossible to export as an executable node.

```bash
molcascade plugins
molcascade doctor
molcascade doctor --run-version-commands
```

The backend catalogue covers alternatives such as the ChEMBL Structure Pipeline, Open
Babel, Lilly Medchem Rules, MordredCommunity, Chemprop, ADMET-AI, FPSim2, RAscore,
AiZynthFinder, scaffold tooling, and multi-objective selection libraries. `molcascade
doctor` reports which of them can run on this machine and what each missing one needs;
`molcascade cite` prints the whole catalogue with the paper behind every entry.

There is no universal “SOTA” tool, threshold, or screening order across targets. Rule of
Five, QED, structural alerts, ADMET models, similarity, SA score, and retrosynthesis answer
different questions and have different domains. The default synthesizability cutoffs are
transparent starter policy, not evidence of route feasibility: SA score counts how common a
molecule's fragments are and SCScore learns complexity from reaction records, so they
disagree by construction and the default drops a molecule only when both call it hard.
Neither is a route or a step count. PAINS matches are triage signals, not experimental proof. Endpoint direction, units, calibration, applicability, and project
costs must be reviewed before a model score becomes a rejection rule.

The ADMET-AI adapter is deliberately strict: it does not download models or install
packages. Before enabling it, inspect and pin both the installed package code and complete
model tree, review the configured endpoints, and explicitly acknowledge trusted local
Torch checkpoint deserialization — a checkpoint runs code when it loads, and pinning its
digest records which bytes ran rather than making running them safe.

### Your own model

A model you trained on your own data can be one tier of a cascade. Put it in a directory
with a manifest, take its digest, and select it in the builder like any other engine:

```bash
molcascade model-bundle /path/to/bundle
```

Two formats exist because two kinds of model do. An **ONNX bundle** takes a fixed-width
vector that MolCascade computes, which covers anything exportable with `skl2onnx`. A
**Chemprop bundle** takes SMILES and lets Chemprop featurize the molecular graph itself,
which is the part an ONNX bundle cannot express. Loading a Chemprop checkpoint executes
code, so it additionally requires an explicit trust acknowledgement — a digest says which
bytes ran, not that running them is safe.

`molcascade model-bundle` is also the reference for the manifest itself: it validates every
field and names what is wrong, so pointing it at an incomplete directory is the fastest way
to learn the format. A multitask model must say which output column it means with
`task_index`, and a molecule the model cannot score is reported as unscoreable rather than
quietly dropped.

## Vendored assets

Some backends read published model weights or rule tables that MolCascade uses but does not
author. Those files are declared in a catalogue with their SHA-256 digests, sizes, upstream
URLs, licences, and the paper to cite, and they are fetched by an explicit command:

```bash
molcascade assets list            # what is declared
molcascade assets status          # what is here and whether it verifies
molcascade assets fetch --all     # download everything missing
molcascade assets fetch scscore   # or just one
molcascade assets where 'asset:scscore/standalone_model_numpy.py'   # reference -> real path
```

Only `fetch` touches the network. **A screening run never downloads.** A run that needs an
absent asset stops and prints the `fetch` command that would supply it, so results depend on
files whose digests are recorded rather than on what a remote host served that afternoon.

Refer to a file from a cascade with `asset:<asset-id>/<path>` rather than an absolute path.
The reference resolves to the same content on every machine, and resolution verifies the
digest before handing the path to a backend.

Each asset also declares whether loading its payload executes code. A digest proves a file is
the one its author published; it does not make deserializing it safe. Gzipped JSON weights
that MolCascade parses into numbers itself and a Torch checkpoint that runs arbitrary code on
load are both pinned, and the catalogue keeps them clearly distinct.

Assets live in `vendor/` in a source checkout and in the user data directory from an installed
wheel. `MOLCASCADE_ASSET_ROOT` overrides both. See [`vendor/MANIFEST.md`](vendor/MANIFEST.md).

## What to cite

A screen ends in a methods section, and the methods section has to name the tools that decided
which molecules survived — not MolCascade, which orchestrates them.

```bash
molcascade cite                         # print every reference
molcascade cite --output citations.md   # write a copy to drop beside a manuscript
```

The output covers every backend the builder can place in a cascade and every vendored weight
file, with its licence, its pip requirement, and its paper. It is generated from the
catalogue rather than maintained by hand, and a test fails if any option is added without a
reference — so the bibliography cannot quietly fall behind the tools.

## Reproducibility and safety

- Raw source records are never overwritten.
- Parent identity is established once and is distinct from downstream simulation states.
- Hard rejects cannot be rescued by later scores.
- Stage outputs are immutable, content-verified artifacts.
- A run records durable status and machine-readable audit events.
- Cache reuse and resume verify artifact identities before accepting prior work.
- Vendored weights and rule tables are pinned by digest and verified before a backend sees
  them; a run stops on an absent or altered asset rather than fetching one.
- Duplicate YAML keys, unknown configuration fields, unknown plugins, contract mismatches,
  cycles, path escapes, and untrusted executable adapters fail closed.
- Assets a run will need are resolved before the first molecule is read, so a missing weight
  file costs nothing instead of surfacing halfway down the funnel.
- No package manager, model downloader, docking engine, or remote service is invoked by the
  HTML builder.

## Development

```bash
python -m pip install -e '.[analytics,dev]'
pytest -q
python -m ruff check src tests
python -m mypy
python -m build
```

Linux or WSL2 is recommended for large production workers and optional native backends.
The generated HTML is intended to remain portable across Windows, Linux, and macOS
browsers.
