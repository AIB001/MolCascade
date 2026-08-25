# Open-source backend landscape for a modular screening cascade

Review date: 2026-08-23  
Audience: MolCascade developers and scientific workflow owners  
Scope: local, pre-docking screening from molecule ingestion through shortlist export

## Technical summary

MolCascade should not expose a single opaque "drug score." The defensible design is a
typed graph in which inexpensive identity and chemistry checks run first, independent
criteria can fan out over the same parent population, an explicit policy node combines
their decisions, and expensive predictive or route-search tools operate only on the
surviving pool. The current package implements that graph model and already has
executable modules for streaming input, RDKit standardization and chemistry evidence,
an ALL/ANY/minimum-pass decision join, exact lead-reference similarity, SA Score,
scaffold/diversity selection, and export. An ADMET-AI v2 adapter is also registered but
requires separately provisioned, explicitly trusted local package and model assets.

Open-source alternatives exist at every scientific layer, but a catalogue entry or an
installed Python module is not the same thing as an executable MolCascade adapter.
This review therefore uses three implementation states:

- **Integrated executable (`I`)**: a versioned MolCascade plugin is registered and can
  be compiled into a pipeline. Core-package dependencies are installed by the normal
  MolCascade environment.
- **Local optional adapter (`O`)**: a versioned MolCascade plugin is registered, but
  execution requires a separately installed backend, model, or a license-policy
  opt-in. Availability must be probed and missing assets must fail closed.
- **Researched only (`R`)**: the project has been reviewed as an option, but there is
  no registered MolCascade plugin. Detection by `doctor` is informational and does not
  make it selectable or executable.

These states are deliberately separate from the backend catalogue's `default`,
`fallback`, `optional`, and `isolated` deployment tiers. A deployment tier expresses a
preference or isolation boundary; only a non-null, registered `plugin_ref` establishes
an executable integration.

No backend in this report is called universally state of the art. Predictive ranking
depends on the endpoint, assay definition, training set, scaffold/time split,
calibration, applicability domain, and exact model assets. Route-planning performance
also depends on reaction policy, stock, search algorithm, and budget. MolCascade should
let projects compare pinned candidates against their own validation set rather than
encode a permanent winner in the UI.

## Recommended graph and screening order

The default order is a cost- and evidence-aware engineering recommendation, not a
claim that the literature defines one universal cascade:

1. **Ingest, standardize, register, and deduplicate.** Freeze raw source identity and
   the parent/tautomer/stereo policy before computing evidence.
2. **Reject terminal structural failures.** Invalid valence, unsupported chemistry,
   mixtures forbidden by project policy, or explicit severe-reactivity rules should
   not reach expensive models.
3. **Run cheap criteria in parallel on the same parents.** Property windows,
   Lipinski Rule of Five, QED, and PAINS/BRENK/NIH/ZINC evidence are sibling branches.
   Their decisions converge through an explicit ALL/ANY/minimum-pass policy.
4. **Compute descriptors, fingerprints, endpoint models, and applicability evidence.**
   ADMET scores, uncertainty, calibration, and domain membership remain separate typed
   facts. A score outside the model domain is not automatically a negative result.
5. **Evaluate lead-space objectives.** Maximum similarity to a frozen lead reference
   can support an analogue floor, a novelty ceiling, or separate exploitation and
   exploration baskets. The reference snapshot, fingerprint, and metric are part of
   the method identity.
6. **Evaluate synthesis in two costs.** Run SA/RAscore-like proxies broadly; reserve
   route search and retrosynthesis step counts for a much smaller pool.
7. **Apply scaffold/diversity and budget selection.** Choose representatives after
   scientific evidence exists, using deterministic tie-breaking and no dense
   all-pairs matrix at million-molecule scale.
8. **Export a typed shortlist.** MolCascade stops before docking, MD, or MMPBSA.

Within one UI tier, **Parallel (ALL)** means every criterion receives the same upstream
parent population and a policy join retains only parents with complete, non-rejecting
evidence from every branch. **Serial** means criterion B receives only criterion A's
survivors. The two layouts can have the same final intersection for deterministic hard
gates, but they differ in work performed and in the evidence retained for early
rejects. Parallel is usually preferable for cheap sibling evidence; serial is useful
when an early inexpensive gate materially reduces the cost of a later model.

## Capability and implementation matrix

| Layer / user objective | Local open-source option | State | Appropriate role | Important boundary |
|---|---|---:|---|---|
| CSV/TSV/SMI ingestion | Python binary framing + CSV parser | I | Byte-bounded streaming with stable record provenance | A row and expanded-buffer limit is still not an operating-system RSS limit |
| XLSX ingestion | openpyxl read-only worksheet iterator | I | Friendly moderate-size spreadsheet input | One worksheet is limited to 1,048,576 rows; use CSV/Parquet for multi-million inputs |
| SDF / MOL2 ingestion | RDKit SDF and MOL2 parsers | I | Local structure-file input; MOL2 directory uses a disk-backed inventory | RDKit documents MOL2 typing limitations; malformed records must remain explicit rejects |
| Existing columnar data | PyArrow Parquet scanner | I | Preferred repeat-run and large-library interchange | Only the exact MolCascade raw contract is accepted; arbitrary schemas are not guessed |
| Standardization / registration | RDKit MolStandardize + RegistrationHash policy | I | Default parent identity and normalized representation | Any rule/version change creates a new identity policy; toolkit canonical forms are not assumed equivalent |
| Alternative standardization | ChEMBL Structure Pipeline | R | Strong candidate for ChEMBL-compatible standardization and structure checking | Its output must be re-registered under an explicit identity policy; adapter and conformance corpus are pending |
| Format rescue / descriptors | Open Babel | O for property descriptors; R for rescue | Optional isolated conversion or descriptor cross-check | GPL policy is blocked by default; rescue output must return through registration; never switch silently |
| Structural validity / project SMARTS | RDKit validation, SMARTS, and FilterCatalog | I | Cheap deterministic hard gate with versioned reason codes | Rule lists and toolkit version are scientific assets, not UI labels |
| Medicinal-chemistry triage | RDKit PAINS/BRENK/NIH/ZINC | I | Parallel alert evidence with per-catalog ignore/warn/reject policy | Alerts are hypotheses, not experimental proof of instability or assay interference |
| Alternative medchem triage | Lilly Medchem Rules; `rd_filters` | R | Independent rule-set cross-check or project-specific gate | Exact rule revision and severity policy must be pinned; Lilly is best isolated behind a CLI adapter |
| Lipinski / QED | RDKit Rule of Five and QED components | I | Cheap oral-drug-likeness evidence; thresholds remain independent | QED and Ro5 are heuristics, not efficacy, safety, or bioavailability measurements |
| Physicochemical descriptors | RDKit descriptors | I | Default bounded 2D property table | Descriptor names, units, toolkit version, and null policy belong in the method ID |
| Alternative descriptors | Open Babel properties | O | Locally executable second implementation | Different calculators need different method IDs; do not compare columns by name alone |
| Broad descriptor exploration | MordredCommunity | R | Large 2D/3D feature panel for model development | Select features before production; thousands of descriptors amplify missingness and compute cost |
| Fingerprints | RDKit Morgan / AtomPair packed fingerprints | I | Local similarity, applicability, and clustering primitives | Radius, length, chirality, feature mode, and RDKit version must be explicit |
| Alternative fingerprint / AD library | scikit-fingerprints | R | Consistent sklearn-style fingerprints, filters, similarities, and applicability-domain methods | Many methods still use RDKit; the value is composability, not toolkit independence |
| ADMET pretrained panel | ADMET-AI v2 | O | Multi-endpoint local prediction over explicitly provisioned checkpoints | Registered adapter emits `prediction/v1`; it does not claim uncertainty, calibration, applicability, or experimental truth |
| Project-trained molecular model | Chemprop 2.x | R | Local checkpoint inference with ensemble/dropout/MVE/evidential uncertainty options | Adapter must pin checkpoint, feature schema, task units, calibration, split provenance, and runtime; model choice is endpoint-specific |
| Reproducible ADMET model development | OpenADMET Models | R | Framework for training, evaluation, featurization, splits, and uncertainty | It is a development framework, not a universal bundled pretrained endpoint service |
| 3D representation / property model | Uni-Mol Tools | R | Optional GPU-oriented representation or project-trained predictor | Official tooling can auto-download weights when paths are absent; a MolCascade adapter must require local hash-pinned assets and disable that behavior |
| Classical project model | scikit-learn; XGBoost; LightGBM | R | Strong, interpretable baselines on frozen descriptors/fingerprints | Local installation does not imply a model; the trained estimator and preprocessing pipeline must be safe, versioned assets |
| Model applicability | Exact RDKit reference similarity | I | Maximum similarity to a bounded frozen lead/training reference set | Outputs evidence only; reference snapshot + fingerprint + Tanimoto definition are mandatory |
| Large indexed similarity | FPSim2 | R | Fast threshold search against larger reference collections, in memory or on disk | It is a neighbor engine, not a clustering policy; index and RDKit fingerprint identity must be pinned |
| Synthetic-accessibility proxy | RDKit Contrib SA Score | I | Fast fragment/complexity ranking on a broad pool | Explicitly not a route, step count, stock check, or proof of synthesis |
| Retrosynthesis-likelihood proxy | RAscore | R | Cheap probability-like proxy for whether AiZynthFinder may solve a route | Learned from a particular planner/data regime and documented as limited in chemical-space coverage; not route evidence |
| Route search and step count | AiZynthFinder | R | Local MCTS-style retrosynthesis to purchasable precursors on a reduced shortlist | Must pin policy model, stock, search config, package, timeout, solved status, and route; step count exists only for a returned route |
| Alternative route framework | Syntheseus | R | Standardized model/search wrappers and retrosynthesis benchmarking | It is a framework with several algorithms/models, not one scientifically interchangeable "Syntheseus score" |
| Scaffold evidence | RDKit Murcko and scaffold grouping | I | Deterministic grouping and scaffold-aware selection | Acyclic molecules need an explicit policy rather than one undifferentiated bucket |
| Scaffold hierarchy | ScaffoldGraph | R | Scaffold networks and trees on a reduced population | Graph size can dominate memory; use after earlier reduction |
| Diversity selection | Native packed-bit streaming leader; scaffold round-robin; hash budget | I | Bounded-memory representative selection and reproducible quotas | Streaming leader is O(candidates x leaders), so cap leaders/comparisons for very large pools |
| Alternative diversity engines | RDKit Leader/MaxMin; BitBIRCH/BitBIRCH-Lean | R | Picker cross-checks or high-throughput clustering | Avoid RDKit Butina for millions because dense all-pairs distances are unsafe; BitBIRCH variants need license/platform review and isolated adapters |
| Multi-objective shortlist | Native deterministic selectors | I | Explicit baskets, quotas, scaffold caps, and stable ties | Hard rejects cannot be recovered by a later weighted rank |
| Pareto utilities | paretoset; pymoo | R | Nondominated-set extraction or broader multi-objective algorithms | A Pareto library does not define objectives, missing-value policy, or budget backfill for the project |
| Delivery | Native Parquet/SMILES manifest; RDKit SDF | I | Auditable, local shortlist export | Docking and molecular simulation remain out of scope |

The table describes MolCascade integration as of this review, not whether a package
happens to be importable on one workstation. The local `doctor` probe on 2026-08-23
found the core Python/RDKit/Arrow/openpyxl/native options available; ADMET-AI,
Chemprop, ChEMBL Structure Pipeline, FPSim2, MordredCommunity, ScaffoldGraph,
AiZynthFinder, and RAscore were not installed. Copyleft Open Babel and BitBIRCH entries
were blocked under the default license policy. Those observations can change and must
not be baked into an exported scientific configuration.

## Metric placement and policy semantics

### Rule of Five and QED

Lipinski violations and QED should sit in the cheap evidence tier. They can be
calculated in one component because they share descriptors, but their thresholds are
independent. A project may allow one Ro5 violation while requiring a QED floor, or may
retain both as annotations. QED itself combines desirability transforms over molecular
weight, lipophilicity, H-bond donors/acceptors, polar surface area, rotatable bonds,
aromatic rings, and alerts; it should not be described as a replacement for ADMET.

### PAINS, reactive motifs, and chemical stability

PAINS and related catalog matches belong beside other cheap structural evidence and
normally start as warnings. Lilly Medchem Rules and versioned project SMARTS can add
orthogonal motif checks. None of these establishes shelf stability, plasma stability,
metabolic stability, hydrolysis rate, or photostability. "Chemical stability" must be
split into a named endpoint backed by a suitable assay or a model validated for that
endpoint; structural alerts may be retained as separate mechanistic hypotheses.

### ADMET and applicability

Each ADMET endpoint should emit a score with an endpoint ID, units/label semantics,
model/checkpoint identity, and calibration metadata. Applicability-domain evidence is
a parallel side output, not a hidden multiplier. ADMET-AI v2 is useful as an optional
pretrained panel, but its official repository notes that v2 was retrained with
Chemprop v2 and does not exactly reproduce the v1 paper's predictions. Chemprop is the
better abstraction when the project owns endpoint-specific checkpoints and needs
uncertainty methods. OpenADMET is attractive for reproducible model development and
comparison. Uni-Mol Tools is a candidate for 3D representation/property work, subject
to a strict local-asset adapter because its documented default may fetch pretrained
weights automatically.

### Lead similarity, uniqueness, and diversity

"Maximum similarity to leads" and "uniqueness" are opposing policies over the same
evidence. Compute the nearest reference and maximum Tanimoto similarity once, then let
the graph define an analogue minimum, novelty maximum, or two selection baskets.
Exact RDKit comparison is suitable for a bounded curated lead set. FPSim2 is the
stronger next adapter for larger indexed reference sets, but it supplies neighbors;
MolCascade still owns threshold, cluster, and selection semantics. Diversity within
the candidate pool should use scaffold grouping or sparse/streaming methods and must
not allocate an N x N similarity matrix for millions of molecules.

### Synthesizability and retrosynthesis steps

SA Score and RAscore-like models are prioritization proxies. They can rank a broad pool
but cannot populate `route_solved` or `retrosynthesis_steps`. Only a route planner such
as AiZynthFinder or a configured Syntheseus model/search combination can produce route
evidence, and even then the result is conditional on exact stock, policy, search
budget, and timeout. Route search should therefore follow strong chemistry and model
reduction, with timeout and unsolved kept distinct from an invalid molecule.

## Adapter acceptance contract

A researched backend should move to `O` or `I` only after it satisfies all of the
following:

1. A versioned plugin declares exact input and named output contracts. No filename or
   column guessing occurs at the plugin boundary.
2. Backend package, rules, model/checkpoint, reference set, reaction policy, stock,
   and feature schema are hashed where they affect results.
3. Execution never installs packages or downloads assets. Tools that auto-fetch by
   default must receive explicit local paths, and the adapter must test that path.
4. Untrusted Python packages and Torch/pickle-like checkpoints are treated as
   executable code. Hashing supplies provenance; it is not a sandbox.
5. Input and output are processed in bounded batches or through a disk-backed index.
   Limits are explicit, and partial outputs are removed after failure.
6. Bad molecules yield structured decisions where scientifically possible; global
   corruption or a model/schema mismatch fails the stage closed.
7. Determinism is stated honestly. GPU/Torch models should generally be best-effort;
   deterministic and seeded plugins must include every result-affecting field in the
   cache identity.
8. A sentinel chemistry suite, schema/cardinality conservation, asset mutation,
   offline behavior, Windows/WSL paths, peak RSS, and reduced/full-scale performance
   are validated before the module is advertised as production-ready.

## Recommended next adapters

The most useful sequence after the current release is:

1. **ChEMBL Structure Pipeline standardizer/checker** with explicit re-registration
   and a differential corpus against the current identity policy.
2. **FPSim2 indexed reference similarity** with immutable index metadata, query
   fingerprint validation, and exact-small-set cross-checks against RDKit.
3. **Chemprop project-checkpoint predictor** with safe local asset policy, typed
   endpoint units, uncertainty/calibration fields, and scaffold/time-split provenance.
4. **AiZynthFinder route-search adapter** in an isolated process, accepting pinned
   policy and stock assets and emitting route/steps/timeout/unsolved as distinct facts.
5. **RAscore** only as a separately named proxy, followed by **Syntheseus** when a
   second search/model framework is needed for comparison.
6. **OpenADMET** and **Uni-Mol Tools** after a project supplies a concrete endpoint,
   validation set, and pinned local assets. A generic "AI model" node would be too
   scientifically ambiguous.

## Primary and official sources

- [RDKit 2026 MolStandardize API](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html)
- [ChEMBL Structure Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline)
- [Open Babel command-line documentation](https://openbabel.org/docs/Command-line_tools/babel.html)
- [Lilly Medchem Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules)
- [MordredCommunity](https://github.com/JacksonBurns/mordred-community)
- [scikit-fingerprints](https://github.com/MLCIL/scikit-fingerprints)
- [ADMET-AI repository and v2 notes](https://github.com/swansonk14/admet_ai)
- [Chemprop local prediction and uncertainty documentation](https://chemprop.readthedocs.io/en/latest/tutorial/cli/predict.html)
- [OpenADMET Models documentation](https://docs.openadmet.org/en/latest/)
- [OpenADMET Models repository](https://github.com/OpenADMET/openadmet-models)
- [Uni-Mol Tools](https://github.com/deepmodeling/unimol_tools)
- [AiZynthFinder documentation](https://molecularai.github.io/aizynthfinder/)
- [AiZynthFinder repository](https://github.com/MolecularAI/aizynthfinder)
- [RAscore repository](https://github.com/reymond-group/RAscore)
- [Syntheseus repository](https://github.com/microsoft/syntheseus)
- [RDKit SA Score implementation](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score)
- [FPSim2 repository](https://github.com/chembl/FPSim2)
- [ScaffoldGraph](https://github.com/UCLCheminformatics/ScaffoldGraph)
- [RDKit diversity pickers](https://rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html)
- [BitBIRCH](https://github.com/mqcomplab/bitbirch) and [BitBIRCH-Lean](https://github.com/mqcomplab/bblean)
- [paretoset](https://github.com/tommyod/paretoset) and [pymoo](https://github.com/anyoptimization/pymoo)

