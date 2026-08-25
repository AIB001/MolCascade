# Round 4: Multi-metric screening components

Status: implementation and targeted verification complete (2026-08-23)

## Goal

This round turns three commonly conflated concepts into auditable local components:

1. oral drug-likeness evidence (Lipinski Rule of Five and QED);
2. structural-alert evidence and project policy (PAINS/BRENK/NIH/ZINC);
3. synthetic-accessibility proxy evidence (RDKit Contrib SA_Score).

All user-facing names, descriptions, errors, configuration fields, reason codes, and
documentation added in this round are English. No component performs docking, MD, or
MMPBSA, and none requires a network connection during execution.

## Implemented components

| Plugin reference | Role | Default behavior | Outputs |
|---|---|---|---|
| `chemistry.rdkit_drug_likeness@0.1.0` | Rule-of-Five and weighted QED | Annotate every molecule; no threshold is assumed | retained `parent/v1`, typed `drug_likeness/v1`, `decision/v1` |
| `chemistry.rdkit_structural_alerts@0.1.0` | RDKit PAINS/BRENK/NIH/ZINC catalog matches | PAINS=`WARN`; BRENK/NIH/ZINC=`IGNORE` | retained `parent/v1`, `decision/v1` |
| `synthesis.rdkit_sa_score@0.1.0` | Ertl-Schuffenhauer fragment/complexity proxy | Score every molecule; higher means harder | unchanged `parent/v1`, `synthesis_score/v1` |

The new `drug_likeness/v1` contract separates `rule_of_five_method_id` from
`qed_method_id` and records `toolkit_version`. Its combined `calculator_id` binds both
methods to the exact implementation. This avoids treating Rule-of-Five as a hidden QED
threshold or treating a changed RDKit release as identical evidence.

### Rule of Five and QED

Rule-of-Five violations use the original strict boundaries: MW > 500 Da, CLogP > 5,
HBD > 5, and HBA > 10. QED remains a separate continuous desirability measure and
also records its eight RDKit components: MW, ALogP, HBA, HBD, PSA, rotatable bonds,
aromatic rings, and QED alerts. RDKit documents that its implementation can differ
slightly from the original QED publication because property calculators differ.

The plugin supports two independent project thresholds in one component:

- `maximum_rule_of_five_violations`;
- `minimum_qed`.

Both are `null` by default. If configured, failures may be `warn` or `reject`. A single
component evaluates these metrics against the same input population, so one metric
does not remove records before the other metric is calculated. Typed evidence is
retained for rejected molecules, and every molecule receives an auditable decision.

### Structural alerts

Each RDKit catalog has an independent `ignore`, `warn`, or `reject` action. PAINS is a
warning by default because a SMARTS match is a triage hypothesis, not experimental
proof of interference or instability. Each matched entry creates a stable reason code
derived from the catalog and alert description. Details explicitly say
`substructure triage flag; not experimental proof`.

Catalog matching is bounded by RDKit's fixed local catalogs. Decision rows are flushed
at a configurable bound, rather than accumulated for an entire million-row source.

### Synthetic accessibility

The synthesis plugin loads RDKit's installed `Contrib/SA_Score/sascorer.py` and
`fpscores.pkl.gz`, hashes both assets, and includes the hashes in `method_id` and run
metadata. If either asset is unavailable, it fails before creating outputs with
`SYNTHESIS_SASCORE_UNAVAILABLE` and an installation hint. It never silently substitutes
a different heuristic.

Every score carries `SYNTHESIS_PROXY_NOT_ROUTE`. SA_Score is not a route search, a
retrosynthesis step count, a building-block availability check, or proof of a viable
synthesis.

## Recommended hierarchy and parallel metric groups

The literature does not define one universal linear order for all projects. The order
below is an engineering inference combining scientific semantics, information loss,
and computational cost. Terminal validity checks occur before expensive models;
independent soft evidence is calculated in parallel before a project policy combines
it.

| Level | Parallel work inside the level | Combination guidance | Why here |
|---|---|---|---|
| 0. Ingest and identity | streaming parse, source audit, standardization, registration, global deduplication | terminal parse/identity failures only | Establish immutable molecule identity before any evidence is joined |
| 1. Structural validity | valence/mixture policy, project forbidden chemistry, severe reactive rules | explicit terminal rules; PAINS remains WARN by default | Cheap and prevents invalid structures reaching models |
| 2. Cheap quality evidence | Rule-of-Five, QED, physicochemical windows, PAINS/BRENK/NIH/ZINC, optional project stability alerts | `all`, `any`, `at_least_k`, or annotate-only; do not serialize sibling metrics | All metrics are cheap and should see the same parent population |
| 3. ADMET and applicability | endpoint models in parallel, uncertainty, per-model applicability domain | endpoint-specific policies; out-of-domain is not automatically a negative prediction | Model evidence is meaningful only with model/version/domain provenance |
| 4. Lead-space objectives | maximum similarity to a frozen lead set, novelty ceiling/floor, optional analogue-series membership | treat proximity and novelty as separate objectives or baskets | “Similar to leads” and “unique” can conflict and must not overwrite each other |
| 5. Synthesis evidence | SA/SC proxy on the broad pool; route-aware retrosynthesis only on a reduced pool | proxies rank or warn; route solved/steps/stock are separate fields | Route search is much more expensive and semantically different from SA |
| 6. Diversity and budget | scaffold assignment, scalable clustering, Pareto/basket ranking, quotas | deterministic tie breaking and explicit backfill | Diversity is most useful after evidence is available, before final export |
| 7. Handoff | SMILES/SDF/Parquet shortlist plus provenance/report | no docking execution | PRISM or another downstream system owns docking/MD/MMPBSA |

### Required graph semantics for the HTML builder

A visual “level” must be a container, not a forced sequence. One parent output fans out
to sibling metric nodes, and an explicit policy node joins their side outputs:

```text
                         +-- Rule of Five + QED --------+
standardized parents ----+-- structural alerts ---------+--> group policy --> next level
                         +-- property windows ----------+
                         +-- ADMET endpoints (optional) -+
```

The group policy must expose at least:

- `annotate_only`: retain all and attach evidence;
- `all`: every enabled hard condition must pass;
- `any`: at least one enabled condition must pass;
- `at_least_k`: useful for endpoint panels;
- weighted rank or Pareto selection: useful when metrics conflict;
- per-metric action: `ignore`, `warn`, or `reject`.

A terminal rejection cannot be rescued by a later weighted score. WARN evidence remains
visible in reports. The UI should visually distinguish evidence nodes, terminal gates,
group-policy joins, and budget selectors.

## Coverage of the requested metrics

| Requested metric | Current executable coverage | Local alternatives / next adapter | Scientific boundary |
|---|---|---|---|
| Lipinski Rule of Five | Implemented in this round | project-specific ranges already exist in `chemistry.rdkit_property_range_gate` | Oral absorption/permeation heuristic, not efficacy |
| QED | Implemented in this round with components | alternate weights can be a later versioned config | Relative desirability, not a universal acceptance threshold |
| ADMET | No pretrained ADMET plugin yet | ADMET-AI v2 for an optional local endpoint panel; Chemprop for project checkpoints; scikit-learn baselines | Every endpoint needs model/checkpoint hash, units, calibration, split provenance, and applicability evidence |
| Chemical stability | No general experimental-stability predictor | versioned project SMARTS and endpoint-specific validated models | PAINS/BRENK/NIH/ZINC matches must not be relabeled as measured stability |
| Synthesizability | Typed RDKit SA_Score implemented | SCScore can be a separate proxy adapter | Proxy ranking is not route feasibility |
| Retrosynthesis steps | Not implemented | optional isolated AiZynthFinder adapter after strong prefiltering | Store solved status, route/stock hashes, step count, timeout, and failure separately |
| PAINS/RDKit filters | PAINS/BRENK/NIH/ZINC implemented; existing hard-gate rules remain available | Lilly Medchem Rules or versioned project SMARTS | PAINS defaults to WARN and requires orthogonal follow-up |
| Maximum lead similarity | Fingerprint and clustering primitives exist, but no reference-set maximum plugin yet | RDKit for small frozen lead sets; FPSim2 for indexed large references | Reference snapshot and fingerprint specification are part of method identity |
| Uniqueness/diversity | Murcko, scaffold grouping, streaming leader, hash budget, scaffold round-robin exist | FPSim2 sparse neighbor engine / BitBIRCH for larger pools | Never construct a dense N x N matrix for million-scale libraries |

ADMET-AI is a promising optional adapter because its official project supports local
CLI/Python prediction, CPU or GPU execution, and multiple endpoints. It should not be
silently installed or downloaded during a run. The adapter must freeze the exact model
assets and clarify that v2 predictions are not identical to the paper's v1 models.

For lead similarity, the recommended output is one row per parent containing reference
snapshot ID, fingerprint spec ID, nearest lead ID, maximum Tanimoto similarity, and
thresholds. A project can then choose “analogue” (minimum similarity), “novel” (maximum
similarity), or two explicit selection baskets without changing the evidence.

## Verification

Targeted verification performed in this round:

- contract registry and validation tests;
- typed evidence and parent-cardinality conservation;
- annotation, WARN, and REJECT behavior;
- simultaneous QED and Rule-of-Five policy failures for one molecule;
- independent PAINS and BRENK actions;
- decision-buffer flushing with one-row buffers;
- deterministic SA scores and asset hashes;
- structured behavior when SA assets are unavailable;
- full source-tree bytecode compilation.

Result at the round checkpoint: `23 passed` for the new component and contract tests;
the adjacent diversity/property/export/handoff audit suite also passes.

## Independent audit of adjacent components

The audit was read-only for `diversity.py`, `property_gate.py`, `exporters.py`, and
`handoff.py`. Two transient duplicate-keyword merge defects were reported immediately
and repaired concurrently; the complete source tree compiles after repair.

Items for the next iteration:

1. Streaming leader clustering is O(N x leaders). A `max_clusters` bound controls
   memory but not total comparisons; add a comparison/time budget and keep this method
   off the default path for million-scale inputs.
2. Scaffold grouping parses each molecule twice (fingerprint and scaffold); reuse the
   parsed molecule. Treating every acyclic molecule as one scaffold can also erase
   useful diversity, so offer an acyclic fingerprint-subcluster policy.
3. Scaffold round-robin recomputes scaffolds instead of consuming a precomputed
   scaffold side output. A multi-input join would better support the requested DAG and
   avoid duplicate chemistry work.
4. Property-range gating can hold several decisions per molecule for an entire large
   batch. Add an independent decision-buffer limit like the structural-alert plugin.
5. Export filename validation should reject every ASCII control character. SMILES TSV
   fields and SDF names should also reject or escape tabs/newlines even if standard
   MolCascade parent IDs are currently safe.
6. Handoff should validate a manifest-suggested filename as a basename before using it
   when no explicit output path is supplied. This matters for third-party exporters.
7. Handoff validates each Arrow batch, but an additional global uniqueness/count check
   would defend against duplicate keys split across multiple export files.
8. Selection reports should distinguish a per-scaffold-cap exclusion from a global
   budget exclusion and expose a target shortfall when caps prevent filling the target.

## Primary and official sources

- Lipinski et al., “Experimental and computational approaches to estimate solubility
  and permeability in drug discovery and development settings,” DOI
  [10.1016/S0169-409X(00)00129-0](https://pubmed.ncbi.nlm.nih.gov/11259830/).
- Bickerton et al., “Quantifying the chemical beauty of drugs,” DOI
  [10.1038/nchem.1243](https://www.nature.com/articles/nchem.1243).
- [RDKit QED API](https://www.rdkit.org/docs/source/rdkit.Chem.QED.html).
- [RDKit FilterCatalog API](https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html)
  and [catalog enumeration](https://www.rdkit.org/docs/cppapi/classRDKit_1_1FilterCatalogParams.html).
- Baell and Holloway, PAINS filters, DOI
  [10.1021/jm901137j](https://pubs.acs.org/doi/10.1021/jm901137j).
- Capuzzi, Muratov, and Tropsha, “Phantom PAINS,” DOI
  [10.1021/acs.jcim.6b00465](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00465).
- Ertl and Schuffenhauer, synthetic-accessibility score, DOI
  [10.1186/1758-2946-1-8](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8).
- Genheden et al., AiZynthFinder, DOI
  [10.1186/s13321-020-00472-1](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1).
- Swanson et al., ADMET-AI, DOI
  [10.1093/bioinformatics/btae416](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)
  and the [official local package](https://github.com/swansonk14/admet_ai).
- [Chemprop local prediction and uncertainty documentation](https://chemprop.readthedocs.io/en/latest/tutorial/cli/predict.html).
- [FPSim2 official documentation](https://chembl.github.io/FPSim2/).
