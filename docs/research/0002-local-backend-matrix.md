# Local Screening Backend Matrix (Research Freeze)

Date: 2026-08-23  
Scope: Stages 0–7 in `plan/AI分子筛选模块化路径文档_200万到2万.docx`.  
Principle: Every layer must provide at least two interchangeable local implementations. Optional backends that are not installed must be shown explicitly as `unavailable`; the system must neither switch silently nor download models over the network during production runs.

## 1. Overall Conclusions

MolCascade uses “stable scientific slots + interchangeable adapters” instead of allowing arbitrary plugins to modify arbitrary data. All adapters exchange only versioned Parquet contracts. Tool versions, rule/model SHA-256 hashes, configurations, random seeds, commands, and licenses are all recorded in provenance.

Different chemistry tools do not inherently produce equivalent canonical SMILES. To prevent the same `parent_id` from changing with the backend, the default authoritative identity source is fixed to RDKit RegistrationHash. Outputs from the ChEMBL Structure Pipeline, Open Babel, and other tools must pass back through the selected identity registrar, while retaining the adapter's own transformation records.

Backends are divided into three tiers:

- `default`: Installed with MolCascade core or the chemistry extra and capable of running fully offline.
- `optional`: Supplied with a complete adapter and health check, and enabled only after explicit installation by the user.
- `isolated`: Heavy dependencies, complex GPL/commercial licensing, or GPU requirements make WSL, a container, or a separate process preferable.

## 2. Backend Choices by Layer

| Layer | Default local implementation | Secondary implementation/fallback | Optional heavyweight implementation | Key limitations |
|---|---|---|---|---|
| 0 Reference and contracts | Pydantic v2 + PyArrow | Pandera sampling/reference-table review | DuckDB QC | Pydantic does not validate two million records row by row |
| 1 Ingestion | PyArrow/stdlib delimited + RDKit SDF | Raw Parquet source | Open Babel CLI format rescue | Open Babel output must still be registered by RDKit |
| 2 Standardization/registration | RDKit MolStandardize + RegistrationHash | ChEMBL Structure Pipeline adapter | Project-specific rules package | MolVS is used only to reproduce legacy workflows |
| 3 Hard gates/alerts | RDKit FilterCatalog | Versioned project SMARTS | Lilly Medchem Rules / MayaChemTools | PAINS is `WARN` by default, not a hard rejection |
| 4A Descriptors/fingerprints | RDKit descriptors + Morgan/AtomPair | Open Babel descriptors or MordredCommunity | PaDEL/CDK | The feature manifest is part of model identity |
| 4B QSAR | scikit-learn baseline | XGBoost / LightGBM | Chemprop / QSARtuna | Compare classical baselines first on small reference sets; prevent leakage from random splits |
| 4C Applicability domain/UQ | RDKit Tanimoto + sklearn neighbors | scikit-fingerprints / MAPIE | Chemprop UQ | AD and uncertainty are separate fields and must not be combined into “confidence” |
| 4D Synthetic accessibility | RDKit SA Score | SCScore | AiZynthFinder route-aware | SA, SC, and RA are not proof of synthesizability |
| 5A Scaffolds | RDKit Murcko | ScaffoldGraph | Project-specific scaffold definition | Acyclic molecules must enter an explicit acyclic channel |
| 5B Clustering/selection | RDKit LeaderPicker | Python packed-bit greedy leader | BitBIRCH-Lean / FPSim2 / chemfp | Prohibit dense `N×N` matrices; chemfp 5.1 is proprietary software |
| 6 Final selection | MolCascade deterministic four-basket selection + blockwise Pareto | paretoset | pymoo cross-validation | Create mutually exclusive baskets first, then apply quotas/backfill; hard rejections cannot be recovered |
| 7 Shortlist delivery | Parquet manifest | RDKit SDF export | PRISM handoff adapter (later) | MolCascade does not execute docking/MD/MMPBSA |

## 3. Current Explicit Exclusions

- SwissADME is a web service with no official local distribution. It cannot serve as an offline adapter and must not be scraped.
- chemfp 5.1 is not open-source software. It may be offered only as `optional_licensed`, and its health check must verify license capabilities and platform support.
- The original Mordred 1.2.0, MolVS, and legacy RAscore are retained only as adapters for legacy reproducibility.
- Euclidean centroids from MiniBatchKMeans are not equivalent to Morgan/ECFP Tanimoto clustering. MiniBatchKMeans may be used only with continuous descriptors or in an embedding space, and the method name must state the distance semantics.
- FPSim2 is a similarity-search/sparse-adjacency tool and must not be described as a sphere-exclusion clustering implementation.

## 4. Health-Check Standard

Every backend provides five diagnostic levels before entering the plugin registry:

1. `H0 static`: import/CLI version, OS/CPU/GPU capabilities, license, and model/rule-file checksums.
2. `H1 chemistry`: sentinels covering benzene, salts, zwitterions, isotopes, stereochemistry/tautomerism, and invalid structures.
3. `H2 deterministic`: IDs, counts, and columns are identical for the same input/configuration/version/seed; floating-point values have explicit tolerances.
4. `H3 offline/resource`: disconnected-network behavior, a 10k pilot, peak RSS/temporary disk, isolation of individual bad molecules, and Windows/WSL paths.
5. `H4 asset/model`: complete feature/model/rule/stock hashes and training/calibration/test ID lists.

`doctor` only reports capabilities; it does not automatically install, download, or trust third-party code.

## 5. Adapter Conformance

All backends share the following acceptance requirements:

- Schema, primary keys, nullability, enumerations, and cross-table conservation must pass before an artifact is published;
- Input artifacts are completely immutable, and plugins write only to the staging root assigned by the runner;
- A single bad molecule enters a reject/decision output and does not cause silent loss of the entire batch;
- Multithreaded output is restored to a deterministic order by stable entity ID;
- Provenance includes at least plugin, backend, version, license, configuration, seed, thread count, and asset hashes;
- Pickle is not used for public artifacts; private databases from external tools are converted immediately to Parquet;
- The runtime remains offline; models, checkpoints, stocks, and JavaScript assets may be prepared only during provisioning/installation;
- A 10k functional validation and a 500k load test must pass before the complete 2M run is allowed;
- Default and replacement backends are compared only on shared scientific semantics; their canonicalization must not be claimed to be fully equivalent.

## 6. License and Platform Summary

| Software | License/boundary | Windows/WSL recommendation |
|---|---|---|
| RDKit | BSD-3-Clause | Native Windows and WSL are both supported |
| PyArrow | Apache-2.0 | Official wheel |
| DuckDB | MIT | Official wheel; prohibit automatic network extension loading |
| Open Babel | GPL-2.0-only | Prefer a CLI process boundary and review distribution obligations |
| ChEMBL Structure Pipeline | MIT | Python/RDKit; include in Windows CI |
| Lilly Medchem Rules | Apache-2.0 | Prefer WSL/container on Windows |
| MordredCommunity | BSD-3-Clause | Python 3.9–3.14; run selected columns late in the pipeline |
| PaDEL/CDK | PaDELPy MIT; CDK LGPL | Requires JRE/JAR hashes and a NOTICE file |
| scikit-learn | BSD-3-Clause | Native Windows and WSL are both supported |
| XGBoost | Apache-2.0 | Requires a runtime-library health check on Windows |
| LightGBM | MIT | CPU by default; validate GPU configurations separately |
| Chemprop | MIT | Windows with `num_workers=0`, or WSL |
| BitBIRCH-Lean | GPL-3.0-only, beta | Prefer WSL/Linux and an isolated process |
| FPSim2 | MIT | Windows wheel available |
| chemfp 5.1 | Proprietary license | Primarily supports Linux; licensed plugin only |
| paretoset | MIT | Cross-platform |
| pymoo | Apache-2.0 | C extensions are optional |

## 7. Key Primary Sources

- [RDKit documentation](https://www.rdkit.org/docs/)
- [RDKit RegistrationHash](https://rdkit.org/docs/source/rdkit.Chem.RegistrationHash.html)
- [ChEMBL Structure Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline)
- [Open Babel documentation](https://openbabel.org/docs/)
- [RDKit FilterCatalog](https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html)
- [Lilly Medchem Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules)
- [MordredCommunity](https://github.com/JacksonBurns/mordred-community)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Chemprop](https://chemprop.readthedocs.io/)
- [scikit-fingerprints AD](https://scikit-fingerprints.readthedocs.io/stable/modules/applicability_domain.html)
- [RDKit Murcko scaffold](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
- [RDKit diversity pickers](https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html)
- [BitBIRCH-Lean](https://github.com/mqcomplab/bblean)
- [FPSim2](https://chembl.github.io/FPSim2/)
- [chemfp licensing](https://chemfp.com/docs/installing.html)
- [paretoset](https://paretoset.readthedocs.io/)
- [pymoo](https://pymoo.org/)
