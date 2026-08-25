# What to cite

Every tool the cascade builder can put in a screen, and the reference to cite when
it decided which molecules survived.

This file is generated from `src/molcascade/cascade/catalog.py`. Do not edit it by
hand -- regenerate it with:

```bash
molcascade cite --output docs/citations.md
```

Vendored weights and rule tables carry their own citation next to the bytes; see
[`vendor/MANIFEST.md`](../vendor/MANIFEST.md). MolCascade itself is not a method
and does not want citing in place of the tools it orchestrates.

**Runnable here** means an adapter is registered and the option can be selected in
the builder. **Reviewed** means the tool was assessed and its reference recorded,
but nothing in MolCascade can execute it yet -- it is listed so that the menu shows
what was considered and rejected, not only what shipped.

24 tools across 11 criteria; 17 of them have an adapter and can run here. 2 vendored asset(s) carry weights or rule tables of their own.

## Tools

### Structural validity

Terminal rejects for unparseable strings, valence errors, empty or fragment-only records, disallowed elements and severe reactivity.

#### RDKit hard rules

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~850 molecules/second on one core

RDKit: open-source cheminformatics. https://www.rdkit.org — the sanitization, valence and FilterCatalog machinery this gate drives.

#### ChEMBL Structure Pipeline checker

- **Engine** — ChEMBL
- **Licence** — MIT
- **Status** — Runnable here
- **Needs installed** — chembl_structure_pipeline
- **Measured throughput** — ~100 molecules/second on one core

Bento AP, Hersey A, Félix E, et al. An open source chemical structure curation pipeline using RDKit. J Cheminform. 2020;12:51. doi:10.1186/s13321-020-00456-1

### Physicochemical window

Editable MW/cLogP/TPSA/HBD/HBA/rotor/charge ranges with complete decisions.

#### RDKit property ranges

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~270 molecules/second on one core

Wildman SA, Crippen GM. Prediction of physicochemical parameters by atomic contributions. J Chem Inf Comput Sci. 1999;39(5):868-873. doi:10.1021/ci990307l (cLogP); Ertl P, Rohde B, Selzer P. Fast calculation of molecular polar surface area. J Med Chem. 2000;43(20):3714-3717. doi:10.1021/jm000942e (TPSA)

#### Mordred descriptor window

- **Engine** — Mordred
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Needs installed** — mordred
- **Measured throughput** — ~400 molecules/second on one core

Moriwaki H, Tian Y-S, Kawashita N, Takagi T. Mordred: a molecular descriptor calculator. J Cheminform. 2018;10:4. doi:10.1186/s13321-018-0258-y; Ritchie TJ, Macdonald SJF. The impact of aromatic ring count on compound developability. Drug Discov Today. 2009;14(21-22):1011-1020. doi:10.1016/j.drudis.2009.07.014 (the shipped default window)

### Rule of Five and QED

Rule of Five violation counting and QED are reported separately because they measure different things.

#### RDKit Rule of Five + QED

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~113 molecules/second on one core

Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. Quantifying the chemical beauty of drugs. Nat Chem. 2012;4(2):90-98. doi:10.1038/nchem.1243 (QED); Lipinski CA, Lombardo F, Dominy BW, Feeney PJ. Adv Drug Deliv Rev. 1997;23(1-3):3-25. doi:10.1016/S0169-409X(96)00423-1 (Rule of Five)

#### medchem published rule sets

- **Engine** — medchem
- **Licence** — Apache-2.0
- **Status** — Runnable here
- **Needs installed** — medchem
- **Measured throughput** — ~45 molecules/second on one core

medchem, Datamol.io. https://github.com/datamol-io/medchem — every rule set additionally carries the reference of the paper that published it, and that reference is what belongs in a methods section alongside this one.

### PAINS and reactivity alerts

Substructure alert catalogues. These are triage hypotheses about assay behaviour, not measured chemical stability.

#### RDKit PAINS and reactivity catalogues

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~85 molecules/second on one core

Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds (PAINS). J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j; Brenk R, Schipani A, James D, et al. Lessons learnt from assembling screening libraries for drug discovery for neglected diseases. ChemMedChem. 2008;3(3):435-444. doi:10.1002/cmdc.200700139

#### medchem alert collections and NIBR rules

- **Engine** — medchem
- **Licence** — Apache-2.0
- **Status** — Runnable here
- **Needs installed** — medchem
- **Measured throughput** — ~28 molecules/second on one core

Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds (PAINS). J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j; Schuffenhauer A, Schneider N, Hintermann S, et al. Evolution of Novartis' small molecule screening deck design. J Med Chem. 2020;63(23):14425-14447. doi:10.1021/acs.jmedchem.0c01332 (NIBR)

#### rd_filters alert collection (Walters)

- **Engine** — MolCascade + RDKit
- **Licence** — MIT
- **Status** — Runnable here
- **Needs assets** — rd_filters
- **Measured throughput** — ~31 molecules/second on one core

Walters WP. rd_filters. https://github.com/PatWalters/rd_filters — the alert definitions are the ChEMBL structural-alert sets, and the table shipped with a run is pinned by digest.

### ADMET property rules

Fixed equations over molecular properties, published between 2000 and 2008 and in continuous use since. No model, no weights, no download, and no GPU -- which is what makes them affordable above a neural ensemble rather than instead of one.

#### medchem ADMET rule sets

- **Engine** — medchem
- **Licence** — Apache-2.0
- **Status** — Runnable here
- **Needs installed** — medchem
- **Measured throughput** — ~45 molecules/second on one core

Egan WJ, Merz KM Jr, Baldwin JJ. Prediction of drug absorption using multivariate statistics. J Med Chem. 2000;43(21):3867-3877. doi:10.1021/jm000292e (Egan); Hughes JD, Blagg J, Price DA, et al. Physiochemical drug properties associated with in vivo toxicological outcomes. Bioorg Med Chem Lett. 2008;18(17):4872-4875. doi:10.1016/j.bmcl.2008.07.071 (Pfizer 3/75); Gleeson MP. Generation of a set of simple, interpretable ADMET rules of thumb. J Med Chem. 2008;51(4):817-834. doi:10.1021/jm701122q (GSK 4/400); implemented in medchem, Datamol.io. https://github.com/datamol-io/medchem

### ADMET endpoint

Runs a local model, then applies one explicit numeric window to one exact endpoint. Missing evidence is an explicit reject, never a pass.

#### ADMET-AI v2

- **Engine** — ADMET-AI
- **Licence** — MIT
- **Status** — Runnable here
- **Needs installed** — admet_ai, chemprop, lightning, torch
- **Measured throughput** — ~7 molecules/second on one core

Swanson K, Walther P, Leitz J, et al. ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries. Bioinformatics. 2024;40(7):btae416. doi:10.1093/bioinformatics/btae416

#### Chemprop 2.x project checkpoint

- **Engine** — Chemprop
- **Licence** — MIT
- **Status** — Runnable here
- **Needs installed** — chemprop, torch

Heid E, Greenman KP, Chung Y, et al. Chemprop: a machine learning package for chemical property prediction. J Chem Inf Model. 2024;64(1):9-17. doi:10.1021/acs.jcim.3c01250; Yang K, Swanson K, Jin W, et al. Analyzing learned molecular representations for property prediction. J Chem Inf Model. 2019;59(8):3370-3388. doi:10.1021/acs.jcim.9b00237

#### OpenADMET models

- **Engine** — OpenADMET
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — openadmet_models

OpenADMET. https://openadmet.org — an open data and model consortium; cite the specific released model actually used, not the project.

### Synthesizability score

A score is a fast proxy for how hard something looks to make. It is not a route and it is not a step count. Two scores that disagree are more informative than one score you trust.

#### SA Score (Ertl and Schuffenhauer)

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~248 molecules/second on one core

Ertl P, Schuffenhauer A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. J Cheminform. 2009;1:8. doi:10.1186/1758-2946-1-8

#### SCScore (Coley, reaction-trained)

- **Engine** — MolCascade + numpy
- **Licence** — MIT
- **Status** — Runnable here
- **Needs assets** — scscore

Coley CW, Rogers L, Green WH, Jensen KF. SCScore: synthetic complexity learned from a reaction corpus. J Chem Inf Model. 2018;58(2):252-261. doi:10.1021/acs.jcim.7b00622

#### RAscore retrosynthetic accessibility

- **Engine** — RAscore
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — RAscore

Thakkar A, Chadimová V, Bjerrum EJ, Engkvist O, Reymond JL. Retrosynthetic accessibility score (RAscore). Chem Sci. 2021;12:3339-3349. doi:10.1039/D0SC05401A

#### SYBA Bayesian classifier

- **Engine** — SYBA
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — syba

Voršilák M, Kolář M, Čmelo I, Svozil D. SYBA: Bayesian estimation of synthetic accessibility of organic compounds. J Cheminform. 2020;12:35. doi:10.1186/s13321-020-00439-2

### Retrosynthesis route search

The only component that produces genuine step counts. Route search is orders of magnitude slower than a score and belongs late in the funnel.

#### AiZynthFinder 4

- **Engine** — AiZynthFinder
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — aizynthfinder

Genheden S, Thakkar A, Chadimová V, Reymond JL, Engkvist O, Bjerrum E. AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. J Cheminform. 2020;12:70. doi:10.1186/s13321-020-00472-1

#### Syntheseus

- **Engine** — Syntheseus
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — syntheseus

Maziarz K, Tripp A, Liu G, et al. Re-evaluating retrosynthesis algorithms with Syntheseus. arXiv:2310.19796 (2023). https://github.com/microsoft/syntheseus

### Similarity to reference leads

Maximum Tanimoto similarity to a reference set. Use a minimum to stay near known chemistry, or a maximum to demand novelty.

#### RDKit exact reference similarity

- **Engine** — RDKit
- **Licence** — BSD-3-Clause
- **Status** — Runnable here
- **Measured throughput** — ~610 molecules/second on one core

Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t

#### FPSim2 indexed similarity

- **Engine** — FPSim2
- **Licence** — MIT
- **Status** — Runnable here
- **Needs installed** — FPSim2
- **Measured throughput** — ~220 molecules/second on one core

Félix E. FPSim2: simple package for fast molecular similarity searches. EMBL-EBI. https://github.com/chembl/FPSim2

### Custom model prediction

Point at a model bundle you exported yourself. MolCascade computes the features, runs the graph locally, and thresholds one named endpoint.

#### MolCascade model bundle (ONNX)

- **Engine** — ONNX Runtime
- **Licence** — MolCascade
- **Status** — Runnable here
- **Needs installed** — onnxruntime

ONNX Runtime. https://onnxruntime.ai — cite the model you exported into the bundle; this line names only the runtime that executed it.

### Predicted binding affinity

Needs a target structure in addition to the ligand. Structure-aware affinity models rank; they do not produce calibrated Ki.

#### Boltz-2 affinity head

- **Engine** — Boltz
- **Licence** — MIT
- **Status** — Reviewed, no adapter
- **Needs installed** — boltz, torch

Passaro S, Corso G, Wohlwend J, et al. Boltz-2: towards accurate and efficient binding affinity prediction. bioRxiv (2025). https://github.com/jwohlwend/boltz

#### Your own affinity model

- **Engine** — ONNX Runtime
- **Licence** — unknown
- **Status** — Reviewed, no adapter

Cite your own model and its training set. This option is a signpost to the custom model bundle criterion, not a method of its own; the bundle's own metadata is what a reader needs.

## Weights and rule tables

Bytes that ship separately from the code, each pinned by SHA-256 so the thing that
ran is the thing that was reviewed. Every directory also carries its own
`CITATION.md`; this section is the same content in one place, and
[`vendor/MANIFEST.md`](../vendor/MANIFEST.md) records the digests.

One set of weights is not listed here because it is not vendored: ADMET-AI v2 ships
its checkpoints inside its wheel, at `admet_ai/resources/models`. The digest check
still applies -- see the ADMET endpoint options above for its reference.

### SCScore published weights (Coley et al.)

Feed-forward weights trained on twelve million Reaxys reactions to predict synthetic complexity on a 1-5 scale. Three published variants differ in fingerprint width and in whether counts or bits were used during training.

- **Version** — git 37090a6aa822
- **Licence** — MIT
- **Size on disk** — 20.4 MiB across 5 file(s)
- **Trust** — data
- **Read by** — synthesis.scscore@0.1.0
- **Local path** — `vendor/scscore/`

Coley, C. W.; Rogers, L.; Green, W. H.; Jensen, K. F. SCScore: Synthetic Complexity Learned from a Reaction Corpus. Journal of Chemical Information and Modeling 58(2), 252-261 2018. DOI: 10.1021/acs.jcim.7b00622

### rd_filters structural alert collection (Walters)

1,251 SMARTS alerts assembled from eight published medicinal-chemistry filter sets -- BMS, Dundee, Glaxo, Inpharmatica, LINT, MLSMR, PAINS and SureChEMBL -- each carrying the occurrence count above which it fires.

- **Version** — git 11a3b31f5b36
- **Licence** — MIT
- **Size on disk** — 0.1 MiB across 4 file(s)
- **Trust** — data
- **Read by** — chemistry.rd_filters_alerts@0.1.0
- **Local path** — `vendor/rd_filters/`

Walters, W. P. rd_filters: filter compounds using RDKit and a set of structural alerts. GitHub: PatWalters/rd_filters 2018.

Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion in Bioassays. Journal of Medicinal Chemistry 53(7), 2719-2740 2010. DOI: 10.1021/jm901137j

Bruns, R. F.; Watson, I. A. Evaluating the Utility of Compound Filters: Lessons from a Large-Scale Analysis. Journal of Medicinal Chemistry 55(22), 9763-9772 2012. DOI: 10.1021/jm301008n

Rishton, G. M.; Hann, M. M.; and the Glaxo Wellcome group. A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery. Journal of Combinatorial Chemistry 1(1), 55-68 1999. DOI: 10.1021/cc9800071
