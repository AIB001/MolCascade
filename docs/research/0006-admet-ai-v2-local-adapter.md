# ADMET-AI v2 local adapter

Status: implemented as the optional plugin
`prediction.admet_ai_v2@0.1.0`.

## Why this backend is optional

[ADMET-AI](https://github.com/swansonk14/admet_ai) is an MIT-licensed local
multi-endpoint predictor built on Chemprop models trained on Therapeutics Data
Commons ADMET datasets. Its official Python API accepts a list of SMILES and
returns a Pandas table whose columns are prediction tasks. This makes it useful
as one selectable prediction module, but not as a universal truth source.

The repository's current main branch is ADMET-AI v2. The project explicitly
states that v2 uses Chemprop v2, was retrained from scratch, and does not exactly
match the v1 predictions reported in the 2024
[ADMET-AI paper](https://doi.org/10.1093/bioinformatics/btae416). MolCascade
therefore accepts major version 2 only and records the exact installed version,
package-code digest, model-tree digest, and checkpoint digests in every stage
result. Versions of the scientific runtime dependencies (including Chemprop,
Torch, Lightning, RDKit, NumPy, Pandas, SciPy, and scikit-learn) are also bound
into the model identity because they can affect inference.

ADMET predictions are model evidence, not experimental measurements. This
adapter does not infer an applicability domain, uncertainty, calibration,
chemical stability, or a pass/fail policy. Add separate modules for those
claims and join their decisions explicitly.

## Security and adapter network behavior

ADMET-AI v2 loads local Torch checkpoints. Model deserialization may execute
code, so MolCascade will not load a checkpoint unless all of these conditions
are true:

1. `models_dir` resolves to an existing local directory.
2. The complete directory is symlink-free and contains regular files only.
3. Its top level contains only ensemble directories, each with at least one
   `.pt` checkpoint, matching the official v2 loader's discovery rules.
4. The installed ADMET-AI package-code SHA-256 equals
   `expected_package_code_sha256`.
5. Its recursive canonical SHA-256 manifest equals
   `expected_model_manifest_sha256`.
6. Every optional per-checkpoint pin also matches.
7. `allow_unsafe_model_deserialization` is explicitly set to `true`.

MolCascade's adapter never installs packages, downloads checkpoints, or
intentionally calls a web service. It copies the verified model tree into the
isolated stage directory, loads that snapshot, hashes it again after inference,
and deletes it. ADMET-AI is imported only after package and model hash checks
pass. The constructor receives `include_physchem=False` and
`drugbank_path=None`; this prevents implicit use of an unpinned DrugBank
reference asset and avoids duplicating MolCascade's typed descriptor layer.

The trusted ADMET-AI package, its dependencies, and checkpoint loader can still
execute arbitrary Python code, including opening network connections. Hash
verification identifies reviewed bytes; it is neither a process sandbox nor a
network firewall. Run untrusted software in an externally isolated environment.

## Provision and inspect local assets

Install ADMET-AI v2 and its model files yourself in a dedicated environment.
MolCascade deliberately has no automatic installation path. Before creating a
pipeline, inspect the installed code and model assets without importing
ADMET-AI or deserializing checkpoints:

```python
import json

from molcascade.plugins.builtin.admet import (
    inspect_admet_ai_installation,
    inspect_admet_ai_model_assets,
)

print(json.dumps(inspect_admet_ai_installation(), indent=2))
print(json.dumps(inspect_admet_ai_model_assets("/verified/admet/models"), indent=2))
```

Review the source and provenance of those exact files. Both top-level digests
are mandatory: copy
`package_code_sha256` to `expected_package_code_sha256` and
`model_manifest_sha256` to `expected_model_manifest_sha256`. Pin individual
checkpoint entries as an additional audit control.

## Pipeline configuration

Each endpoint maps one exact ADMET-AI DataFrame column to a stable MolCascade
endpoint ID. Do not silently rename a task or change its units; instead create
a new endpoint ID and pipeline revision.

```yaml
- id: admet_panel
  slot: predict
  plugin: prediction.admet_ai_v2@0.1.0
  inputs:
    - request_port: parents
      stage: chemistry_policy
      port: primary
  config:
    schema_version: 1
    models_dir: /verified/admet/models
    expected_model_manifest_sha256: <64 lowercase hex characters>
    expected_package_code_sha256: <64 lowercase hex characters>
    expected_checkpoints:
      - relative_path: absorption/model_0.pt
        sha256: <64 lowercase hex characters>
    endpoints:
      - output_column: Caco2_Wang
        endpoint_id: admet-ai-v2/Caco2_Wang
      - output_column: HIA_Hou
        endpoint_id: admet-ai-v2/HIA_Hou
    batch_size: 256
    num_workers: 0
    allow_unsafe_model_deserialization: true
  enabled: true
```

The plugin emits two typed ports:

- `primary`: unchanged `parent/v1` rows;
- `predictions`: long-form `prediction/v1` rows keyed by parent, endpoint, and
  exact model identity.

ADMET-AI v2 averages each ensemble in its current API and does not return
per-row uncertainty intervals. MolCascade therefore writes `prediction_std`,
`interval_lower`, and `interval_upper` as null. A later calibrated or
uncertainty-aware adapter must use a distinct plugin/model identity.

For list input, the official API exposes the input SMILES as the returned
DataFrame index. MolCascade verifies that index position-by-position before
assigning predictions to parent IDs. A same-cardinality reorder or relabel is a
hard structured failure rather than a silent parent/prediction mismatch.

## Scale and placement in a screening cascade

Keep fast deterministic chemistry gates before this stage, then run selected
ADMET endpoints in parallel on the surviving population. Use small bounded
batches appropriate for the available CPU/GPU memory. Apply endpoint-specific
thresholds only in explicit downstream policy modules, ideally alongside
applicability-domain evidence. This keeps a model score, confidence/domain
evidence, and the final screening decision auditable as separate facts.

Primary implementation references:

- [ADMET-AI repository and v2 usage](https://github.com/swansonk14/admet_ai)
- [ADMET-AI v2 model API source](https://github.com/swansonk14/admet_ai/blob/main/admet_ai/admet_model.py)
- [ADMET-AI paper, Bioinformatics 2024](https://doi.org/10.1093/bioinformatics/btae416)
- [PyTorch serialization security guidance](https://pytorch.org/docs/stable/notes/serialization.html)
