# Bringing your own model

A model bundle is a directory you hand MolCascade: one manifest, one model file,
and a digest that pins the run to exactly those bytes. It is how a model you
trained on your own data becomes one tier of a cascade, scored beside SA score
and PAINS and everything else, with the same evidence-then-policy split.

There are two kinds, and which one you need is decided by the model, not by
preference.

| | **ONNX bundle** | **Chemprop bundle** |
|---|---|---|
| Manifest | `molcascade_model.yaml` | `molcascade_chemprop.yaml` |
| Model file | `.onnx` graph | Chemprop 2.x checkpoint (`.pt`) |
| Input | a fixed-width vector **MolCascade computes** | SMILES, featurized **by Chemprop** |
| You declare the representation | yes — it is part of the manifest | no — there is nothing to declare |
| Loading executes code | no | **yes** |
| Trust flag required | no | **yes** |
| Needs installed | `onnxruntime` | `chemprop`, `torch` |

Use the ONNX bundle for anything that consumes a feature vector: a gradient
boosting model, a random forest, a feed-forward net, anything you can export
with `skl2onnx`. Use the Chemprop bundle for a message-passing network, which
has no feature-vector input to export — its featurization is a trained part of
the model rather than a preprocessing step you chose.

Neither one downloads anything, installs anything, or trains anything. You train
elsewhere and copy the result in.

## Building one

Put the manifest and the model file in one directory. Flat — no subdirectories,
no symlinks. Then ask for its digest:

```bash
molcascade model-bundle /path/to/bundle
```

That reads and hashes; it does not load the model, so it is safe to run on a
bundle before you have decided to trust it. It prints the two lines to paste
into the criterion's settings in the builder.

The digest covers **every file in the directory**, not just the one the manifest
names. Dropping a README in afterwards changes it. That is deliberate: the pin
is over what is there, so a bundle cannot quietly grow a file between the day
you reviewed it and the day it ran.

### An ONNX manifest

```yaml
schema_version: 1
model_name: solubility-rf
endpoint_id: project/logS
task: regression          # or binary_probability
representation:
  blocks:
    - kind: morgan
      radius: 2
      n_bits: 2048
    - kind: descriptors
      names: [MolWt, MolLogP, TPSA]
model:
  file: model.onnx
  sha256: <sha256 of model.onnx>
  n_features: 2051        # must equal the representation's total width
  dtype: float32
  output_column: 1        # for a classifier: the positive class
```

The `n_features` check is not bookkeeping. If the representation in the manifest
is not the one the model was trained on, in the same column order, the model
still runs and still returns numbers — they are just meaningless. Declaring the
width makes the common half of that mistake fail loudly.

Export classifiers with ZipMap disabled, or the graph emits a list of
`{class: probability}` dicts instead of a tensor and there is no column to read:

```python
convert_sklearn(model, initial_types=..., options={id(model): {"zipmap": False}})
```

### A Chemprop manifest

```yaml
schema_version: 1
model_name: kinase-selectivity
endpoint_id: project/pIC50
task: regression          # or binary_probability
units: pIC50              # free text; what the number means
model:
  file: model.pt
  sha256: <sha256 of model.pt>
  format: chemprop_pt
  n_tasks: 1              # tasks the saved head emits
  task_index: 0           # which one decides in this tier
```

Save the checkpoint from Chemprop itself:

```python
from chemprop import models
models.save_model("model.pt", trained_mpnn)
```

`n_tasks` and `task_index` matter for a multitask model. A checkpoint trained on
twelve endpoints contributes **one** of them to a tier — the others are not
averaged in, and the column is not defaulted to zero just because that is what
indexing does. Declaring the shape also means a checkpoint swapped for a
differently-shaped one fails closed instead of continuing to score, silently, on
a different endpoint.

The adapter also checks that the head kind matches `task`: a regression head
returns raw values and a classification head returns probabilities, so declaring
the wrong one would reinterpret every number the stage emits and put a threshold
meant for 0–1 on a scale that has no such bounds.

## Why the Chemprop bundle asks a question the ONNX one does not

`chemprop.models.load_model` calls `torch.load(..., weights_only=False)`.
Deserializing a checkpoint executes code from the file. This is not specific to
Chemprop — it is true of Torch checkpoints generally, and it is why the ONNX
adapter refuses `.pkl`, `.joblib`, `.pt`, `.pth`, `.bin`, `.h5`, `.keras` and
`.pmml` outright.

The Chemprop adapter cannot refuse `.pt`; that is the format. So the refusal is
replaced with an explicit acknowledgement:

```yaml
allow_unsafe_model_deserialization: true
```

The stage will not start without it, and it is checked before the bundle is
read, so an untrusted stage never touches the checkpoint at all.

A digest and a trust flag answer different questions. The digest says *which
bytes ran* — it makes the run reproducible and makes a substituted file fail. It
does not say the bytes are safe. Only a person who knows where the checkpoint
came from can say that, which is why it is a flag and not a default.

## What the stage emits

Both adapters write two outputs: the parents, unchanged, and a `prediction/v1`
dataset with one row per molecule that could be scored. Neither one decides
anything. A numeric evidence gate downstream turns predictions into PASS/REJECT
against the window you set in the builder — the same separation of evidence from
policy that ADMET-AI and SA score use.

Molecules that could not be scored have **no row**, and the gate rejects missing
evidence rather than passing it. Two counts in the stage metadata say how many
and why:

- `unfeaturizable_count` — RDKit could not parse the SMILES.
- `non_finite_prediction_count` — the model returned NaN or an infinity.

One odd molecule does not abort a two-million-row screen, but it does not get
scored either.

Each prediction carries a `model_id` binding the model bytes to how they were
turned into numbers: MolCascade's featurizer version for an ONNX bundle, and the
installed Chemprop release for a Chemprop one, because that is where the
featurizer lives. Two runs that agree on a `model_id` computed the same numbers
the same way.

## Citing it

`molcascade cite` names the runtime — ONNX Runtime, or the Chemprop papers — but
it cannot name your model. Put the reference for the model itself, its training
data, and its applicability domain in the manifest's `notes` field, and repeat it
in the methods section. A screen that deleted molecules on a model's say-so has
to be able to say which model.
