# Environments

MolCascade's screening tiers call out to engines whose dependency pins contradict each
other — and contradict MolCascade's own. KarmaDock needs `rdkit==2022.09.1`; the project
needs `rdkit>=2024.9`. AiZynthFinder wants numpy 1.26; MolCascade runs numpy 2.5. No single
interpreter satisfies all of them, and no amount of solver patience will change that.

This is not a packaging failure that someone should get around to fixing. It is why the
backend catalogue marks these engines `ISOLATED`: MolCascade never imports them, it runs
them as subprocesses addressed by **absolute path**, and records the digest of everything it
handed over. Version skew between the tiers is therefore structurally impossible rather than
merely unlikely — which is also why the environments below can be pinned hard without any of
them constraining the others.

This document is the provisioned result on one machine, in enough detail to reproduce it on
another. For the conceptual account of licences and isolation, see the main
[README](../README.md#installing).

## Topology

Four conda environments and one standalone binary:

| Environment | Python | Holds | Why it is separate |
| --- | --- | --- | --- |
| **`prism`** | 3.12.13 | MolCascade + every in-process backend + ADMET-AI | the environment you work in |
| `karmadock` | 3.10 | KarmaDock's interpreter | pins `rdkit==2022.09.1` |
| `unidock` | 3.14.7 | the `unidock` binary | conda-forge package, built per Python ABI; would overwrite `prism`'s rdkit/numpy/pandas |
| `aizynth` | 3.12.13 | `aizynthcli` | pulls rdkit 2023.9.6 + numpy 1.26 |
| GNINA 1.3.3 | — | a 1.9 GB release binary | not a package at all; GPL-2.0-or-later |

Only `prism` needs to be activated. The other three are never on `PATH` — a cascade names
their executables by absolute path, which is the whole point.

## Provisioning a new machine

Requirements: Linux x86-64, an NVIDIA GPU with a driver new enough for CUDA 12.4+, conda
(Miniforge preferred), git, curl, and roughly **30 GB** of disk.

```bash
git clone <this repo> && cd MolCascade
bash envs/bootstrap.sh
```

The script is idempotent and per-target, so an interrupted run can be resumed, and you can
provision just one piece:

```bash
bash envs/bootstrap.sh prism            # only the main environment
bash envs/bootstrap.sh gnina karmadock  # only the docking engines
bash envs/bootstrap.sh verify           # just re-run the checks
```

Two paths are configurable by environment variable; both default as shown:

```bash
MOLCASCADE_BACKEND_ROOT=~/molcascade-backends   # GNINA, its cuDNN, the KarmaDock checkout
MOLCASCADE_AIZYNTH_DATA=~/aizynth-data          # ~750 MB of retrosynthesis models
```

**Use `--solver=libmamba`** for anything you install by hand. The classic solver was
observed taking 10 GB of RSS on these specs and getting OOM-killed on a 24 GB box.

## `prism` — the working environment

Python 3.12.13. Chosen because every picky package has a cp312 wheel: FPSim2, onnxruntime,
and medchem's `cuik_molmaker` pin all publish one, and datamol / mordredcommunity /
descriptastorus are pure Python.

```bash
conda create -n prism -y -c conda-forge python=3.12
conda activate prism
pip install -e ".[all-backends,analytics,dev]"
pip install gemmi            # meeko's undeclared dependency -- see Gotchas
pip install openbabel-wheel  # GPL-2.0-only -- omit if your project cannot accept it
```

`all-backends` is FPSim2 + admet-ai + chembl_structure_pipeline + medchem + meeko +
mordredcommunity + onnxruntime. `analytics` adds duckdb, which `all-backends` deliberately
excludes. `dev` brings pytest, ruff and mypy.

Resolved versions (full list in [`prism-requirements.txt`](prism-requirements.txt), 140
packages):

| Role | Package | Version |
| --- | --- | --- |
| the project | molcascade | 0.1.0a0 (editable) |
| chemistry core | rdkit | 2026.3.5 |
| | chembl_structure_pipeline | 1.2.4 |
| | medchem | 2.0.5 |
| | mordredcommunity | 2.0.7 |
| | FPSim2 | 0.7.4 |
| | openbabel-wheel | 3.1.1.23 |
| ADMET | admet-ai | 2.0.1 |
| | chemprop | 2.3.1 |
| | lightning / torchmetrics | 2.6.5 / 1.9.0 |
| ML runtimes | torch | 2.13.0+cu130 |
| | onnxruntime | 1.29.0 |
| docking prep | meeko | 0.7.1 |
| | gemmi | 0.7.5 |
| data | numpy / pandas / pyarrow | 2.5.2 / 3.0.5 / 25.0.1 |
| | duckdb | 1.5.5 |
| | pydantic | 2.13.4 |
| | scipy / scikit-learn | 1.18.1 / 1.9.0 |

### ADMET-AI runs in-process here

Worth stating plainly, because a shared environment can make it look otherwise: admet-ai
2.0.1 requires `rdkit>=2025.9.5` and `torch>=2.8.0`, both of which MolCascade's own pins
already satisfy. It needs no environment of its own. If ADMET-AI appears to conflict, the
conflict is with something else already installed — a conda `pytorch` build, or orphaned
`torchvision`/`torchaudio` — not with MolCascade.

### torch arrives CUDA-enabled

A plain `pip install` of the ADMET extra produced `torch 2.13.0+cu130`, `cuda available:
True`, and a working GPU matmul on an RTX 4090 with no index-url argument. The older advice
to install a CUDA build *before* the ADMET extra, so as not to get a CPU wheel, no longer
applies.

## `karmadock`

KarmaDock predicts a pose from a reference ligand rather than searching a box, so its
criterion deliberately does not inherit the box fields the other two engines use.

```bash
git clone --depth 1 https://github.com/schrojunzhang/KarmaDock.git \
  ~/molcascade-backends/KarmaDock

conda create -n karmadock --solver=libmamba -y -c conda-forge python=3.10 "rdkit=2022.09.1"

P=~/anaconda3/envs/karmadock/bin/python
$P -m pip install "torch==2.5.1" "numpy==1.26.4" \
    --index-url https://download.pytorch.org/whl/cu124
$P -m pip install "torch_scatter==2.1.2" "torch_cluster==1.6.3" "torch_sparse==0.6.18" \
    "numpy==1.26.4" -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
$P -m pip install "torch_geometric==2.5.3" "numpy==1.26.4" \
    MDAnalysis prody networkx joblib scipy pandas tqdm rmsd prefetch_generator
```

| Package | Version | Note |
| --- | --- | --- |
| python | 3.10 | matches the prebuilt PyG wheels below |
| rdkit | 2022.09.1 | KarmaDock's own pin |
| numpy | 1.26.4 | **must stay below 2** — rdkit 2022.09.1's compiled extension is built against the numpy 1.x ABI |
| torch | 2.5.1+cu124 | era-matched to 2023-era code |
| torch_scatter | 2.1.2+pt25cu124 | prebuilt |
| torch_cluster | 1.6.3+pt25cu124 | prebuilt |
| torch_sparse | 0.6.18+pt25cu124 | prebuilt |
| torch_geometric | 2.5.3 | |

Four things the upstream instructions do not tell you:

**The weights already ship in the repo.** `trained_models/karmadock_screening.pkl` is
15,615,727 bytes, committed directly with no LFS pointer. The Zenodo record 7789066 download
that KarmaDock's README describes — and that this project's own catalogue note repeated — is
no longer necessary.

**`torch_scatter` alone is not enough.** `torch_cluster` is imported by
`dataset/protein_feature.py` and `torch_sparse` by
`architecture/GraphTransformer_Block.py`. Neither KarmaDock's README nor MolCascade's backend
catalogue mentions either, and both fail at import time — long after the environment looks
complete.

**The checkpoint can silently fail to load.** `utils/fns.py` calls `load_state_dict(...,
strict=False)`, so a state dict that matches *nothing* still returns successfully and leaves
the network at its random initialisation — scoring every ligand with an untrained model, with
no error anywhere. The keys carry a `module.` prefix because the checkpoint was saved from a
`DataParallel` wrapper, and `virtual_screening.py:89` builds that wrapper before loading, so
in the real code path they do match: 581 of 585 tensors, 0 missing. Worth asserting rather
than assuming — `bootstrap.sh` checks that a reference tensor actually changes value.

**Install the torch stack with pip, not conda.** Asking libmamba to solve
`pytorch=2.5.1=cuda126*` + `pytorch_scatter` + `pytorch_cluster` + `pytorch_geometric` +
`rdkit=2022.09.1` together sent it past 10 GB of RSS with no answer after eight minutes. The
PyG wheel index at `data.pyg.org` publishes `torch_scatter` and `torch_cluster` prebuilt
against this exact torch, so there is nothing to solve and nothing to compile.

`weights_path` in the criterion is identity-only — it is hashed into `method_id`, not passed
to the engine. `virtual_screening.py` loads the checkpoint from a fixed path with no flag to
redirect it, which is why `repo_path` matters and `weights_path` does not change behaviour.

## `unidock`

```bash
conda create -n unidock --solver=libmamba -y -c conda-forge unidock
```

| Package | Version |
| --- | --- |
| unidock | 1.2.0 `cuda129_h74c3ed6_1` |
| python | 3.14.7 |
| rdkit | 2026.03.1 |
| numpy / pandas | 2.5.2 / 3.0.5 |
| openmm | 8.6.0 |
| mdanalysis | 2.10.0 |

Do not install this into `prism`. The conda-forge package declares `python`, `rdkit`,
`numpy`, `pandas`, `mdanalysis`, `openmm`, `pathos` and `networkx` as dependencies and is
built per-Python-ABI, so conda would replace `prism`'s pip-installed rdkit, numpy and pandas
with its own builds. Since MolCascade only ever runs the binary as a subprocess, there is
nothing to gain by co-locating it.

Uni-Dock is GPU-only — there is no CPU code path and no device flag — and it has no weights
to fetch; its scoring functions are physics.

## `aizynth`

```bash
conda create -n aizynth --solver=libmamba -y -c conda-forge python=3.12
~/anaconda3/envs/aizynth/bin/pip install "aizynthfinder==4.4.1"
mkdir -p ~/aizynth-data && cd ~/aizynth-data
~/anaconda3/envs/aizynth/bin/download_public_data .
```

| Package | Version |
| --- | --- |
| aizynthfinder | 4.4.1 |
| python | 3.12.13 |
| rdkit | 2023.9.6 |
| numpy / pandas | 1.26.4 / 2.3.3 |
| onnxruntime | 1.29.0 |
| tables | 3.11.1 |

The data is 754 MB: `uspto_model.onnx` (91 MB), `uspto_ringbreaker_model.onnx`,
`uspto_filter_model.onnx`, two template CSVs, and `zinc_stock.hdf5` (663 MB).
`download_public_data` writes a `config.yml` containing **absolute paths**, so it must be
re-run on each machine rather than copied — or its paths rewritten by hand.

## GNINA 1.3.3

There is no conda package for GNINA. It is one 1.9 GB asset from the GitHub release:

```
asset   gnina.cuda12.8.static   (v1.3.3)
bytes   2,056,131,000
sha256  3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172
```

**Despite the name, it is not fully static.** `ldd` resolves 16 of its 17 libraries and
reports exactly one missing: `libcudnn.so.9`. Without it the binary will not even print its
version.

Rather than borrow the cuDNN that some environment's torch happens to bundle — which would
hard-code both a Python version and the assumption that torch still ships cuDNN 9 — the
bootstrap gives GNINA a copy of its own and a launcher that finds it:

```bash
pip install --no-deps --target ~/molcascade-backends/deps/cudnn "nvidia-cudnn-cu12==9.*"
# then use the wrapper at ~/molcascade-backends/bin/gnina, not the .static binary
```

`--no-deps` skips the cublas and nvrtc wheels the cuDNN wheel declares, taking the directory
from 2.3 GB to 1.3 GB. That is safe rather than hopeful: a real CNN-rescoring run produces
bit-identical scores with those two directories removed, because GNINA links its own cublas
statically. Verified with `--cnn_scoring rescore` on a 3,736-atom receptor — CNN pose score
and CNN affinity columns populated, 3.1 s on an RTX 4090.

GNINA is **GPL-2.0-or-later**, only because it links Open Babel. The default deployment
policy refuses copyleft backends, so it stays blocked until a run says otherwise:

```bash
molcascade screen --config cascade.json --library molecules.csv --allow-copyleft
```

The flag is recorded in the audit log and the exported provenance, which is what lets a
screening campaign state in its methods section which licences it accepted.

## Wiring the isolated backends into a cascade

Every one of these is an absolute path. Substitute your own `MOLCASCADE_BACKEND_ROOT`.

| Criterion | Field | Value on this machine |
| --- | --- | --- |
| Uni-Dock | `executable` | `/home/shizq/anaconda3/envs/unidock/bin/unidock` |
| GNINA | `executable` | `/home/shizq/molcascade-backends/bin/gnina` ← the wrapper |
| KarmaDock | `executable` | `/home/shizq/anaconda3/envs/karmadock/bin/python` |
| | `repo_path` | `/home/shizq/molcascade-backends/KarmaDock` |
| | `reference_ligand_path` | your reference ligand `.mol2` (required) |
| | `pocket_pdb_path` | optional |
| AiZynthFinder | `executable` | `/home/shizq/anaconda3/envs/aizynth/bin/aizynthcli` |
| | `config_path` | `/home/shizq/aizynth-data/config.yml` |

All four also take `receptor_path` (the docking three) or the shared engine fields
`num_modes`, `seed`, `max_molecules`, `timeout_per_molecule_seconds`, `scratch_dir`. Set
`scratch_dir` when the system temporary directory is a small tmpfs.

## Verifying

```bash
conda activate prism
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import meeko, admet_ai, rdkit; print(rdkit.__version__)"
molcascade doctor --allow-copyleft
pytest -q
ruff check src tests
```

Expected on a correct install: **1077 passed, 3 skipped**, plus the one known failure below.

`molcascade doctor` will report the isolated backends as `unavailable — local command is not
installed`. That is correct behaviour, not a broken install: doctor probes `PATH`, while a
cascade uses the absolute paths above. `--allow-copyleft` is how you tell a licence refusal
apart from an engine that is genuinely missing. Put the executables on `PATH` temporarily and
doctor confirms it can see them:

```
$ PATH=/tmp/probe-path:$PATH molcascade doctor --allow-copyleft
  available   gnina           — local executable found
  available   unidock         — local executable found
  available   aizynthfinder   — local executable found
  unavailable karmadock       — local command is not installed: karmadock
```

### What was actually exercised here

Rather than assume, each engine was run:

| Check | Result |
| --- | --- |
| `pytest` in `prism` | 1077 passed, 3 skipped |
| `ruff check src tests` | clean |
| end-to-end `molcascade screen` | 12 molecules → 7 shortlisted through all 6 tiers of the shipped default cascade, including the vendored SCScore weights, both medchem arms and the ChEMBL checker |
| torch in `prism` | 2.13.0+cu130, RTX 4090 visible, GPU matmul |
| GNINA CNN rescoring | real run on a 3,736-atom receptor; CNN pose score + CNN affinity populated, 3.1 s |
| Uni-Dock | `Uni-Dock v1.2.0` |
| KarmaDock CUDA stack | `torch_scatter.scatter_mean` and `torch_cluster.knn_graph` both executed on GPU |
| KarmaDock checkpoint | 581/585 tensors loaded, 0 missing, reference tensor changed value |
| KarmaDock CLI | `virtual_screening.py --help` |
| AiZynthFinder | 2-molecule retrosynthesis cascade, 18 s, both molecules passing a ≤6-step gate |

Not exercised: a docking tier driven end-to-end *through a cascade config*. Each engine was
verified directly and through MolCascade's own executable probe, but wiring a receptor and a
box into a cascade JSON and running the docking criterion is a separate step.

## Gotchas

**`test_doctor_never_runs_commands_without_explicit_opt_in` fails on any machine with
NVIDIA tooling.** `doctor`'s GPU probe shells out to `nvidia-smi`; the test asserts doctor
executes no commands by default. It can only pass where `nvidia-smi` does not exist. Not an
install problem, and unrelated to anything above.

**`doctor` can never find KarmaDock, on any machine.** The catalogue probes it with
`command=("karmadock", "--help")`, but the KarmaDock project ships no `setup.py`, no
`pyproject.toml`, no `console_scripts` and no executable of that name — it is a repository of
scripts, invoked as `python utils/virtual_screening.py`. So `unavailable karmadock` is
permanent and says nothing about whether the environment is correct. Use the checkpoint smoke
test in `bootstrap.sh` to tell whether KarmaDock is really installed.

**meeko 0.7.1 needs gemmi, but does not say so.** `meeko/__init__.py` imports
`polymer` → `chemtempgen` → `gemmi` at package-import time, and the wheel metadata omits it.
pip reports success and every `import meeko` then raises `ModuleNotFoundError`. Worse,
MolCascade probes optional backends with `importlib.util.find_spec` — deliberately, to avoid
importing CUDA-heavy packages during preflight — so `doctor` reports meeko as *available*
while the Uni-Dock ligand-prep path fails at run time. `pip install gemmi`.

**Some `unavailable` lines in `doctor` are not gaps.** `skfp`, `bblean` (BitBirch),
`lightgbm`, `xgboost`, `scaffoldgraph`, `paretoset` and `pymoo` are catalogued backend
entries with **no plugin behind them** — nothing under `src/molcascade/plugins/` imports any
of the seven. They belong to no extra and installing them changes nothing, which is why
`all-backends` really is everything installable. The `similarity` extra's comment in
`pyproject.toml` says as much about scikit-fingerprints: it was listed and then removed
because no adapter imports it.

**Open Babel is in no extra.** Two backends are catalogued against it (format rescue,
descriptors) and two tests assume it is installed, but it is GPL-2.0-only, so the choice is
left to you. Without it those two tests fail; with it they pass and the backends remain
licence-blocked until `--allow-copyleft`.

**One test skips on RDKit 2026.03.5.** The exact-identity golden is locked to 2026.03.1 on
purpose. The skip message says so.

**402 stale `.pyc` files** in the checkout carry a `co_filename` from the machine the repo
was authored on, which is why a pytest traceback can print `???` next to an unfamiliar path.
Cosmetic. `find . -name '__pycache__' -type d -exec rm -rf {} +` if it bothers you.

**WSL2 caps memory at 50% of host RAM** when no `.wslconfig` exists — 23 GiB on a 48 GB
machine, which is not enough headroom for a classic-solver conda run. See
`%USERPROFILE%\.wslconfig` and `wsl --shutdown` to apply.

## Disk

| | |
| --- | --- |
| `prism` | ~6.0 GB |
| `karmadock` | ~4 GB |
| `unidock` | ~2 GB |
| `aizynth` | ~2 GB |
| GNINA binary | 1.9 GB |
| GNINA's cuDNN | 1.3 GB |
| AiZynthFinder data | 754 MB |
| KarmaDock checkout | 50 MB |
