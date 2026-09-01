#!/usr/bin/env bash
#
# Provision every environment MolCascade needs on a fresh Linux + NVIDIA machine.
#
#   bash envs/bootstrap.sh              # everything
#   bash envs/bootstrap.sh prism        # just the main environment
#   bash envs/bootstrap.sh gnina karmadock
#   bash envs/bootstrap.sh boltz2      # opt-in: not in the default set
#
# Reads nothing from the network except conda-forge, PyPI, download.pytorch.org,
# data.pyg.org and the GNINA release asset. Every step is idempotent: re-running
# skips what is already in place, so an interrupted bootstrap can simply be re-run.
#
# See envs/README.md for what each environment is for and why they are separate.
set -euo pipefail

# --- knobs -------------------------------------------------------------------
# Where the out-of-process backends live. Nothing here is on PATH by design;
# cascades address these by absolute path.
BACKEND_ROOT="${MOLCASCADE_BACKEND_ROOT:-${HOME}/molcascade-backends}"
# AiZynthFinder's ~750 MB of models and stock.
AIZYNTH_DATA="${MOLCASCADE_AIZYNTH_DATA:-${HOME}/aizynth-data}"
# Boltz-2's ~3 GB of weights plus the CCD. Boltz reads this variable itself, so
# it is spelled its way rather than ours.
BOLTZ_CACHE_DIR="${BOLTZ_CACHE:-${HOME}/.boltz}"
# The checkout this script is being run from.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

GNINA_VERSION="1.3.3"
GNINA_ASSET="gnina.cuda12.8.static"
GNINA_SHA256="3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172"

# --- plumbing ----------------------------------------------------------------
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found; install Miniforge or Anaconda first"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }
pybin() { echo "$(conda info --base)/envs/$1/bin/python"; }

# The classic solver has been observed to consume >10 GB of RAM and get OOM-killed
# on these specs. libmamba is not optional here.
#
# --override-channels because otherwise these environments are built from
# whatever the user's ~/.condarc happens to list, which makes the same script
# produce different environments on different machines and, on one of them,
# none at all: a Tsinghua mirror of pkgs/msys2 answers 403 and conda refuses to
# solve at all. Everything asked for below -- python, rdkit, unidock -- is on
# conda-forge, so naming it and only it is both reproducible and sufficient.
CONDA_CREATE=(conda create --solver=libmamba -y --override-channels -c conda-forge)

# =============================================================================
# prism -- MolCascade itself, plus every backend that can share its interpreter.
# =============================================================================
setup_prism() {
  if env_exists prism; then
    note "env 'prism' already exists; installing/refreshing packages into it"
  else
    say "creating env 'prism' (Python 3.12)"
    "${CONDA_CREATE[@]}" -n prism python=3.12
  fi
  local py; py="$(pybin prism)"

  say "prism: MolCascade (editable) with all in-process extras"
  # 'all-backends' is FPSim2 + admet-ai + chembl_structure_pipeline + medchem +
  # meeko + mordredcommunity + onnxruntime. 'analytics' adds duckdb, which
  # all-backends deliberately excludes. 'dev' brings pytest/ruff/mypy.
  "${py}" -m pip install -e "${REPO_ROOT}[all-backends,analytics,dev]"

  # meeko 0.7.1 imports gemmi at package-import time but does not declare it.
  # Without this, `import meeko` raises ModuleNotFoundError while
  # `molcascade doctor` still reports meeko as available (it probes with
  # importlib.util.find_spec, which only needs the directory to exist).
  say "prism: meeko's undeclared gemmi dependency"
  "${py}" -m pip install gemmi

  # Open Babel is in no extra on purpose -- it is GPL-2.0-only and the default
  # deployment policy refuses copyleft backends. Two catalogued backends and two
  # tests need it present. Skip this line if your project cannot accept GPL.
  say "prism: Open Babel (GPL-2.0-only; stays licence-blocked until --allow-copyleft)"
  "${py}" -m pip install openbabel-wheel

  note "torch: $("${py}" -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())')"
}

# =============================================================================
# karmadock -- pins rdkit 2022.09.1, so it can never share prism's interpreter.
# =============================================================================
setup_karmadock() {
  local repo="${BACKEND_ROOT}/KarmaDock"

  if [ -d "${repo}/.git" ]; then
    note "KarmaDock checkout already at ${repo}"
  else
    say "cloning KarmaDock into ${repo}"
    mkdir -p "${BACKEND_ROOT}"
    git clone --depth 1 https://github.com/schrojunzhang/KarmaDock.git "${repo}"
  fi
  # The screening weights are committed to the repo (15.6 MB, not LFS), so the
  # Zenodo 7789066 download that KarmaDock's README describes is unnecessary.
  [ -f "${repo}/trained_models/karmadock_screening.pkl" ] \
    || die "KarmaDock weights missing at ${repo}/trained_models/karmadock_screening.pkl"

  if env_exists karmadock; then
    note "env 'karmadock' already exists"
  else
    # conda handles only python + rdkit. Asking it to solve the torch stack too
    # sends libmamba into a multi-GB search; the PyG wheel index below is both
    # faster and exactly version-matched.
    say "creating env 'karmadock' (Python 3.10 + rdkit 2022.09.1)"
    "${CONDA_CREATE[@]}" -n karmadock python=3.10 "rdkit=2022.09.1"
  fi
  local py; py="$(pybin karmadock)"

  # rdkit 2022.09.1's compiled extension is built against the numpy 1.x ABI, so
  # numpy must stay below 2. Pinning it in each pip call stops a transitive
  # dependency from silently upgrading it and breaking `import rdkit`.
  local NP="numpy==1.26.4"

  say "karmadock: torch 2.5.1+cu124"
  "${py}" -m pip install "torch==2.5.1" "${NP}" \
    --index-url https://download.pytorch.org/whl/cu124

  # Prebuilt against this exact torch -- no compilation, no solver.
  # All three are required: torch_cluster by dataset/protein_feature.py and
  # torch_sparse by architecture/GraphTransformer_Block.py. Neither KarmaDock's
  # README nor MolCascade's backend catalogue mentions either one, and both fail
  # only at import time, well after the environment looks finished.
  say "karmadock: torch_scatter / torch_cluster / torch_sparse prebuilt for pt25cu124"
  "${py}" -m pip install "torch_scatter==2.1.2" "torch_cluster==1.6.3" \
    "torch_sparse==0.6.18" "${NP}" \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

  say "karmadock: remaining runtime dependencies"
  # bio-embeddings-esm appears in KarmaDock's own env file but is training-only.
  "${py}" -m pip install "torch_geometric==2.5.3" "${NP}" \
    MDAnalysis prody networkx joblib scipy pandas tqdm rmsd prefetch_generator

  # KarmaDock loads its checkpoint with strict=False, so a state_dict that
  # matches nothing at all still "succeeds" and leaves the network at its random
  # initialisation -- scoring every ligand with an untrained model. Assert that
  # the weights actually land. The checkpoint was saved from a DataParallel
  # wrapper, so its keys carry a "module." prefix and the wrapper is required
  # for them to match; that is what virtual_screening.py:89 does.
  say "karmadock: checking the checkpoint really loads"
  ( cd "${repo}" && "${py}" - <<'SMOKE'
import sys, torch, torch.nn as nn
sys.path.insert(0, ".")
from architecture.KarmaDock_architecture import KarmaDock

model = nn.DataParallel(KarmaDock(), device_ids=[0], output_device=0).to("cuda:0")
state = torch.load("trained_models/karmadock_screening.pkl",
                   map_location="cuda:0", weights_only=True)["model_state_dict"]
key = "module.lig_encoder.node_encoder.weight"
before = model.state_dict()[key].clone()
missing, _ = model.load_state_dict(state, strict=False)
if missing:
    sys.exit(f"FAIL: {len(missing)} model tensors were not in the checkpoint")
if torch.equal(before, model.state_dict()[key]):
    sys.exit("FAIL: weights did not change; the model is still random")
print(f"    OK: {len(state)} tensors loaded, weights changed")
SMOKE
  )

  note "karmadock python: ${py}"
  note "karmadock repo:   ${repo}"
}

# =============================================================================
# unidock -- conda-forge package, built per Python ABI; must not enter prism.
# =============================================================================
setup_unidock() {
  if env_exists unidock; then
    note "env 'unidock' already exists"
  else
    # This package declares python/rdkit/numpy/pandas/mdanalysis/openmm deps and
    # is built per-Python-ABI. Installed into prism, conda would overwrite
    # prism's pip-installed rdkit and numpy. MolCascade only ever runs the
    # binary as a subprocess, so there is no benefit to co-locating it.
    say "creating env 'unidock' (Uni-Dock 1.2.0, CUDA build)"
    "${CONDA_CREATE[@]}" -n unidock unidock
  fi
  note "unidock binary: $(conda info --base)/envs/unidock/bin/unidock"
}

# =============================================================================
# aizynth -- pulls rdkit 2023.9.6 + numpy 1.26; cannot share prism either.
# =============================================================================
setup_aizynth() {
  if env_exists aizynth; then
    note "env 'aizynth' already exists"
  else
    say "creating env 'aizynth' (Python 3.12 + AiZynthFinder 4.4.1)"
    "${CONDA_CREATE[@]}" -n aizynth python=3.12
    # No extra needed: 4.4.1's base dependencies already include onnxruntime,
    # rdkit, tables and everything aizynthcli uses.
    "$(pybin aizynth)" -m pip install "aizynthfinder==4.4.1"
  fi

  if [ -f "${AIZYNTH_DATA}/config.yml" ]; then
    note "AiZynthFinder data already at ${AIZYNTH_DATA}"
  else
    say "downloading AiZynthFinder public data (~750 MB) to ${AIZYNTH_DATA}"
    mkdir -p "${AIZYNTH_DATA}"
    ( cd "${AIZYNTH_DATA}" && "$(conda info --base)/envs/aizynth/bin/download_public_data" . )
  fi
  note "aizynthcli: $(conda info --base)/envs/aizynth/bin/aizynthcli"
  note "config:     ${AIZYNTH_DATA}/config.yml"
}

# =============================================================================
# boltz2 -- Boltz-2 co-folding, for the optional last tier. MIT, code and weights.
#
# The conflict here is torch, not Python: boltz accepts prism's 3.12, but
# resolving it into prism re-pins the torch ADMET-AI predicts with in-process
# and KarmaDock was built against. So it gets an interpreter of its own, and
# MolCascade calls the `boltz` CLI in it by absolute path like every other
# ISOLATED backend.
#
# Not in the default target list: this is ~3 GB of weights on top of another
# torch, for a tier that ships switched off. Ask for it by name.
# =============================================================================
setup_boltz2() {
  if env_exists boltz2; then
    note "env 'boltz2' already exists"
  else
    say "creating env 'boltz2' (Python 3.12 + Boltz-2)"
    "${CONDA_CREATE[@]}" -n boltz2 python=3.12
    # The [cuda] extra is cuequivariance's fused kernels; boltz runs without
    # them, slower. It brings its own torch -- that is the entire point of the
    # separate environment.
    "$(pybin boltz2)" -m pip install "boltz[cuda]==2.2.1" \
      || "$(pybin boltz2)" -m pip install "boltz==2.2.1"
  fi

  # Boltz downloads on first use, inside whatever process got there first. That
  # is a 3 GB network call in the middle of a screen, so it happens here
  # instead, and the plugin refuses to start until this directory is populated.
  if [ -f "${BOLTZ_CACHE_DIR}/boltz2_conf.ckpt" ]; then
    note "Boltz-2 weights already at ${BOLTZ_CACHE_DIR}"
  else
    say "downloading Boltz-2 weights and the CCD (~3 GB) to ${BOLTZ_CACHE_DIR}"
    mkdir -p "${BOLTZ_CACHE_DIR}"
    BOLTZ_CACHE="${BOLTZ_CACHE_DIR}" "$(pybin boltz2)" -c "
import sys
from pathlib import Path
cache = Path(sys.argv[1])
try:
    from boltz.main import download_boltz2
except ImportError:
    sys.exit('boltz.main.download_boltz2 is gone; run one prediction by hand to populate the cache')
download_boltz2(cache)
" "${BOLTZ_CACHE_DIR}"
    [ -f "${BOLTZ_CACHE_DIR}/boltz2_conf.ckpt" ] \
      || die "the download finished but ${BOLTZ_CACHE_DIR}/boltz2_conf.ckpt is not there"
  fi

  note "boltz CLI: $(conda info --base)/envs/boltz2/bin/boltz"
  note "cache:     ${BOLTZ_CACHE_DIR}"
  note "Boltz-2 needs a target FASTA and an a3m alignment; it never fetches one for you"
}

# =============================================================================
# gnina -- a 1.9 GB release binary, not a package. GPL-2.0-or-later.
# =============================================================================
setup_gnina() {
  local bin="${BACKEND_ROOT}/bin"
  mkdir -p "${bin}"

  if [ -f "${bin}/${GNINA_ASSET}" ] \
     && echo "${GNINA_SHA256}  ${bin}/${GNINA_ASSET}" | sha256sum -c --status 2>/dev/null; then
    note "GNINA binary already present and checksum matches"
  else
    say "downloading GNINA ${GNINA_VERSION} (~1.9 GB)"
    curl -fL --retry 5 --retry-delay 3 -C - -o "${bin}/${GNINA_ASSET}" \
      "https://github.com/gnina/gnina/releases/download/v${GNINA_VERSION}/${GNINA_ASSET}"
    echo "${GNINA_SHA256}  ${bin}/${GNINA_ASSET}" | sha256sum -c \
      || die "GNINA checksum mismatch"
  fi
  chmod +x "${bin}/${GNINA_ASSET}"

  # Despite the "static" in its name, ldd reports exactly one unresolved
  # library: libcudnn.so.9. Give GNINA a cuDNN of its own rather than borrowing
  # the one some environment's torch happens to bundle. --no-deps skips the
  # cublas and nvrtc wheels: a real CNN-rescoring run produces bit-identical
  # scores without them, because GNINA links its own cublas statically.
  if [ -d "${BACKEND_ROOT}/deps/cudnn/nvidia/cudnn/lib" ]; then
    note "dedicated cuDNN already present"
  else
    say "installing a dedicated cuDNN 9 for GNINA (~1.3 GB)"
    "$(pybin prism)" -m pip install --no-deps \
      --target "${BACKEND_ROOT}/deps/cudnn" "nvidia-cudnn-cu12==9.*"
  fi

  say "writing the GNINA launcher"
  cat > "${bin}/gnina" <<'WRAPPER'
#!/usr/bin/env bash
# GNINA 1.3.3 launcher. Supplies the one library the "static" binary still needs
# (libcudnn.so.9) from a copy that belongs to GNINA alone.
# Point a cascade's `executable` at THIS file, not at the raw .static binary.
set -euo pipefail
_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for _d in "${_root}"/deps/cudnn/nvidia/*/lib; do
  [ -d "${_d}" ] && LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
done
export LD_LIBRARY_PATH
exec "${_root}/bin/gnina.cuda12.8.static" "$@"
WRAPPER
  chmod +x "${bin}/gnina"

  note "gnina: $("${bin}/gnina" --version | head -1)"
  note "GNINA is GPL-2.0-or-later: runs need --allow-copyleft"
}

# --- verification ------------------------------------------------------------
verify() {
  say "verifying"
  local py; py="$(pybin prism)"
  "${py}" -c 'import torch;print("  torch", torch.__version__, "| cuda", torch.cuda.is_available())'
  "${py}" -c 'import meeko, rdkit, admet_ai; print("  rdkit", rdkit.__version__, "| meeko + admet_ai import OK")'
  "${py}" -m molcascade doctor --allow-copyleft 2>&1 | head -30 || true
  cat <<EOF

  Isolated backends are addressed by absolute path, so 'doctor' -- which probes
  PATH -- will keep calling them unavailable. That is expected. The paths to put
  in a cascade are listed in envs/README.md.
EOF
}

# --- dispatch ----------------------------------------------------------------
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(prism karmadock unidock aizynth gnina verify)

for t in "${targets[@]}"; do
  case "${t}" in
    prism)     setup_prism ;;
    karmadock) setup_karmadock ;;
    unidock)   setup_unidock ;;
    aizynth)   setup_aizynth ;;
    boltz2)    setup_boltz2 ;;
    gnina)     setup_gnina ;;
    verify)    verify ;;
    *)         die "unknown target: ${t}" ;;
  esac
done

say "done"
