"""Reviewed local software options. Catalogue membership performs no imports."""

from __future__ import annotations

from molcascade.backends.models import (
    BackendInterface as I,
)
from molcascade.backends.models import (
    BackendSpec,
)
from molcascade.backends.models import (
    BackendTier as T,
)
from molcascade.backends.models import (
    Capability as C,
)
from molcascade.backends.models import (
    LicenseClass as L,
)
from molcascade.backends.registry import BackendRegistry


def _python(
    backend_id: str,
    capability: C,
    name: str,
    tier: T,
    license_spdx: str,
    license_class: L,
    module: str,
    distribution: str,
    *,
    plugin_ref: str | None = None,
    extra: str | None = None,
    platforms: tuple[str, ...] = ("windows", "linux", "darwin"),
    notes: str | None = None,
) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        capability=capability,
        display_name=name,
        tier=tier,
        interface=I.PYTHON,
        license_spdx=license_spdx,
        license_class=license_class,
        module=module,
        distribution=distribution,
        plugin_ref=plugin_ref,
        extra=extra,
        platforms=platforms,
        notes=notes,
    )


BUILTIN_BACKEND_SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        id="python.delimited",
        capability=C.SOURCE,
        display_name="Python delimited reader",
        tier=T.DEFAULT,
        interface=I.NATIVE,
        license_spdx="Python-2.0",
        license_class=L.PERMISSIVE,
        plugin_ref="source.delimited_smiles@0.1.0",
        notes="Streaming CSV/TSV/SMI input; adapter plugin is packaged with MolCascade.",
    ),
    _python("arrow.parquet", C.SOURCE, "PyArrow Parquet", T.DEFAULT, "Apache-2.0", L.PERMISSIVE, "pyarrow", "pyarrow", plugin_ref="source.raw_molecule_parquet@0.1.0"),
    _python("rdkit.sdf", C.SOURCE, "RDKit SDF supplier", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="source.sdf@0.1.0"),
    _python("openpyxl.xlsx", C.SOURCE, "openpyxl streaming XLSX", T.FALLBACK, "MIT", L.PERMISSIVE, "openpyxl", "openpyxl", plugin_ref="source.xlsx@0.1.0"),
    _python("rdkit.mol2", C.SOURCE, "RDKit MOL2 directory parser", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="source.mol2_directory@0.1.0"),
    _python("rdkit.standardize", C.STANDARDIZE, "RDKit MolStandardize", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_standardize@0.1.0"),
    _python("chembl.standardize", C.STANDARDIZE, "ChEMBL Structure Pipeline standardizer", T.FALLBACK, "MIT", L.PERMISSIVE, "chembl_structure_pipeline", "chembl-structure-pipeline", notes="Reviewed; MolCascade registers identity through RDKit, so only the checker below is wired in."),
    _python("chembl.checker", C.GATE, "ChEMBL Structure Pipeline checker", T.DEFAULT, "MIT", L.PERMISSIVE, "chembl_structure_pipeline", "chembl-structure-pipeline", plugin_ref="chemistry.chembl_structure_check@0.1.0", extra="standardize", notes="EBI's curation checker on its published 0-9 penalty scale; the InChI verdicts are a second toolkit's opinion, not RDKit's."),
    _python("openbabel.rescue", C.STANDARDIZE, "Open Babel format rescue", T.ISOLATED, "GPL-2.0-only", L.COPYLEFT, "openbabel", "openbabel-wheel", notes="Rescued structures must be registered again by RDKit."),
    _python("rdkit.filtercatalog", C.GATE, "RDKit FilterCatalog", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_hard_gate@0.1.0"),
    _python("rdkit.smarts", C.GATE, "Versioned RDKit SMARTS rules", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_hard_gate@0.1.0"),
    _python("rdkit.property-range-gate", C.GATE, "RDKit physicochemical range gate", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_property_range_gate@0.1.0"),
    _python("rdkit.drug-likeness", C.GATE, "RDKit Rule of Five and QED", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_drug_likeness@0.1.0"),
    _python("rdkit.structural-alerts", C.GATE, "RDKit PAINS/BRENK/NIH/ZINC alerts", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="chemistry.rdkit_structural_alerts@0.1.0"),
    BackendSpec(id="native.decision-join", capability=C.GATE, display_name="MolCascade parallel decision policy join", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="policy.native_decision_join@0.1.0", notes="Explicit all/any/minimum-pass barrier for parallel decision branches."),
    BackendSpec(id="native.prediction-evidence-gate", capability=C.PREDICT, display_name="MolCascade prediction evidence threshold gate", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="prediction.numeric_evidence_gate@0.1.0", notes="Inclusive endpoint/model numeric window with complete fail-closed decisions."),
    BackendSpec(id="lilly.medchem", capability=C.GATE, display_name="Lilly Medchem Rules", tier=T.OPTIONAL, interface=I.CLI, license_spdx="Apache-2.0", license_class=L.PERMISSIVE, command=("Lilly_Medchem_Rules.py", "--help"), platforms=("linux",), notes="Use WSL/container on Windows."),
    _python("medchem.rules", C.GATE, "medchem published rule sets", T.OPTIONAL, "Apache-2.0", L.PERMISSIVE, "medchem", "medchem", plugin_ref="chemistry.medchem_rules@0.1.0", extra="alerts", notes="Twenty-two named rule sets combined in series or in parallel; unknown rule names stop the run."),
    _python("medchem.alerts", C.GATE, "medchem alert collections and NIBR rules", T.OPTIONAL, "Apache-2.0", L.PERMISSIVE, "medchem", "medchem", plugin_ref="chemistry.medchem_alerts@0.1.0", extra="alerts", notes="About 2,400 curated SMARTS across 23 collections; the run is pinned to the digest of the rule table that produced it."),
    _python("rdkit.descriptors", C.DESCRIPTORS, "RDKit descriptors", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="features.rdkit_properties@0.1.0"),
    _python("openbabel.descriptors", C.DESCRIPTORS, "Open Babel descriptors", T.FALLBACK, "GPL-2.0-only", L.COPYLEFT, "openbabel", "openbabel-wheel", plugin_ref="features.openbabel_properties@0.1.0"),
    _python("mordred.community", C.DESCRIPTORS, "MordredCommunity", T.OPTIONAL, "BSD-3-Clause", L.PERMISSIVE, "mordred", "mordredcommunity", plugin_ref="chemistry.mordred_descriptor_gate@0.1.0", extra="descriptors", notes="1,613 2D descriptors. Note the distribution is 'mordredcommunity': 'pip install mordred' fetches the abandoned 2018 release, which does not run against current RDKit."),
    _python("rdkit.morgan", C.FINGERPRINT, "RDKit Morgan/AtomPair fingerprints", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="features.rdkit_fingerprint@0.1.0"),
    _python("sklearn.baseline", C.PREDICT, "scikit-learn baseline", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "sklearn", "scikit-learn"),
    _python("onnxruntime.custom-model", C.PREDICT, "Your own model, as an ONNX bundle", T.OPTIONAL, "MIT", L.PERMISSIVE, "onnxruntime", "onnxruntime", plugin_ref="prediction.custom_model@0.1.0", extra="custom-model", notes="The reserved interface for a project's own featurizer and estimator. ONNX is a data format, so nothing from the bundle is executed as Python."),
    _python("xgboost", C.PREDICT, "XGBoost", T.OPTIONAL, "Apache-2.0", L.PERMISSIVE, "xgboost", "xgboost"),
    _python("lightgbm", C.PREDICT, "LightGBM", T.OPTIONAL, "MIT", L.PERMISSIVE, "lightgbm", "lightgbm"),
    _python("chemprop", C.PREDICT, "Chemprop", T.ISOLATED, "MIT", L.PERMISSIVE, "chemprop", "chemprop"),
    _python(
        "admet-ai",
        C.PREDICT,
        "ADMET-AI v2",
        T.OPTIONAL,
        "MIT",
        L.PERMISSIVE,
        "admet_ai",
        "admet-ai",
        plugin_ref="prediction.admet_ai_v2@0.1.0", extra="admet",
        notes=(
            "Executable local multi-endpoint adapter. Requires an exact model-tree "
            "digest and explicit trust for Torch checkpoint deserialization; no "
            "applicability-domain claim is inferred."
        ),
    ),
    _python("rdkit.sklearn-ad", C.APPLICABILITY, "RDKit Tanimoto + sklearn AD", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "sklearn", "scikit-learn"),
    _python("rdkit.reference-similarity", C.APPLICABILITY, "RDKit maximum lead similarity", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="applicability.rdkit_reference_similarity@0.1.0", notes="Exact bounded reference-set comparison for applicability, analogue, or novelty evidence."),
    _python("skfp.ad", C.APPLICABILITY, "scikit-fingerprints AD", T.OPTIONAL, "MIT", L.PERMISSIVE, "skfp", "scikit-fingerprints"),
    _python("rdkit.sa-score", C.SYNTHESIS, "RDKit SA Score", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="synthesis.rdkit_sa_score@0.1.0", notes="Fast fragment/complexity proxy; not a retrosynthesis route."),
    BackendSpec(id="native.scscore", capability=C.SYNTHESIS, display_name="SCScore (Coley reaction-trained complexity)", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="synthesis.scscore@0.1.0", notes="Evaluated in numpy, so nothing extra is installed; the published weight file is supplied by the user and pinned by digest."),
    BackendSpec(id="native.synthesis-evidence-gate", capability=C.SYNTHESIS, display_name="MolCascade synthesis-score threshold gate", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="synthesis.numeric_evidence_gate@0.1.0", notes="Inclusive exact-method window with declared direction and fail-closed missing evidence."),
    _python("rascore", C.SYNTHESIS, "RAscore", T.OPTIONAL, "MIT", L.PERMISSIVE, "RAscore", "RAscore", notes="Fast classifier learned from AiZynthFinder outcomes; chemical-space limits apply."),
    BackendSpec(id="aizynthfinder", capability=C.SYNTHESIS, display_name="AiZynthFinder", tier=T.ISOLATED, interface=I.CLI, license_spdx="MIT", license_class=L.PERMISSIVE, command=("aizynthcli", "--help"), notes="Route search is reserved for a reduced shortlist and requires explicit local policy/stock assets."),
    _python("rdkit.murcko", C.SCAFFOLD, "RDKit Murcko", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="scaffold.rdkit_murcko@0.1.0"),
    _python("scaffoldgraph", C.SCAFFOLD, "ScaffoldGraph", T.OPTIONAL, "MIT", L.PERMISSIVE, "scaffoldgraph", "scaffoldgraph"),
    _python("rdkit.leader", C.CLUSTER, "RDKit LeaderPicker", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit"),
    BackendSpec(id="python.bitset-leader", capability=C.CLUSTER, display_name="Python/RDKit streaming leader", tier=T.FALLBACK, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="cluster.native_streaming_leader@0.1.0", notes="O(N x leaders), O(leaders) memory; refuses unsafe cluster growth."),
    _python("rdkit.scaffold-groups", C.CLUSTER, "RDKit Murcko scaffold groups", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="cluster.rdkit_scaffold_groups@0.1.0"),
    _python("fpsim2", C.CLUSTER, "FPSim2 similarity index", T.OPTIONAL, "MIT", L.PERMISSIVE, "FPSim2", "FPSim2", plugin_ref="applicability.fpsim2_reference_similarity@0.1.0", extra="similarity", notes="Popcount-bounded nearest-reference search; the indexed counterpart to the exact RDKit backend, and worth switching to above ~1,000 references."),
    _python("bitbirch", C.CLUSTER, "BitBIRCH-Lean", T.ISOLATED, "GPL-3.0-only", L.COPYLEFT, "bblean", "bblean", platforms=("linux",), notes="Beta; prefer WSL/Linux subprocess."),
    BackendSpec(id="native.four-basket", capability=C.SELECT, display_name="MolCascade deterministic four-basket selector", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE),
    BackendSpec(id="native.hash-budget", capability=C.SELECT, display_name="MolCascade deterministic exploration budget", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="select.native_hash_budget@0.1.0"),
    BackendSpec(id="native.scaffold-round-robin", capability=C.SELECT, display_name="MolCascade scaffold round-robin", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="select.native_scaffold_round_robin@0.1.0"),
    _python("paretoset", C.SELECT, "paretoset", T.FALLBACK, "MIT", L.PERMISSIVE, "paretoset", "paretoset"),
    _python("pymoo", C.SELECT, "pymoo nondominated sorting", T.OPTIONAL, "Apache-2.0", L.PERMISSIVE, "pymoo", "pymoo"),
    _python("export.rdkit-sdf", C.EXPORT, "RDKit shortlist SDF export", T.DEFAULT, "BSD-3-Clause", L.PERMISSIVE, "rdkit", "rdkit", plugin_ref="export.rdkit_sdf_shortlist@0.1.0"),
    BackendSpec(id="export.parquet-manifest", capability=C.EXPORT, display_name="Parquet/SMILES shortlist manifest", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE, plugin_ref="export.native_smiles_shortlist@0.1.0"),
    BackendSpec(id="report.native-html", capability=C.REPORT, display_name="MolCascade self-contained HTML", tier=T.DEFAULT, interface=I.NATIVE, license_spdx="MIT", license_class=L.PERMISSIVE),
)


def create_backend_registry() -> BackendRegistry:
    return BackendRegistry(BUILTIN_BACKEND_SPECS)


#: Import name -> the distribution to ``pip install`` for it.
#:
#: These are not the same string often enough to matter, and one of the
#: divergences is a trap rather than a nuisance: ``import mordred`` is satisfied
#: by ``mordredcommunity``, while ``pip install mordred`` fetches the abandoned
#: 2018 original, which does not run against a current RDKit.  A user told only
#: the import name will type the wrong command and get a broken install that
#: still *looks* like the right package.
#:
#: Derived from the specs rather than written out a second time, so a backend
#: cannot be added with its distribution recorded in one place and missing from
#: the other.
_DISTRIBUTION_BY_MODULE: dict[str, str] = {
    spec.module: spec.distribution
    for spec in BUILTIN_BACKEND_SPECS
    if spec.module and spec.distribution
}


def distribution_for_module(module: str) -> str | None:
    """The pip distribution providing ``module``, or ``None`` if unrecorded."""

    return _DISTRIBUTION_BY_MODULE.get(module)


__all__ = [
    "BUILTIN_BACKEND_SPECS",
    "create_backend_registry",
    "distribution_for_module",
]
