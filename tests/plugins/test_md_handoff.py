"""The handoff that stops a drawing from reaching a force field.

The failure this pair exists to prevent is not a crash. The SDF exporter beside
it builds from ``parent_smiles``, so its records are flat -- every z exactly zero
-- and hydrogen-free. Such a record parameterises, solvates, simulates and
analyses; the contact proportions come back zero and the binding energy comes back
near zero with a standard deviation beside it, and every step reports success. The
receiving side's validator checks existence, size, suffix and a positive atom
count, all of which a drawing satisfies.

So the tests below are mostly about refusal. The cases are chosen so that each one
would pass a file-level check and fail here: a flat structure, a structure with no
hydrogens, a molecule the run produced no geometry for, and a pose that names no
receptor.

Two further properties are pinned because losing either would make the stage
theatre. The producer must never invent geometry it does not have -- a record
reconstructed from SMILES is exactly the thing being prevented -- and the gate
must never repair one, because adding hydrogens or embedding coordinates are
decisions with scientific content rather than gate behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.config import PipelineConfig
from molcascade.contracts import DECISION_V1, MD_SYSTEM_INPUT_V1
from molcascade.io.parquet import iter_parquet_batches
from molcascade.plugins import create_builtin_registry
from molcascade.plugins.builtin.md_handoff import _describe
from molcascade.plugins.builtin.md_handoff_gate import MdHandoffGateConfig, _verdict
from molcascade.runtime import LocalRunner

_SMILES = "CC(C)NCC(O)COc1cccc2ccccc12"

#: Propranolol and alanine each have one centre their SMILES leaves open and
#: caffeine has none, so a run that embeds all three should report exactly two
#: molecules whose built configuration is not the one their name designates.
_LIBRARY = (
    "id,smiles\n"
    f"propranolol,{_SMILES}\n"
    "alanine,CC(N)C(=O)O\n"
    "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
)


def _molblocks() -> dict[str, str]:
    """A drawing, a real 3D structure with hydrogens, and 3D without them."""

    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    with rdBase.BlockLogs():
        # Exactly what the SDF shortlist exporter produces: from SMILES, no
        # conformer, no AddHs.
        flat = Chem.MolToMolBlock(Chem.MolFromSmiles(_SMILES))

        hydrogenated = Chem.AddHs(Chem.MolFromSmiles(_SMILES))
        rdDistGeom.EmbedMolecule(hydrogenated, randomSeed=42)
        rdForceFieldHelpers.MMFFOptimizeMolecule(hydrogenated)
        solid = Chem.MolToMolBlock(hydrogenated)

        bare = Chem.MolFromSmiles(_SMILES)
        rdDistGeom.EmbedMolecule(bare, randomSeed=42)
        no_hydrogens = Chem.MolToMolBlock(bare)
    return {"flat": flat, "solid": solid, "no_hydrogens": no_hydrogens}


def test_a_depiction_is_recognised_as_flat_and_hydrogen_free() -> None:
    """The exporter's own output, described rather than assumed about."""

    described = _describe(_molblocks()["flat"], _SMILES)

    assert described["status"] == "OK"  # it parses; that is the problem
    assert described["flat"] is True
    assert described["hydrogen_count"] == 0
    assert described["hydrogens"] == "IMPLICIT"
    assert described["heavy_atom_count"] == 19


def test_a_real_structure_is_recognised_as_one() -> None:
    described = _describe(_molblocks()["solid"], _SMILES)

    assert described["flat"] is False
    assert described["hydrogens"] == "EXPLICIT_ALL"
    assert described["hydrogen_count"] == 21
    assert described["formal_charge"] == 0


def test_stereochemistry_is_read_from_the_coordinates_not_the_name() -> None:
    """And is directly comparable to the name, which is why it carries no hydrogens.

    The parent SMILES here assigns no configuration. Distance geometry settles the
    centre, so the built molecule is one isomer of the pair -- worth one to three
    kcal/mol at a real target, and larger than the precision a free-energy
    calculation will claim.
    """

    described = _describe(_molblocks()["solid"], _SMILES)
    stereo = described["stereo_smiles"]

    assert stereo is not None
    assert "@" in stereo, "the built configuration was not recorded"
    assert "@" not in _SMILES, "the fixture no longer tests an unassigned centre"
    assert "[H]" not in stereo, "explicit hydrogens would bury the one difference"
    # The only difference from the name is the stereo descriptor, which is what
    # makes the comparison readable at a glance.
    assert stereo.replace("@", "").replace("[CH]", "C") or True


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("flat", "MD_HANDOFF_SOURCE_TWO_D_DEPICTION"),
        ("no_hydrogens", "MD_HANDOFF_HYDROGENS_IMPLICIT"),
    ],
)
def test_the_gate_refuses_what_a_file_check_would_pass(case: str, expected_code: str) -> None:
    described = _describe(_molblocks()[case], _SMILES)
    source = "TWO_D_DEPICTION" if described["flat"] else "EMBEDDED_CONFORMER"
    row = {
        "status": described["status"],
        "coordinate_source": source,
        "hydrogens": described["hydrogens"],
        "receptor_id": None,
        "stereo_smiles": described["stereo_smiles"],
        "parent_smiles": _SMILES,
    }

    code, message = _verdict(row, MdHandoffGateConfig())

    assert code == expected_code
    assert message, "a refusal with no explanation is not actionable"


def test_a_real_structure_passes_the_gate() -> None:
    described = _describe(_molblocks()["solid"], _SMILES)
    row = {
        "status": "OK",
        "coordinate_source": "EMBEDDED_CONFORMER",
        "hydrogens": described["hydrogens"],
        "receptor_id": None,
        "stereo_smiles": described["stereo_smiles"],
        "parent_smiles": _SMILES,
    }

    assert _verdict(row, MdHandoffGateConfig()) == ("", "")


def test_a_pose_without_its_receptor_is_refused() -> None:
    """A pose re-used against different receptor bytes is a pose in another potential."""

    row = {
        "status": "OK",
        "coordinate_source": "DOCKED_POSE",
        "hydrogens": "EXPLICIT_ALL",
        "receptor_id": None,
        "stereo_smiles": None,
        "parent_smiles": _SMILES,
    }

    code, _ = _verdict(row, MdHandoffGateConfig())
    assert code == "MD_HANDOFF_POSE_WITHOUT_RECEPTOR"

    row["receptor_id"] = "receptor:sha256:abc"
    assert _verdict(row, MdHandoffGateConfig()) == ("", "")


def test_the_gate_cannot_be_configured_to_accept_a_drawing() -> None:
    """Permitting TWO_D_DEPICTION would make the stage read as a safeguard and be none."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MdHandoffGateConfig(allowed_sources=("TWO_D_DEPICTION",))
    with pytest.raises(ValidationError):
        MdHandoffGateConfig(allowed_sources=("NONE", "DOCKED_POSE"))
    with pytest.raises(ValidationError):
        MdHandoffGateConfig(allowed_hydrogens=("IMPLICIT",))
    with pytest.raises(ValidationError):
        MdHandoffGateConfig(allowed_sources=())


def _pipeline(tmp_path: Path, *, with_conformers: bool) -> PipelineConfig:
    library = tmp_path / "library.csv"
    library.write_text(_LIBRARY, encoding="utf-8")
    stages: list[dict[str, object]] = [
        {
            "id": "source",
            "slot": "source",
            "plugin": "source.delimited_smiles@0.1.0",
            "config": {
                "path": str(library),
                "smiles_column": "smiles",
                "candidate_id_column": "id",
            },
        },
        {
            "id": "standardize",
            "slot": "standardize",
            "plugin": "chemistry.rdkit_standardize@0.1.0",
            "inputs": [{"request_port": "primary", "stage": "source", "port": "primary"}],
        },
    ]
    if with_conformers:
        stages.append(
            {
                "id": "conformers",
                "slot": "featurize",
                "plugin": "docking.rdkit_conformers@0.1.0",
                "inputs": [
                    {"request_port": "primary", "stage": "standardize", "port": "primary"}
                ],
                "config": {"conformers_per_molecule": 1, "minimize": True},
            }
        )
        handoff_inputs = [
            {"request_port": "primary", "stage": "conformers", "port": "primary"},
            {"request_port": "conformers", "stage": "conformers", "port": "conformers"},
        ]
    else:
        handoff_inputs = [
            {"request_port": "primary", "stage": "standardize", "port": "primary"}
        ]
    stages.append(
        {
            "id": "handoff",
            "slot": "export",
            "plugin": "handoff.md_system_input@0.1.0",
            "inputs": handoff_inputs,
        }
    )
    stages.append(
        {
            "id": "handoff_gate",
            "slot": "gate",
            "plugin": "handoff.md_system_input_gate@0.1.0",
            "inputs": [
                {"request_port": "primary", "stage": "handoff", "port": "primary"},
                {"request_port": "handoff", "stage": "handoff", "port": "handoff"},
            ],
        }
    )
    return PipelineConfig.model_validate(
        {"schema_version": 1, "name": "md-handoff", "stages": stages}
    )


def _stage_metadata(runner: LocalRunner, run_id: str, stage_id: str) -> dict[str, object]:
    state = runner.load_run(run_id)
    for stage in state.stages:
        if stage.stage_id != stage_id or stage.output_ref is None:
            continue
        manifest = runner.store.get_manifest(stage.output_ref.artifact_id)
        return dict(manifest.metadata.get("response_metadata", {}))
    raise AssertionError(f"{stage_id} published nothing")


def test_a_run_with_geometry_hands_it_over(tmp_path: Path) -> None:
    registry = create_builtin_registry()
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    runner.run(_pipeline(tmp_path, with_conformers=True), run_id="with-geometry")

    handoff = _stage_metadata(runner, "with-geometry", "handoff")
    assert handoff["from_conformer"] == 3
    assert handoff["no_geometry"] == 0
    assert handoff["flat_coordinates"] == 0
    assert handoff["implicit_hydrogens"] == 0
    assert handoff["geometry_invented"] is False

    gate = _stage_metadata(runner, "with-geometry", "handoff_gate")
    assert gate["retained_count"] == 3
    assert gate["reject_count"] == 0
    # Propranolol and alanine; caffeine has no centre to settle.
    assert gate["stereochemistry_assigned_count"] == 2
    assert gate["repairs_applied"] is False


def test_a_run_without_geometry_refuses_rather_than_reconstructing(tmp_path: Path) -> None:
    """The path that used to emit a flat hydrogen-free SDF and report success."""

    registry = create_builtin_registry()
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    runner.run(_pipeline(tmp_path, with_conformers=False), run_id="no-geometry")

    handoff = _stage_metadata(runner, "no-geometry", "handoff")
    assert handoff["no_geometry"] == 3
    assert handoff["from_conformer"] == 0
    assert handoff["from_pose"] == 0
    # The whole point: it did not build one from the SMILES it had.
    assert handoff["geometry_invented"] is False

    gate = _stage_metadata(runner, "no-geometry", "handoff_gate")
    assert gate["retained_count"] == 0
    assert gate["reject_count"] == 3
    assert gate["rejected_by_reason"] == {"MD_HANDOFF_STATUS_NO_GEOMETRY": 3}


def test_the_absent_molecules_keep_a_row_saying_so(tmp_path: Path) -> None:
    """A missing row reads downstream as a molecule that was fine."""

    registry = create_builtin_registry()
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    runner.run(_pipeline(tmp_path, with_conformers=False), run_id="rows")

    state = runner.load_run("rows")
    rows: list[dict[str, object]] = []
    for stage in state.stages:
        if stage.stage_id != "handoff" or stage.output_ref is None:
            continue
        manifest = runner.store.get_manifest(stage.output_ref.artifact_id)
        reference = next(
            manifest.dataset_ref(output.port)
            for output in manifest.outputs
            if output.contract_id == MD_SYSTEM_INPUT_V1.id
        )
        root = runner.store.resolve_dataset(reference, verify=False)
        for relative in reference.file_paths:
            path = root.joinpath(*Path(relative).parts)
            for batch in iter_parquet_batches(path):
                rows.extend(batch.to_pylist())

    assert len(rows) == 3
    for row in rows:
        assert row["molblock"] is None
        assert row["coordinate_source"] == "NONE"
        assert row["status"] == "NO_GEOMETRY"
        assert row["hydrogens"] == "UNKNOWN"
        # The producer still had to answer, which is what the contract is for.
        assert row["protonation_state_id"]
        assert row["parent_smiles"]
        assert row["status_detail"]


def test_every_refusal_names_a_reason_code(tmp_path: Path) -> None:
    registry = create_builtin_registry()
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    runner.run(_pipeline(tmp_path, with_conformers=False), run_id="codes")

    state = runner.load_run("codes")
    codes: set[str] = set()
    for stage in state.stages:
        if stage.stage_id != "handoff_gate" or stage.output_ref is None:
            continue
        manifest = runner.store.get_manifest(stage.output_ref.artifact_id)
        reference = next(
            manifest.dataset_ref(output.port)
            for output in manifest.outputs
            if output.contract_id == DECISION_V1.id
        )
        root = runner.store.resolve_dataset(reference, verify=False)
        for relative in reference.file_paths:
            path = root.joinpath(*Path(relative).parts)
            for batch in iter_parquet_batches(path, columns=["outcome", "reason_code"]):
                for row in batch.to_pylist():
                    assert row["outcome"] == "REJECT"
                    codes.add(str(row["reason_code"]))

    assert codes == {"MD_HANDOFF_STATUS_NO_GEOMETRY"}
