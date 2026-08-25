"""The geometry three docking engines will share.

A consensus between Uni-Dock, KarmaDock and GNINA only means something if the
three docked the same coordinates, so the property this stage owes the level is
narrow and absolute: the conformer a molecule gets must depend on the molecule
and the configured seed, and on nothing else -- not on which shard it landed in,
not on how many workers were free that afternoon.  The first two tests are that
claim.

The rest are about the molecules that do not cooperate.  Embedding fails for
strained cages, MMFF has no parameters for a platinum complex, and both must end
as a labelled row rather than as a quietly shorter population: the funnel reports
what each stage rejected, and a preparation step that dropped molecules would
make that report a lie.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import LIGAND_CONFORMER_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.docking import RDKitLigandConformerPlugin

#: Ordinary drug-like matter: an alcohol, an aromatic, caffeine, aspirin, an
#: amine, a sugar, an aryl halide, a long-chain acid, a flexible ester.
_SMILES = (
    "CCO",
    "c1ccccc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCN(CC)CC",
    "OCC1OC(O)C(O)C(O)C1O",
    "Clc1ccccc1",
    "CCCCCCCCCCCCCCCCCC(=O)O",
    "CCOC(=O)CCCCN",
)

#: An anti-Bredt bridgehead alkene.  RDKit parses it happily and then cannot
#: build a distance geometry for it, which is exactly the failure this stage has
#: to survive without shrinking the population.
_UNEMBEDDABLE = "C1=C2CCC1CC2"

#: MMFF94s has no parameters for platinum.  RDKit signals that by returning
#: ``(-1, -1.0)`` from the optimiser rather than by raising, so a naive reader
#: records a converged -1 kcal/mol.
_NO_FORCE_FIELD = "[Pt](Cl)(Cl)(N)N"


def _row(index: int, smiles: str) -> dict[str, object]:
    return {
        "parent_id": f"parent-{index:03d}",
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{index:03d}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(root: Path, smiles: tuple[str, ...]) -> StageInput:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_row(index, value) for index, value in enumerate(smiles)],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
        # Small row groups, so a shard boundary genuinely falls inside one.
        row_group_size=2,
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "c" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(
    tmp_path: Path,
    name: str,
    *,
    smiles: tuple[str, ...] = _SMILES,
    resources: StageResources | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Run the stage and return its conformers, its parents, and its metadata."""

    staging = tmp_path / name
    response = RDKitLigandConformerPlugin().execute(
        StageRequest(
            stage_id="ligand_prep",
            inputs={"parents": _input(tmp_path / f"{name}-input", smiles)},
            config=config or {},
        ),
        StageContext(staging, resources or StageResources(shard_rows=1_000)),
    )
    conformers = pa.concat_tables(
        [pq.read_table(staging / path) for path in response.outputs["conformers"].file_paths]
    )
    parents = pa.concat_tables(
        [pq.read_table(staging / path) for path in response.outputs["primary"].file_paths]
    )
    LIGAND_CONFORMER_V1.validate(conformers)
    PARENT_V1.validate(parents)
    return (
        conformers.to_pylist(),
        [str(row["parent_id"]) for row in parents.to_pylist()],
        dict(response.metadata),
    )


def test_every_parent_leaves_with_three_dimensional_geometry(tmp_path: Path) -> None:
    from rdkit import Chem

    rows, parents, metadata = _run(tmp_path, "basic")

    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert len(rows) == len(_SMILES)
    assert metadata["embed_failed_count"] == 0
    assert metadata["conformer_count"] == len(_SMILES)

    for row in rows:
        assert row["conformer_index"] == 0
        assert row["embed_status"] == "MINIMIZED"
        assert row["method_id"] == metadata["method_id"]
        molecule = Chem.MolFromMolBlock(str(row["molblock"]), removeHs=False)
        assert molecule is not None, row["parent_id"]
        conformer = molecule.GetConformer()
        assert conformer.Is3D()
        # Hydrogens are what a docking engine scores against; a molblock that
        # lost them would silently change every pose.
        assert any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms())
        assert any(
            abs(conformer.GetAtomPosition(index).z) > 1e-6
            for index in range(molecule.GetNumAtoms())
        )


def test_geometry_does_not_move_when_the_work_is_split(tmp_path: Path) -> None:
    """The claim the whole docking level rests on.

    Seeding from a row index would have given a molecule different coordinates
    depending on which shard it landed in, so the same library would embed
    differently after a resume and two engines reading two artifacts would be
    docking two different molecules.
    """

    serial, _, serial_metadata = _run(
        tmp_path,
        "serial",
        resources=StageResources(shard_rows=1_000),
    )
    split, _, split_metadata = _run(
        tmp_path,
        "split",
        resources=StageResources(workers=4, devices=("cpu",) * 4, shard_rows=2),
    )

    assert serial_metadata["shard_count"] == 1
    assert split_metadata["shard_count"] == 5
    assert [row["molblock"] for row in serial] == [row["molblock"] for row in split]
    assert [row["energy_kcal_mol"] for row in serial] == [row["energy_kcal_mol"] for row in split]
    assert serial_metadata["method_id"] == split_metadata["method_id"]


def test_a_molecule_that_cannot_be_embedded_keeps_its_row(tmp_path: Path) -> None:
    rows, parents, metadata = _run(
        tmp_path,
        "unembeddable",
        smiles=(_SMILES[0], _UNEMBEDDABLE, _SMILES[1]),
        config={"embed_attempts": 1},
    )

    # The population is untouched: rejecting is the score gate's job, in the
    # open, and this stage declares no filter.
    assert parents == ["parent-000", "parent-001", "parent-002"]
    assert len(rows) == 3
    failed = rows[1]
    assert failed["parent_id"] == "parent-001"
    assert failed["embed_status"] == "EMBED_FAILED"
    assert failed["molblock"] is None
    assert failed["energy_kcal_mol"] is None
    assert metadata["embed_failed_count"] == 1
    assert metadata["conformer_count"] == 2


def test_a_molecule_without_force_field_parameters_keeps_its_etkdg_geometry(
    tmp_path: Path,
) -> None:
    """MMFF's failure code is ``(-1, -1.0)``, not an exception.

    Read as an ordinary result it becomes a converged -1 kcal/mol -- a sentinel
    sitting in the energy column, indistinguishable from a real number, ranking
    ahead of most genuine conformers.
    """

    rows, _, metadata = _run(
        tmp_path,
        "no-force-field",
        smiles=(_NO_FORCE_FIELD,),
    )

    assert len(rows) == 1
    assert rows[0]["embed_status"] == "MINIMIZE_FAILED"
    assert rows[0]["molblock"] is not None
    assert rows[0]["energy_kcal_mol"] is None
    assert metadata["embed_failed_count"] == 0


def test_conformers_are_numbered_best_first(tmp_path: Path) -> None:
    """``pose_rank`` semantics downstream assume index 0 is the one to dock."""

    rows, _, _ = _run(
        tmp_path,
        "ensemble",
        smiles=("CCCCCCCCCCCCCCCCCC(=O)O",),
        config={"conformers_per_molecule": 8},
    )

    assert len(rows) > 1
    assert [row["conformer_index"] for row in rows] == list(range(len(rows)))
    energies = [row["energy_kcal_mol"] for row in rows]
    assert all(value is not None for value in energies)
    assert energies == sorted(energies)


def test_method_identity_ignores_how_the_work_was_split(tmp_path: Path) -> None:
    """Shard size and worker count change throughput, never a coordinate.

    ``method_id`` is what a docking score cites as the geometry it was computed
    on, so letting a machine detail into it would make two identical runs look
    like two different experiments.
    """

    _, _, serial = _run(tmp_path, "identity-serial", resources=StageResources(shard_rows=1_000))
    _, _, split = _run(
        tmp_path,
        "identity-split",
        resources=StageResources(workers=3, devices=("cpu",) * 3, shard_rows=2),
        config={"batch_size": 1},
    )
    _, _, reseeded = _run(tmp_path, "identity-seed", config={"seed": 7})

    assert serial["method_id"] == split["method_id"]
    assert reseeded["method_id"] != serial["method_id"]


def test_an_unparseable_parent_stops_the_stage(tmp_path: Path) -> None:
    """Standardisation ran tiers ago; a parent that will not parse here is a bug.

    Skipping it would hide a broken upstream stage behind a slightly smaller
    conformer table, which is the opposite of what the funnel report promises.
    """

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "invalid", smiles=(_SMILES[0], "not a molecule"))

    assert raised.value.code == "LIGAND_PREP_PARENT_INVALID"
    assert raised.value.context["parent_id"] == "parent-001"


def test_an_empty_population_stops_the_stage(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "empty", smiles=())

    assert raised.value.code == "LIGAND_PREP_EMPTY_INPUT"
