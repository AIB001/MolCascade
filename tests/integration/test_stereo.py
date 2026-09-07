"""Reading back the stereochemistry 3D embedding chose on a molecule's behalf.

``enforceChirality=True`` constrains the centres a SMILES already assigns and
says nothing about the ones it leaves open; distance geometry settles those, and
the molecule that reaches a docking engine is one specific stereoisomer of the
set its ``parent_smiles`` names.

A reader of this shape has two ways to be useless. It can report a finding for
a molecule whose stereochemistry was fully specified -- then every run is all
findings and none of them mean anything. Or it can miss the case it exists for,
which is why the library below pairs molecules with open centres against a
control whose centres are all written down.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from molcascade.config import PipelineConfig
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner
from molcascade.stereo import (
    STEREO_AGREES,
    STEREO_ASSIGNED,
    reconcile_stereo,
)

#: Alanine and ibuprofen each have one unassigned centre, threonine has two, and
#: menthol has three that are all written down. Benzene has none at all. So the
#: expected split is exactly three findings and two silent passes, and any
#: off-by-one in the descriptor count shows up as the wrong molecule.
_LIBRARY = (
    "id,smiles\n"
    "alanine,CC(N)C(=O)O\n"
    "threonine,CC(O)C(N)C(=O)O\n"
    "ibuprofen,CC(C)Cc1ccc(C(C)C(=O)O)cc1\n"
    "menthol,C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1\n"
    "benzene,c1ccccc1\n"
)


def _run(tmp_path: Path, *, run_id: str = "stereo-run") -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "stereo-test",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(source),
                        "smiles_column": "smiles",
                        "candidate_id_column": "id",
                    },
                },
                {
                    "id": "standardize",
                    "slot": "standardize",
                    "plugin": "chemistry.rdkit_standardize@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "source", "port": "primary"}
                    ],
                },
                {
                    "id": "conformers",
                    "slot": "featurize",
                    "plugin": "docking.rdkit_conformers@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "standardize", "port": "primary"}
                    ],
                    "config": {"conformers_per_molecule": 1, "minimize": True},
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    return runner, runner.run(config, run_id=run_id).run_id


def _by_smiles(summary) -> dict[str, str]:
    """Findings keyed by the parent SMILES, which is what a reader recognises."""

    return {f.parent_smiles: f.status for f in summary.findings}


def test_open_centres_are_reported_and_written_ones_are_not(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    summary = reconcile_stereo(runner, run_id)

    assert summary.structures_read == 5
    assert summary.counts[STEREO_ASSIGNED] == 3
    assert summary.counts[STEREO_AGREES] == 2
    assert summary.assigned_parents == 3

    reported = _by_smiles(summary)
    # Menthol and benzene must be silent: one has every centre specified, the
    # other has none to specify.
    assert not any("C1CC" in smiles and "@" in smiles for smiles in reported)


def test_the_number_of_centres_the_geometry_added_is_counted(tmp_path: Path) -> None:
    """Threonine has two open centres and must be reported as two, not one."""

    runner, run_id = _run(tmp_path)
    summary = reconcile_stereo(runner, run_id)

    added = {f.parent_smiles: f.added_centres for f in summary.findings}
    threonine = next(s for s in added if s.count("O") == 3 and "N" in s)
    assert added[threonine] == 2
    assert all(count >= 1 for count in added.values())


def test_the_assigned_structure_is_a_real_stereoisomer_of_the_parent(
    tmp_path: Path,
) -> None:
    """The reported SMILES must name the same molecule, only more precisely.

    If the comparison ever reported a structure that is not a stereoisomer of
    the parent -- a different molecule, a lost atom, a re-protonated form -- the
    finding would be about the reader rather than about the chemistry.
    """

    runner, run_id = _run(tmp_path)
    summary = reconcile_stereo(runner, run_id)

    for finding in summary.findings:
        assert finding.status == STEREO_ASSIGNED
        assert finding.assigned_smiles is not None
        stripped = Chem.MolToSmiles(
            Chem.MolFromSmiles(finding.assigned_smiles), isomericSmiles=False
        )
        parent = Chem.MolToSmiles(
            Chem.MolFromSmiles(finding.parent_smiles), isomericSmiles=False
        )
        assert stripped == parent


def test_a_run_without_geometry_says_so_rather_than_reporting_nothing(
    tmp_path: Path,
) -> None:
    """Silence and "there was nothing to look at" are different answers."""

    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "no-geometry",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {"path": str(source), "smiles_column": "smiles"},
                },
                {
                    "id": "standardize",
                    "slot": "standardize",
                    "plugin": "chemistry.rdkit_standardize@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "source", "port": "primary"}
                    ],
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    run_id = runner.run(config, run_id="flat-run").run_id

    summary = reconcile_stereo(runner, run_id)
    assert summary.datasets_read == 0
    assert summary.structures_read == 0
    assert any("no 3D geometry" in note for note in summary.notes)
    assert "no 3D geometry" in summary.render()
