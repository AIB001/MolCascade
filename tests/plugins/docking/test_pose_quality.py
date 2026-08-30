"""Pose repair and validation, tested on geometry rather than on a fake engine.

The other tests in this directory stub the docking engines out, because what is
under test there is a process boundary.  Here the geometry *is* the subject, so
nothing is faked: a real receptor is written to disk, a real conformer is
embedded with ETKDG, and the pose is deliberately placed inside the protein so
that a repair which does nothing and a repair which works give visibly different
answers.

The receptor is synthetic but not arbitrary.  It is a grid of carbons spaced far
enough apart to be a wall rather than a lump, which is what a ligand pressed into
a protein surface actually encounters, and it is written in the column layout
``MolFromPDBFile`` requires -- a receptor RDKit cannot parse is a separate test.

What is *not* tested here is whether PoseBusters' thresholds are the right ones.
That is settled by its paper (Chem Sci 2024, DOI 10.1039/D3SC04185A) and by the
measurements in the module docstring, and re-deriving it from three atoms in a
test would only be re-asserting the fixture.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from molcascade.errors import PluginError
from molcascade.plugins.builtin.docking.pose_quality import (
    AVAILABLE_CHECKS,
    DEFAULT_CHECKS,
    OPTIONAL_CHECKS,
    PoseQualityConfig,
    PoseQualitySession,
    apply_pose_quality,
    load_receptor_field,
    pose_quality_metadata,
    relax_pose,
)

# PoseBusters ships in the `docking` extra alongside meeko. The repair needs only
# RDKit, so the tests about pushing a pose out of a protein stay armed either way;
# the ones that ask for a verdict say so.
_needs_posebusters = pytest.mark.skipif(
    importlib.util.find_spec("posebusters") is None,
    reason="posebusters is an optional docking extra and is not installed",
)


def _receptor_pdb(path: Path, *, spacing: float = 4.0, count: int = 4) -> Path:
    """A wall of carbons in real PDB columns, centred on the origin."""

    offset = -0.5 * spacing * (count - 1)
    lines = []
    serial = 0
    for i in range(count):
        for j in range(count):
            serial += 1
            x = offset + i * spacing
            y = offset + j * spacing
            # z fixed: a plane, so "outside the protein" is a direction the
            # ligand can actually be pushed in rather than a cavity it is
            # trapped in.
            lines.append(
                f"ATOM  {serial:>5}  CA  ALA A{serial:>4}    "
                f"{x:>8.3f}{y:>8.3f}{0.0:>8.3f}  1.00  0.00           C"
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pose_molblock(*, z: float) -> str:
    """Benzene, hydrogens explicit, its centroid ``z`` angstrom off the wall."""

    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    molecule = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = 20_260_827
    assert rdDistGeom.EmbedMolecule(molecule, parameters) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule)
    conformer = molecule.GetConformer()
    positions = conformer.GetPositions()
    centre = positions.mean(axis=0)
    for index in range(molecule.GetNumAtoms()):
        x, y, old_z = positions[index]
        conformer.SetAtomPosition(
            index, (float(x - centre[0]), float(y - centre[1]), float(old_z - centre[2] + z))
        )
    return str(Chem.MolToMolBlock(molecule))


def _row(molblock: str | None, *, parent: str = "m1", rank: int = 0) -> dict[str, Any]:
    """One ``docking_score/v1`` row, as an engine's shard worker builds it."""

    return {
        "parent_id": parent,
        "engine_id": "karmadock",
        "receptor_id": "f" * 64,
        "pose_rank": rank,
        "score": 44.0,
        "score_kind": "KARMADOCK_MDN",
        "direction": "HIGHER_STRONGER",
        "secondary_score": None,
        "pose_molblock": molblock,
        "method_id": "docking-karmadock:sha256:" + "0" * 64,
    }


def _closest_contact(molblock: str, receptor: Path) -> float:
    """Nearest ligand-heavy-atom to receptor-heavy-atom distance, in angstrom.

    The heavy-atom filter is applied by atomic number rather than by asking the
    parser for it: ``removeHs=True`` is silently ignored when ``sanitize=False``,
    which on this project's reference ligand left 19 of 45 atoms in place and
    read the contact as 1.96 A instead of 2.88 A.  Distances that quietly include
    hydrogens are not comparable to the published thresholds, all of which are
    heavy-atom.
    """

    import numpy as np
    from rdkit import Chem

    molecule = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False)
    assert molecule is not None
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    assert heavy, "a pose with no heavy atoms is not a pose"
    field = load_receptor_field(str(receptor), ignore_waters=True)
    ligand = molecule.GetConformer().GetPositions()[heavy]
    deltas = ligand[:, None, :] - np.asarray(field.coordinates)[None, :, :]
    return float(np.sqrt((deltas**2).sum(axis=-1)).min())


def test_a_pose_pressed_into_the_protein_is_pushed_out_of_it(tmp_path: Path) -> None:
    """The defect this module exists for, reproduced and then repaired.

    The measured failure on real KarmaDock output was placement, not internal
    geometry: 22.7% of shortlisted poses cleared the protein's van der Waals
    surface.  A pose sitting on the wall reproduces that, and the assertion is
    the one that matters in a screen -- the contact distance improves, and it
    improves without the ligand being sent somewhere else entirely.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    molblock = _pose_molblock(z=1.2)
    before = _closest_contact(molblock, receptor)
    assert before < 2.0, "fixture is meant to start inside the protein"

    rows, report = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(checks=()),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    kept = rows[0]["pose_molblock"]
    assert isinstance(kept, str)
    after = _closest_contact(kept, receptor)
    assert after - before > 0.5
    # Against this carbon wall the repair converges on 2.72 A from any starting
    # overlap -- 1.63, 1.86 and 2.13 A all land there -- because 2.72 is exactly
    # the flat bottom the repulsion asks for, 0.80 * (1.7 + 1.7).  Asserting the
    # designed target rather than a round number means a change to the 0.80
    # factor shows up here as a failure instead of as a screen that silently
    # keeps closer contacts.
    assert after > 2.7, f"still clashing after repair: {after:.2f} A"
    assert report.relaxed == 1
    assert report.relax_failed == 0


def test_the_pose_is_repaired_rather_than_re_docked(tmp_path: Path) -> None:
    """Displacement stays bounded, so the engine's answer survives the repair.

    A pose free to travel is one being re-docked by a force field that cannot
    dock, and the score would then be about a geometry nothing computed it on.
    The position restraints are what keep this a repair; this is the test that
    would notice if they stopped being applied.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    molblock = _pose_molblock(z=1.2)

    import numpy as np
    from rdkit import Chem

    molecule = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=True)
    assert molecule is not None
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    before = molecule.GetConformer().GetPositions().copy()

    moved = relax_pose(
        molecule,
        receptor=load_receptor_field(str(receptor), ignore_waters=True),
        config=PoseQualityConfig(),
    )
    assert moved is not None

    after = molecule.GetConformer().GetPositions()
    per_atom = np.sqrt(((after - before) ** 2).sum(axis=-1))[heavy]
    # The return value is the median over heavy atoms -- hydrogens ride along
    # and would flatter it.  Pinning the definition here means a change to it
    # cannot pass silently while the metadata keeps the same name.
    assert moved == pytest.approx(float(np.median(per_atom)), abs=1e-9)
    assert float(per_atom.max()) < 3.0, "a repair this large is a re-dock"


def test_a_row_that_never_carried_a_pose_is_left_alone(tmp_path: Path) -> None:
    """A null molblock is the engine's own miss, already counted by the engine.

    Discarding it here would double-count a molecule that was never scored
    against a pose, and would make the funnel blame pose quality for a gap
    docking left.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    rows, report = apply_pose_quality(
        [_row(None, parent="m1"), _row(None, parent="m2")],
        config=PoseQualityConfig(),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert len(rows) == 2
    assert report.skipped == 2
    assert report.considered == 0
    assert report.discarded == 0
    assert report.metadata()["pose_unchecked_count"] == 2


def test_a_pose_that_will_not_parse_is_discarded(tmp_path: Path) -> None:
    """Text that failed to become a molecule here would fail downstream too."""

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    rows, report = apply_pose_quality(
        [_row("this is not a molblock", parent="m1")],
        config=PoseQualityConfig(),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert rows == []
    assert report.unparsable == 1
    assert report.discarded == 1


def test_switching_it_off_changes_nothing_at_all(tmp_path: Path) -> None:
    """The off-switch the operator is entitled to, held to being a no-op.

    On by default is only defensible if off is exact: an operator who wants the
    engine's raw output must get the engine's raw output, byte for byte.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    molblock = _pose_molblock(z=1.2)
    rows, report = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(enabled=False),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert rows[0]["pose_molblock"] == molblock
    assert report.considered == 0
    assert report.relaxed == 0
    metadata = pose_quality_metadata(
        lambda key: report.metadata().get(key, 0), config=PoseQualityConfig(enabled=False)
    )
    assert metadata["pose_quality_enabled"] is False
    assert metadata["pose_quality_checks"] == []


def test_a_pose_is_still_checked_when_the_operator_only_wants_scores(tmp_path: Path) -> None:
    """``keep_poses`` decides storage, not whether the geometry was examined.

    A scores-only run still wants its scores to be about structures that exist,
    so the pose is read and judged and only then dropped from the row.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    rows, report = apply_pose_quality(
        [_row(_pose_molblock(z=1.2))],
        config=PoseQualityConfig(checks=()),
        receptor_path=str(receptor),
        keep_poses=False,
    )

    assert rows[0]["pose_molblock"] is None
    assert report.considered == 1
    assert report.relaxed == 1


def test_discarding_a_best_pose_renumbers_the_ones_that_survive(tmp_path: Path) -> None:
    """The rank-zero trap: two consumers read the best pose as rank zero.

    ``handoff.py`` filters ``pose_rank == 0`` to build the best-pose view and
    ``trace.py`` does the same to pick the pose that enters the shortlist SDF.
    Leaving a gap where a discarded pose was would hide the molecule from both --
    the pose that passed would be the one nobody could see.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    good = _pose_molblock(z=1.2)
    rows, report = apply_pose_quality(
        [
            _row("not a molblock", parent="m1", rank=0),
            _row(good, parent="m1", rank=1),
            _row(good, parent="m2", rank=0),
        ],
        config=PoseQualityConfig(checks=()),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert report.discarded == 1
    assert [(row["parent_id"], row["pose_rank"]) for row in rows] == [("m1", 0), ("m2", 0)]


def test_an_unreadable_receptor_stops_the_shard_rather_than_skipping_the_check(
    tmp_path: Path,
) -> None:
    """Fail closed.  A check that quietly does nothing is worse than no check."""

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("REMARK this is not a structure\n", encoding="utf-8")

    with pytest.raises(PluginError) as raised:
        apply_pose_quality(
            [_row(_pose_molblock(z=1.2))],
            config=PoseQualityConfig(),
            receptor_path=str(receptor),
            keep_poses=True,
        )
    assert raised.value.code in {
        "DOCKING_POSE_RECEPTOR_UNREADABLE",
        "DOCKING_POSE_RECEPTOR_EMPTY",
    }


def test_a_misspelled_check_is_refused_when_the_config_is_read() -> None:
    """The silent-pass hazard, closed at configuration time.

    PoseBusters fills a column it was not asked for with ``NA``, and the
    reduction that decides whether a pose passed skips ``NA``.  So a misspelled
    check name would not fail -- it would pass every pose while appearing in the
    run report as enforced.  It has to be rejected before that can happen.
    """

    with pytest.raises(ValueError, match="unknown pose checks: minimum_distance"):
        PoseQualityConfig(checks=("minimum_distance",))


def test_the_checks_survive_the_round_trip_into_a_worker() -> None:
    """The config reaches a shard as a plain dict, so a list has to be accepted."""

    config = PoseQualityConfig.model_validate(
        {"checks": ["bond_lengths", "bond_lengths", "no_radicals"]}
    )
    assert config.checks == ("bond_lengths", "no_radicals")
    again = PoseQualityConfig.model_validate(config.model_dump(mode="json"))
    assert again.identity() == config.identity()


def test_every_documented_check_is_one_the_module_can_produce() -> None:
    """The two published lists are subsets of what is wired up, not aspirations."""

    assert set(DEFAULT_CHECKS) <= set(AVAILABLE_CHECKS)
    assert set(OPTIONAL_CHECKS) <= set(AVAILABLE_CHECKS)
    assert set(DEFAULT_CHECKS).isdisjoint(OPTIONAL_CHECKS)
    assert set(DEFAULT_CHECKS) | set(OPTIONAL_CHECKS) == set(AVAILABLE_CHECKS)


def test_repairing_the_same_pose_twice_gives_the_same_answer(tmp_path: Path) -> None:
    """Determinism, because the descriptor claims it.

    Nothing here is seeded -- the minimisation starts from the engine's own
    coordinates -- so this is a check that no unordered iteration crept into the
    neighbour search.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    molblock = _pose_molblock(z=1.2)
    first, _ = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(checks=()),
        receptor_path=str(receptor),
        keep_poses=True,
    )
    second, _ = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(checks=()),
        receptor_path=str(receptor),
        keep_poses=True,
    )
    assert first[0]["pose_molblock"] == second[0]["pose_molblock"]


def test_one_session_serves_many_batches(tmp_path: Path) -> None:
    """Two engines emit rows batch by batch, so the counters have to accumulate."""

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    session = PoseQualitySession(PoseQualityConfig(checks=()), receptor_path=str(receptor))
    for parent in ("m1", "m2", "m3"):
        kept = session.apply([_row(_pose_molblock(z=1.2), parent=parent)], keep_poses=True)
        assert len(kept) == 1
    assert session.report.considered == 3
    assert session.report.relaxed == 3


@_needs_posebusters
def test_a_pose_inside_the_protein_fails_the_check_that_names_it(tmp_path: Path) -> None:
    """The verdict has to disagree with the score, or the check adds nothing.

    On the real shortlist the correlation between KarmaDock's score and the
    closest protein contact was +0.014, which is why this is a separate
    judgement at all.  Here the same score is attached to a clashing pose and a
    clear one, and only the geometry decides.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    config = PoseQualityConfig(relax=False, checks=("minimum_distance_to_protein",))

    clashing, clash_report = apply_pose_quality(
        [_row(_pose_molblock(z=1.2))],
        config=config,
        receptor_path=str(receptor),
        keep_poses=True,
    )
    clear, clear_report = apply_pose_quality(
        [_row(_pose_molblock(z=8.0))],
        config=config,
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert clashing == []
    assert clash_report.failures == {"minimum_distance_to_protein": 1}
    assert len(clear) == 1
    assert clear_report.passed == 1
    assert clear_report.failures == {}


@_needs_posebusters
def test_repairing_first_is_what_makes_the_pose_pass(tmp_path: Path) -> None:
    """The measured 18.0% -> 98.6%, in miniature.

    Rejecting on the raw pose would have thrown away 82% of the real shortlist
    to pay for a bias in the pose writer rather than in the chemistry.  This is
    the test that would fail if repair were ever moved after the judgement.
    """

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    checks = ("minimum_distance_to_protein", "internal_steric_clash", "bond_lengths")
    molblock = _pose_molblock(z=1.2)

    without, _ = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(relax=False, checks=checks),
        receptor_path=str(receptor),
        keep_poses=True,
    )
    with_repair, report = apply_pose_quality(
        [_row(molblock)],
        config=PoseQualityConfig(relax=True, checks=checks),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert without == []
    assert len(with_repair) == 1
    assert report.passed == 1


@_needs_posebusters
def test_counting_without_discarding_measures_a_threshold_before_enforcing_it(
    tmp_path: Path,
) -> None:
    """``discard_failing: false`` is how an operator sees the cost of a check."""

    receptor = _receptor_pdb(tmp_path / "receptor.pdb")
    rows, report = apply_pose_quality(
        [_row(_pose_molblock(z=1.2))],
        config=PoseQualityConfig(
            relax=False, discard_failing=False, checks=("minimum_distance_to_protein",)
        ),
        receptor_path=str(receptor),
        keep_poses=True,
    )

    assert len(rows) == 1
    assert report.discarded == 0
    assert report.metadata()["pose_check_failed__minimum_distance_to_protein"] == 1


@_needs_posebusters
def test_only_the_requested_checks_are_computed(tmp_path: Path) -> None:
    """Cost control, and the reason the config is a dict rather than a preset.

    The full ``dock`` set measured 351 ms a molecule against 106 for the nine
    defaults, and the difference is entirely modules nobody asked for.
    """

    from rdkit import Chem

    from molcascade.plugins.builtin.docking.pose_quality import judge_poses, posebusters_config

    receptor_path = _receptor_pdb(tmp_path / "receptor.pdb")
    receptor = Chem.MolFromPDBFile(
        str(receptor_path), removeHs=False, sanitize=False, proximityBonding=False
    )
    molecule = Chem.MolFromMolBlock(_pose_molblock(z=8.0), removeHs=False, sanitize=True)
    assert receptor is not None and molecule is not None

    checks = ("bond_lengths", "no_radicals")
    config = posebusters_config(checks)
    requested = {module["function"] for module in config["modules"]}
    assert requested == {"distance_geometry", "check_radicals"}, requested
    assert "energy_ratio" not in requested, (
        "internal_energy is off by default: it cost 102 ms/mol on top of the "
        "defaults and failed outright on a fused polycyclic scaffold"
    )

    verdicts = judge_poses([molecule], receptor=receptor, checks=checks)
    assert set(verdicts[0]) == set(checks)
