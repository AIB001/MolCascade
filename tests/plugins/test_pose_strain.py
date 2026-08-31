"""What the docked conformer costs the molecule to hold.

A docking engine will bend a ligand into any shape that scores, so a strong
score can be paid for with internal strain the molecule would never pay in
solution.  These tests pin down the measurement itself -- that a relaxed
conformer measures near zero and a twisted one measures more, in kcal/mol
against a freely embedded reference -- and then pin down the three ways it can
be wrong without looking wrong: measuring a pose the run rejected, measuring
nothing because the geometry was never stored, and reporting a failed reference
search as an excellent pose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DERIVED_METRIC_V1, DOCKING_SCORE_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.pose_strain import (
    PoseStrainBackend,
    PoseStrainConfig,
    PoseStrainPlugin,
)

pytest.importorskip("rdkit")

_SMILES = {
    "p1": "c1ccccc1",  # benzene: rigid, nothing to strain
    "p2": "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: an ester torsion worth 13 kcal/mol
    "p3": "Cc1ccccc1",  # toluene: one nearly free methyl rotation, so also zero
}
#: Ethanol, kept out of ``_SMILES`` because it does not measure zero.  Three
#: heavy atoms is small enough that the default 0.5 A pruning collapses its
#: whole reference ensemble to one conformer, and the one it keeps is the
#: higher of its two minima -- see the noise-band test below.
_SMALL = "CCO"
_RECEPTOR = "receptor:sha256:" + "c" * 64


def _minimised(smiles: str, *, seed: int = 7, conformers: int = 8) -> tuple[Any, int]:
    """A well-relaxed conformer, standing in for a pose a good engine returned."""

    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(rdDistGeom.EmbedMultipleConfs(molecule, numConfs=conformers, randomSeed=seed))
    results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
        molecule, mmffVariant="MMFF94s", maxIters=2_000
    )
    best = min(range(len(ids)), key=lambda index: results[index][1])
    return molecule, ids[best]


def _relaxed_molblock(smiles: str) -> str:
    from rdkit import Chem

    molecule, conformer_id = _minimised(smiles)
    return str(Chem.MolToMolBlock(molecule, confId=conformer_id))


def _twisted_molblock(smiles: str, *, degrees: float = 90.0) -> str:
    """The same conformer with one real torsion driven away from its minimum.

    The torsion is the first rotatable bond that has a heavy atom on both ends,
    so the distortion lands on a bond with a genuine preference rather than on a
    methyl group, whose rotation is nearly free and would measure as no strain
    at all.  For aspirin this is the ester C-O bond and +90 degrees measures
    about 13 kcal/mol.
    """

    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    molecule, conformer_id = _minimised(smiles)
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    for second, third in molecule.GetSubstructMatches(pattern):
        first = next(
            (
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(second).GetNeighbors()
                if atom.GetIdx() != third and atom.GetAtomicNum() > 1
            ),
            None,
        )
        fourth = next(
            (
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(third).GetNeighbors()
                if atom.GetIdx() != second and atom.GetAtomicNum() > 1
            ),
            None,
        )
        if first is None or fourth is None:
            continue
        twisted = Chem.Mol(molecule)
        conformer = twisted.GetConformer(conformer_id)
        current = rdMolTransforms.GetDihedralDeg(conformer, first, second, third, fourth)
        rdMolTransforms.SetDihedralDeg(conformer, first, second, third, fourth, current + degrees)
        return str(Chem.MolToMolBlock(twisted, confId=conformer_id))
    raise AssertionError(f"{smiles} has no rotatable bond between heavy atoms")


def _raw_molblock(smiles: str) -> str:
    """An embedded but unminimised conformer, for molecules MMFF cannot type."""

    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=7)
    return str(Chem.MolToMolBlock(molecule))


def _parent(parent_id: str, smiles: str | None = None) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles or _SMILES[parent_id],
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _pose(
    parent_id: str,
    score: float,
    *,
    molblock: str | None,
    pose_rank: int = 0,
    engine_id: str = "unidock",
    receptor_id: str = _RECEPTOR,
    score_kind: str = "VINA_KCAL_MOL",
    direction: str = "LOWER_STRONGER",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "engine_id": engine_id,
        "receptor_id": receptor_id,
        "pose_rank": pose_rank,
        "score": score,
        "score_kind": score_kind,
        "direction": direction,
        "secondary_score": None,
        "pose_molblock": molblock,
        "method_id": f"{engine_id}:test",
    }


def _dataset(
    tmp_path: Path,
    name: str,
    contract: Any,
    rows: list[dict[str, object]],
    *,
    artifact_hex: str,
) -> StageInput:
    root = tmp_path / name
    root.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=contract.schema), root / "part.parquet")
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + artifact_hex * 64,
            kind=contract.id,
        ),
        root=root,
        contract_id=contract.id,
    )


def _run(
    tmp_path: Path,
    poses: list[dict[str, object]],
    config: dict[str, object] | None = None,
    *,
    parents: list[dict[str, object]] | None = None,
    suffix: str = "",
) -> tuple[Path, Any]:
    settings: dict[str, object] = dict(config or {})
    output = tmp_path / f"out{suffix}"
    response = PoseStrainPlugin().execute(
        StageRequest(
            stage_id=f"pose_strain{suffix}",
            inputs={
                "parents": _dataset(
                    tmp_path,
                    f"parents{suffix}",
                    PARENT_V1,
                    parents or [_parent(key) for key in ("p1", "p2", "p3")],
                    artifact_hex="a",
                ),
                "evidence": _dataset(
                    tmp_path,
                    f"poses{suffix}",
                    DOCKING_SCORE_V1,
                    poses,
                    artifact_hex="b",
                ),
            },
            config=settings,
        ),
        StageContext(output),
    )
    return output, response


def _metrics(output: Path, response: Any) -> dict[str, dict[str, object]]:
    table = pq.read_table(output / response.outputs["derived_metrics"].file_paths[0])
    DERIVED_METRIC_V1.validate(table)
    return {str(row["parent_id"]): row for row in table.to_pylist()}


def test_it_measures_the_strain_of_every_pose_it_is_given(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [
            _pose("p1", -6.0, molblock=_relaxed_molblock(_SMILES["p1"])),
            _pose("p2", -8.5, molblock=_twisted_molblock(_SMILES["p2"])),
            _pose("p3", -3.2, molblock=_relaxed_molblock(_SMILES["p3"])),
        ],
    )
    rows = _metrics(output, response)
    assert set(rows) == {"p1", "p2", "p3"}
    for row in rows.values():
        assert row["metric_id"] == "pose_strain"
        assert row["units"] == "KCAL_PER_MOL"
        assert row["direction"] == "LOWER_BETTER"
        assert row["method_id"] == response.metadata["method_id"]
    # The twisted ester torsion measures about 13 kcal/mol.  The two relaxed
    # conformers reach the same energy as their own reference minimum, so they
    # land inside the band that decides OK rather than merely near it.
    assert rows["p2"]["status"] == "OK"
    assert float(rows["p2"]["value"]) > 3.0
    band = PoseStrainConfig().negative_tolerance_kcal_mol
    for parent_id in ("p1", "p3"):
        assert rows[parent_id]["status"] == "OK"
        assert abs(float(rows[parent_id]["value"])) <= band
    assert response.metadata["output_count"] == 3
    assert response.metadata["backend"] == "mmff94s_local"
    assert response.metadata["pose_policy"] == "BEST_POSE"


def test_a_twisted_pose_is_strained_and_a_relaxed_one_is_not(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [
            _pose("p2", -8.5, molblock=_relaxed_molblock(_SMILES["p2"])),
            _pose("p3", -3.2, molblock=_relaxed_molblock(_SMILES["p3"])),
        ],
        parents=[_parent("p2"), _parent("p3")],
        suffix="-relaxed",
    )
    relaxed = float(_metrics(output, response)["p2"]["value"])

    output, response = _run(
        tmp_path,
        [
            _pose("p2", -8.5, molblock=_twisted_molblock(_SMILES["p2"])),
            _pose("p3", -3.2, molblock=_relaxed_molblock(_SMILES["p3"])),
        ],
        parents=[_parent("p2"), _parent("p3")],
        suffix="-twisted",
    )
    twisted = float(_metrics(output, response)["p2"]["value"])

    assert abs(relaxed) < 0.5
    assert twisted > relaxed + 3.0


def test_it_measures_the_pose_the_run_accepted_not_the_first_one(tmp_path: Path) -> None:
    """The strongest pose is the one the docking gate kept, so it is the one measured.

    Rank 0 here is the relaxed conformer and rank 1 is the twisted one that
    scores better.  Reading ``pose_rank == 0`` would report almost no strain for
    a molecule the run accepted on a badly strained pose.
    """

    output, response = _run(
        tmp_path,
        [
            _pose("p2", -5.0, molblock=_relaxed_molblock(_SMILES["p2"]), pose_rank=0),
            _pose("p2", -9.5, molblock=_twisted_molblock(_SMILES["p2"]), pose_rank=1),
        ],
        parents=[_parent("p2")],
    )
    row = _metrics(output, response)["p2"]
    assert row["status"] == "OK"
    assert float(row["value"]) > 3.0
    source = json.loads(str(row["source_json"]))
    assert source["pose_rank"] == 1
    assert source["score"] == pytest.approx(-9.5)


def test_higher_stronger_evidence_takes_the_highest_score(tmp_path: Path) -> None:
    """Strain is defined on any engine's pose, so the sense comes from the row.

    A KarmaDock-style score is higher-is-stronger and is not in kcal/mol, which
    the size-normalised metrics refuse -- strain does not care, because it is
    measured on the geometry rather than on the score.  What it must get right
    is which pose that engine considered best.
    """

    output, response = _run(
        tmp_path,
        [
            _pose(
                "p2",
                2.0,
                molblock=_relaxed_molblock(_SMILES["p2"]),
                pose_rank=0,
                engine_id="karmadock",
                score_kind="KARMADOCK_MDN",
                direction="HIGHER_STRONGER",
            ),
            _pose(
                "p2",
                9.0,
                molblock=_twisted_molblock(_SMILES["p2"]),
                pose_rank=1,
                engine_id="karmadock",
                score_kind="KARMADOCK_MDN",
                direction="HIGHER_STRONGER",
            ),
        ],
        parents=[_parent("p2")],
    )
    row = _metrics(output, response)["p2"]
    assert float(row["value"]) > 3.0
    assert json.loads(str(row["source_json"]))["score"] == pytest.approx(9.0)
    assert response.metadata["engine_id"] == "karmadock"


def test_a_reference_search_that_failed_keeps_its_value_and_says_so(tmp_path: Path) -> None:
    """A negative strain is evidence about the measurement, not about the pose.

    Forced here by giving the reference search one conformer and one
    minimisation step against a pose that is already at its global minimum: the
    reference cannot reach the energy it is supposed to measure against.  The
    measured value is kept so the size of the failure is visible, and the status
    stops it being read as the best strain in the library.
    """

    output, response = _run(
        tmp_path,
        [_pose("p2", -8.5, molblock=_relaxed_molblock(_SMILES["p2"]))],
        {"reference_conformers": 1, "max_minimize_iterations": 1},
        parents=[_parent("p2")],
    )
    row = _metrics(output, response)["p2"]
    assert row["status"] == "OUT_OF_DOMAIN"
    assert float(row["value"]) < 0.0
    assert "reference conformer" in str(row["status_detail"])
    assert json.loads(str(row["source_json"]))["reference_conformers_embedded"] == 1
    assert response.metadata["out_of_domain_count"] == 1
    assert response.metadata["measured_count"] == 1


def test_the_band_and_not_the_sign_decides_what_a_negative_strain_means(
    tmp_path: Path,
) -> None:
    """Where the line sits is a setting, and moving it changes the reading only.

    On this project's shortlist, eight of the twenty-two negative strains were
    exactly -0.0: the restrained pose and the reference reached the same energy.
    Reading those as failures would flag the best geometry a docking run can
    produce, so the sign alone does not decide.

    Ethanol is the smallest honest example of the other kind.  Its reference
    ensemble has two minima 0.18 kcal/mol apart, and at the shipped 0.5 A
    pruning threshold three heavy atoms are not enough to tell them apart: all
    twelve embeddings collapse to one kept conformer, the higher one.  So a
    well-relaxed pose comes out 0.18 below its own reference -- a real
    shortfall, far too small to be a bent molecule, and reproducible without
    crippling the search the way the previous test does.  Under the default band
    it is reported; widen the band past it and the same measured number is read
    as zero.
    """

    poses = [_pose("small", -3.2, molblock=_relaxed_molblock(_SMALL))]
    parents = [_parent("small", _SMALL)]

    output, response = _run(tmp_path, poses, parents=parents, suffix="-strict")
    strict = _metrics(output, response)["small"]
    assert strict["status"] == "OUT_OF_DOMAIN"
    assert float(strict["value"]) == pytest.approx(-0.18, abs=0.02)
    assert response.metadata["out_of_domain_count"] == 1

    output, response = _run(
        tmp_path,
        poses,
        {"negative_tolerance_kcal_mol": 0.5},
        parents=parents,
        suffix="-wide",
    )
    wide = _metrics(output, response)["small"]
    assert wide["status"] == "OK"
    assert wide["status_detail"] is None
    assert float(wide["value"]) == pytest.approx(float(strict["value"]))
    assert response.metadata["out_of_domain_count"] == 0


def test_a_molecule_mmff_cannot_type_is_a_backend_failure(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [_pose("se", -7.0, molblock=_raw_molblock("CC[SeH]"))],
        parents=[_parent("se", "CC[SeH]")],
    )
    row = _metrics(output, response)["se"]
    assert row["status"] == "BACKEND_FAILED"
    assert row["value"] is None
    assert "MMFF94s" in str(row["status_detail"])
    assert row["source_json"] is None
    assert response.metadata["backend_failed_count"] == 1
    assert response.metadata["measured_count"] == 0


def test_a_pose_that_cannot_be_parsed_is_a_backend_failure(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [_pose("p3", -3.2, molblock="not a molblock")],
        parents=[_parent("p3")],
    )
    row = _metrics(output, response)["p3"]
    assert row["status"] == "BACKEND_FAILED"
    assert row["value"] is None
    assert "parsed" in str(row["status_detail"])


def test_a_parent_with_no_pose_is_not_applicable(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [
            _pose("p1", -6.0, molblock=_relaxed_molblock(_SMILES["p1"])),
            _pose("p2", -8.5, molblock=_twisted_molblock(_SMILES["p2"])),
        ],
    )
    rows = _metrics(output, response)
    assert rows["p3"]["status"] == "NOT_APPLICABLE"
    assert rows["p3"]["value"] is None
    assert "unidock" in str(rows["p3"]["status_detail"])
    assert response.metadata["unscored_count"] == 1
    assert response.metadata["output_count"] == 3


def test_the_accepted_pose_without_geometry_is_never_swapped_for_a_weaker_one(
    tmp_path: Path,
) -> None:
    """Only the pose that decided the molecule is measurable.

    Here the strongest pose was stored without geometry while a weaker one kept
    it.  Falling back would report the strain of a conformer the run rejected,
    which is a number about a different molecule's fate.
    """

    output, response = _run(
        tmp_path,
        [
            _pose("p2", -9.5, molblock=None, pose_rank=0),
            _pose("p2", -5.0, molblock=_relaxed_molblock(_SMILES["p2"]), pose_rank=1),
        ],
        parents=[_parent("p2")],
    )
    row = _metrics(output, response)["p2"]
    assert row["status"] == "NOT_APPLICABLE"
    assert row["value"] is None
    assert "pose_rank 0" in str(row["status_detail"])
    assert response.metadata["missing_geometry_count"] == 1
    assert response.metadata["measured_count"] == 0


def test_evidence_with_no_geometry_at_all_stops_the_stage(tmp_path: Path) -> None:
    """The likeliest misconfiguration, and it must not be discovered per molecule.

    ``pose_molblock`` is filled only when the docking stage was asked to keep
    poses, so a run that forgot would otherwise pay for every shard and then
    report a library-wide failure that reads like strained molecules.
    """

    with pytest.raises(PluginError) as error:
        _run(tmp_path, [_pose(key, -7.0, molblock=None) for key in ("p1", "p2", "p3")])
    assert error.value.code == "POSE_STRAIN_POSES_UNAVAILABLE"
    assert "keep_poses" in str(error.value.hint)


def test_an_engine_that_scored_nobody_here_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(
            tmp_path,
            [_pose("other", -7.0, molblock=_relaxed_molblock(_SMILES["p3"]))],
        )
    assert error.value.code == "POSE_STRAIN_NO_MATCHING_EVIDENCE"
    assert "evidence_from" in str(error.value.hint)


def test_two_engines_must_be_told_apart(tmp_path: Path) -> None:
    relaxed = _relaxed_molblock(_SMILES["p2"])
    twisted = _twisted_molblock(_SMILES["p2"])
    poses = [
        _pose("p2", -9.5, molblock=twisted, engine_id="unidock"),
        _pose("p2", 8.0, molblock=relaxed, engine_id="karmadock", direction="HIGHER_STRONGER"),
    ]
    with pytest.raises(PluginError) as error:
        _run(tmp_path, poses, parents=[_parent("p2")], suffix="-ambiguous")
    assert error.value.code == "POSE_STRAIN_IDENTITY_AMBIGUOUS"
    assert error.value.context["observed_examples"] == ["karmadock", "unidock"]

    output, response = _run(
        tmp_path,
        poses,
        {"engine_id": "unidock"},
        parents=[_parent("p2")],
        suffix="-pinned",
    )
    assert response.metadata["engine_id"] == "unidock"
    assert float(_metrics(output, response)["p2"]["value"]) > 3.0


def test_the_same_seed_is_the_same_measurement(tmp_path: Path) -> None:
    poses = [_pose("p2", -8.5, molblock=_twisted_molblock(_SMILES["p2"]))]
    first = _run(tmp_path, poses, {"seed": 4321}, parents=[_parent("p2")], suffix="-a")
    second = _run(tmp_path, poses, {"seed": 4321}, parents=[_parent("p2")], suffix="-b")
    other = _run(tmp_path, poses, {"seed": 8765}, parents=[_parent("p2")], suffix="-c")

    assert first[1].metadata["method_id"] == second[1].metadata["method_id"]
    assert other[1].metadata["method_id"] != first[1].metadata["method_id"]
    assert float(_metrics(*first)["p2"]["value"]) == pytest.approx(
        float(_metrics(*second)["p2"]["value"])
    )


def test_the_method_identity_carries_everything_that_changes_the_number(
    tmp_path: Path,
) -> None:
    poses = [_pose("p2", -8.5, molblock=_relaxed_molblock(_SMILES["p2"]))]

    def method_id(config: dict[str, object], suffix: str) -> str:
        _, response = _run(tmp_path, poses, config, parents=[_parent("p2")], suffix=suffix)
        return str(response.metadata["method_id"])

    baseline = method_id({}, "-base")
    assert method_id({"reference_conformers": 4}, "-confs") != baseline
    assert method_id({"prune_rms_threshold": 0.75}, "-prune") != baseline
    assert method_id({"restraint_tolerance_angstrom": 0.5}, "-tol") != baseline
    assert method_id({"restraint_force_constant": 250.0}, "-force") != baseline
    assert method_id({"max_minimize_iterations": 400}, "-iters") != baseline
    assert method_id({"negative_tolerance_kcal_mol": 0.5}, "-band") != baseline
    assert method_id({"seed": 99}, "-seed") != baseline
    # How the work is divided is not part of the measurement.
    assert method_id({"batch_size": 1}, "-batch") == baseline


def test_an_unknown_backend_is_refused_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(
            tmp_path,
            [_pose("p3", -3.2, molblock=_relaxed_molblock(_SMILES["p3"]))],
            {"backend": "torsion_library"},
            parents=[_parent("p3")],
        )
    assert error.value.code == "PLUGIN_CONFIG_INVALID"


def test_the_backend_survives_being_written_out_as_json_and_read_back() -> None:
    """Naming the backend as a string has to work, and twice over here.

    Strict validation does not coerce, so an enum field refuses a bare ``str``
    before any validator of ours runs -- the validator has to hand back the
    member, not merely approve the string.  Every exported cascade names it as a
    string once, and this stage names it a second time: each shard re-validates
    the settings it was handed as JSON, so a validator that only checked
    membership would fail inside the worker rather than at config time, on every
    molecule and nowhere near the cause.
    """

    settings = PoseStrainConfig().model_dump(mode="json")
    assert settings["backend"] == "mmff94s_local"
    assert PoseStrainConfig.model_validate(settings).backend is PoseStrainBackend.MMFF94S_LOCAL


def test_it_does_not_accept_the_campaign_receptor() -> None:
    """Lowering injects the campaign target into any config declaring this name.

    This plugin measures the ligand alone and has no use for a receptor, so the
    field must stay absent: declaring it -- even optionally -- would have the
    target silently written in.
    """

    assert "receptor_path" not in PoseStrainConfig.model_fields
    assert tuple(PoseStrainBackend) == (PoseStrainBackend.MMFF94S_LOCAL,)


def test_it_writes_only_the_vocabulary_the_contract_declares(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [
            _pose("p1", -6.0, molblock=_relaxed_molblock(_SMILES["p1"])),
            _pose("p2", -8.5, molblock=_twisted_molblock(_SMILES["p2"])),
        ],
    )
    rows = _metrics(output, response)
    statuses = {str(row["status"]) for row in rows.values()}
    assert statuses <= set(DERIVED_METRIC_V1.enum_values["status"])
    assert {"OK", "NOT_APPLICABLE"} & statuses
    assert response.metadata["units"] in DERIVED_METRIC_V1.enum_values["units"]
    assert response.metadata["direction"] in DERIVED_METRIC_V1.enum_values["direction"]
