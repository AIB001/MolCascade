"""Pose consistency must not become a ligand-aligned or missing-value pass."""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pytest
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Geometry import Point3D

from molcascade.cascade.defaults import _docking_tier, _redock_tier, default_cascade
from molcascade.cascade.lower import lower_cascade
from molcascade.contracts import DOCKING_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.builtin.docking import redock
from molcascade.plugins.builtin.docking.redock import (
    RedockConfig,
    fixed_frame_rmsd,
    fresh_conformer,
)
from molcascade.plugins.builtin.evidence_gates import _outcome_for_value
from molcascade.plugins.registry import create_builtin_registry


def config(**extra):
    return RedockConfig(
        receptor_path="/tmp/protein.pdb",
        receptor_pdbqt_path="/tmp/protein.pdbqt",
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        size_x=20.0,
        size_y=20.0,
        size_z=20.0,
        **extra,
    )


def block(smiles="CCO", shift=0.0):
    original = fresh_conformer(smiles, parent_id="test", seed=21)
    mol = Chem.MolFromMolBlock(original, removeHs=False)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + shift, p.y, p.z))
    return Chem.MolToMolBlock(mol)


def test_translation_is_not_aligned_away_and_uses_heavy_atoms():
    value, failure = fixed_frame_rmsd(block(), block(shift=3.0))
    assert failure is None
    assert value == pytest.approx(3.0, abs=1e-4)
    mol = Chem.MolFromMolBlock(block(), removeHs=False)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            mol.GetConformer().SetAtomPosition(atom.GetIdx(), Point3D(50.0, 50.0, 50.0))
    assert fixed_frame_rmsd(block(), Chem.MolToMolBlock(mol))[0] == pytest.approx(0.0)


def test_atom_reordering_and_symmetry_do_not_inflate_rmsd():
    original = block("c1ccccc1")
    mol = Chem.MolFromMolBlock(original, removeHs=True)
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert fixed_frame_rmsd(original, Chem.MolToMolBlock(reordered))[0] == pytest.approx(0.0)
    assert fixed_frame_rmsd(original, original, max_matches=1)[0] is None


def test_incompatible_molecules_missing_and_bad_poses_fail_closed():
    assert fixed_frame_rmsd(block(), block("CCN"))[0] is None
    assert fixed_frame_rmsd("garbage", block())[0] is None
    reference = {"pose_molblock": block(), "method_id": "original"}
    row = redock._metric("p", "m", reference, None, config())
    assert row["value"] is None and row["status"] == "BACKEND_FAILED"


@pytest.mark.parametrize(
    "value,accepted",
    [(1.9999, True), (2.0, False), (2.0001, False), (None, False), (float("nan"), False)],
)
def test_strict_boundary_and_nonfinite_never_pass(value, accepted):
    result = _outcome_for_value(
        value, minimum=None, maximum=2.0, maximum_exclusive=True, reason_prefix="REDOCK"
    )
    assert result[0] is accepted


def test_old_inclusive_gate_behavior_stays_inclusive():
    assert _outcome_for_value(2.0, minimum=None, maximum=2.0, reason_prefix="OLD")[0]


def test_seed_independence_and_fresh_geometry():
    with pytest.raises(ValidationError, match="must differ"):
        config(seed=42, source_seed=42)
    with pytest.raises(ValidationError, match="keep_poses"):
        config(keep_poses=False)
    a = fresh_conformer("CCCOCC", parent_id="p", seed=42)
    assert a == fresh_conformer("CCCOCC", parent_id="p", seed=42)
    assert a != fresh_conformer("CCCOCC", parent_id="p", seed=43)


def test_provenance_names_the_unchanged_original_pose():
    original = block()
    source = {"pose_molblock": original, "method_id": "source-method"}
    result = {"pose_molblock": block(shift=1.0), "score": -8.0}
    row = redock._metric("p", "m", source, result, config())
    provenance = json.loads(row["source_json"])
    assert provenance["source_pose_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert provenance["retained_pose"] == "original_docking_pose"
    assert provenance["ligand_alignment"] is False
    assert source["pose_molblock"] == original
    assert DOCKING_SCORE_V1.id not in redock.RedockPlugin.descriptor.outputs


def test_defaults_and_lowered_plan_explicitly_bind_redock_and_gate():
    registry = create_builtin_registry()
    cascade = default_cascade(registry=registry)
    tiers = {tier.id: tier for tier in cascade.tiers}
    if "t9_redock" not in tiers:
        pytest.skip("Uni-Dock optional dependencies not installed")
    criterion = tiers["t9_redock"].criteria[0]
    assert criterion.gate.settings["maximum"] == 2.0
    assert criterion.gate.settings["maximum_exclusive"] is True
    assert criterion.gate.settings["on_unscorable"] == "reject"
    assert criterion.evidence_from == {DOCKING_SCORE_V1.id: "docking_score"}
    plan = lower_cascade(
        cascade, registry=registry, allow_missing_target=True, allow_missing_library=True
    )
    stages = {stage.id: stage for stage in plan.pipeline.stages}
    assert any(binding.stage == "docking_score" for binding in stages[criterion.id].inputs)
    assert all(tier.id != "t9_redock" for tier in default_cascade(include_redock=False).tiers)
    assert _redock_tier(registry, None) is None


def test_default_redock_inherits_source_search_policy_but_changes_seed():
    registry = create_builtin_registry()
    docking = _docking_tier(registry)
    if docking is None or not any(
        c.backend.startswith("docking.unidock") for c in docking.criteria
    ):
        pytest.skip("Uni-Dock optional dependencies not installed")
    docking = docking.model_copy(
        update={
            "criteria": tuple(
                c.model_copy(
                    update={"settings": {**c.settings, "seed": 42, "search_mode": "detail"}}
                )
                if c.backend.startswith("docking.unidock")
                else c
                for c in docking.criteria
            )
        }
    )
    stage = _redock_tier(registry, docking).criteria[0]
    assert stage.settings["source_seed"] == 42
    assert stage.settings["seed"] != 42
    assert stage.settings["search_mode"] == "detail"


def test_source_identity_mismatch_is_not_silently_compared(monkeypatch):
    row = {
        "engine_id": "unidock",
        "pose_rank": 0,
        "receptor_id": "receptor",
        "parent_id": "p",
        "method_id": "wrong-seed-or-settings",
    }
    monkeypatch.setattr(redock, "read_side_input", lambda *a, **kw: pa.Table.from_pylist([row]))
    with pytest.raises(PluginError) as err:
        redock._reference_poses(None, ["p"], config(), "receptor", "expected")
    assert err.value.code == "REDOCK_SOURCE_METHOD_MISMATCH"
    row["receptor_id"] = "another-receptor"
    with pytest.raises(PluginError) as err:
        redock._reference_poses(None, ["p"], config(), "receptor", "expected")
    assert err.value.code == "REDOCK_RECEPTOR_MISMATCH"


def test_shard_searches_fresh_inputs_and_emits_every_parent_including_failures(
    tmp_path, monkeypatch
):
    import pyarrow.parquet as pq

    from molcascade.contracts import DERIVED_METRIC_V1, PARENT_V1
    from molcascade.parallel import ShardTask

    cfg = config()
    original = block()
    expected = redock.source_method_id(
        cfg.model_copy(update={"seed": cfg.source_seed}), receptor_id="r", prepared_id="p"
    )
    source_rows = [
        {
            "parent_id": p,
            "engine_id": "unidock",
            "receptor_id": "r",
            "pose_rank": 0,
            "score": -8.0,
            "score_kind": "VINA_KCAL_MOL",
            "direction": "LOWER_STRONGER",
            "secondary_score": None,
            "pose_molblock": original,
            "method_id": expected,
        }
        for p in ["a", "b", "c"]
    ]
    parents = [
        {
            "parent_id": p,
            "parent_smiles": "CCO",
            "identity_policy_id": "i",
            "registration_key": p,
            "stereo_key": None,
            "formula": None,
            "duplicate_count": 1,
        }
        for p in ["a", "b", "c"]
    ]
    input_file, pose_file = tmp_path / "input.parquet", tmp_path / "poses.parquet"
    pq.write_table(pa.Table.from_pylist(parents, schema=PARENT_V1.schema), input_file)
    pq.write_table(pa.Table.from_pylist(source_rows, schema=DOCKING_SCORE_V1.schema), pose_file)
    monkeypatch.setattr(redock, "require_meeko", lambda **kwargs: None)
    monkeypatch.setattr(redock, "resolved_executable", lambda *a, **kw: tmp_path / "unidock")
    monkeypatch.setattr(redock, "verified_receptor", lambda c: (tmp_path / "receptor", "r"))
    monkeypatch.setattr(redock, "_prepared_receptor", lambda c: (tmp_path / "prepared", "p"))
    monkeypatch.setattr(redock, "ligand_pdbqt", lambda text: text)
    calls = []

    def dock(ligands, **kwargs):
        calls.append((ligands, kwargs["config"].seed))
        assert all(text != original for _, text in ligands)
        return (
            [
                {**source_rows[0], "pose_molblock": block(shift=1.0)},
                {**source_rows[1], "pose_molblock": block(shift=3.0)},
            ],
            1,
        )

    monkeypatch.setattr(redock, "_dock_batch", dock)
    paths = {"primary": tmp_path / "passed.parquet", "derived_metrics": tmp_path / "metric.parquet"}
    task = ShardTask(
        index=0,
        input_files=(input_file,),
        row_offset=0,
        row_count=3,
        output_paths=paths,
        side_inputs={"poses": (pose_file,)},
        config=cfg.model_dump(mode="json"),
        device="cuda:0",
    )
    outcome = redock._run_shard(task)
    metrics = pq.read_table(paths["derived_metrics"])
    DERIVED_METRIC_V1.validate(metrics)
    values = metrics.column("value").to_pylist()
    assert values[:2] == pytest.approx([1.0, 3.0])
    assert values[2] is None
    assert outcome.metadata["failed_count"] == 1
    assert calls[0][1] != cfg.source_seed
    assert pq.read_table(paths["primary"]).num_rows == 3
    assert pq.read_table(pose_file).column("pose_molblock").to_pylist() == [original] * 3


def test_fresh_input_preserves_instantiated_stereo_and_parent_constraints():
    from molcascade.plugins.builtin.docking.redock import redock_input_smiles

    for i in range(10):
        original = fresh_conformer("CC(O)C(=O)O", parent_id=f"p{i}", seed=20_260_823)
        chosen = redock_input_smiles("CC(O)C(=O)O", original)
        assert chosen is not None and "@" in chosen
        repeated = fresh_conformer(chosen, parent_id=f"p{i}", seed=20_260_824)
        assert fixed_frame_rmsd(original, repeated)[1] is None
    source = block("C[C@H](O)C(=O)O")
    assert redock_input_smiles("C[C@H](O)C(=O)O", source) is not None
    assert redock_input_smiles("C[C@@H](O)C(=O)O", source) is None
    assert redock_input_smiles("CC(O)C(=O)N", source) is None


def test_nonfinite_engine_score_is_recorded_as_failure_without_crashing_shard():
    original = block()
    row = redock._metric(
        "p",
        "m",
        {"pose_molblock": original, "method_id": "m"},
        {"pose_molblock": original, "score": float("nan")},
        config(),
    )
    assert row["value"] is None and row["status"] == "BACKEND_FAILED"
    assert json.loads(row["source_json"])["redock_score"] is None


def test_redock_preflight_reuses_the_actual_unidock_installation():
    from types import SimpleNamespace

    from molcascade.backends.preflight import required_backends

    stages = (SimpleNamespace(stage_id="redock", plugin_key="docking.redock_consistency@0.1.0"),)
    found = required_backends(stages)
    assert len(found) == 1 and found[0][0] == "redock"
    assert found[0][1].id == "unidock"
    assert found[0][1].executable_config_key == "executable"
