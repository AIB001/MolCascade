"""The Uni-Dock adapter, tested at the process boundary rather than at the physics.

Uni-Dock is a CUDA binary that is not installed here and could not run if it
were -- there is no NVIDIA card in this environment.  So what is under test is
everything this project actually owns: which receptor bytes a score is bound
to, that a molecule's score is its own and not its neighbour's, that a lane it
cannot use is refused before any work starts, and that an engine failure
arrives as the engine's own words rather than as a traceback.

The stand-in binary is a fake of the *protocol* -- it parses the real flags,
reads the real ``--ligand_index`` file, and writes the real ``REMARK VINA
RESULT`` output.  Its scores are derived from each ligand's atom count, which
is the trick that makes the identity join checkable: the test can recompute
which number belongs to which molecule and catch an off-by-one join that a
fixture of invented numbers would hide.  It also copies its inputs into a
record file outside the scratch directory, because the adapter deletes that
directory before any test can look inside it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DOCKING_SCORE_V1, LIGAND_CONFORMER_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.docking import UniDockPlugin
from molcascade.plugins.manifest import PluginKind
from molcascade.plugins.registry import create_builtin_registry

# meeko converts the shared conformers into the PDBQT the engine reads. It is an
# optional extra, so the tests that need a real conversion say so; the ones about
# refusing to start do not need it and stay armed either way.
_needs_meeko = pytest.mark.skipif(
    importlib.util.find_spec("meeko") is None,
    reason="meeko is an optional docking extra and is not installed",
)

#: Distinct heavy-atom counts on purpose: the fake engine's score is a function
#: of atom count, so a score landing on the wrong molecule is visible as a
#: number rather than as a subtly reordered table.
_SMILES = (
    "CCO",
    "Clc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCCCCCCCCCCCCCCCC(=O)O",
)


# --------------------------------------------------------------- the fake engine

_FAKE_ENGINE = '''\
import json
import sys
from pathlib import Path

behaviour = json.loads(_BEHAVIOUR)
record_path = Path(behaviour["record_path"])
record = (
    json.loads(record_path.read_text())
    if record_path.exists()
    else {"calls": [], "ligands": []}
)

arguments = {}
key = None
for token in sys.argv[1:]:
    if token.startswith("--"):
        key = token[2:]
        arguments[key] = []
    else:
        arguments[key].append(token)
record["calls"].append(arguments)

if behaviour.get("exit_code"):
    record_path.write_text(json.dumps(record))
    sys.stdout.write(behaviour.get("message", "receptor parse error near residue 214"))
    sys.exit(behaviour["exit_code"])

paths = [line for line in Path(arguments["ligand_index"][0]).read_text().split("\\n") if line]
output_dir = Path(arguments["dir"][0])
modes = int(arguments["num_modes"][0])
skip = set(behaviour.get("skip", []))

for path_text in paths:
    path = Path(path_text)
    text = path.read_text()
    atoms = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    ordinal = len(record["ligands"])
    record["ligands"].append({"stem": path.stem, "atom_count": len(atoms)})
    if ordinal in skip:
        continue
    blocks = []
    for mode in range(modes):
        score = -float(len(atoms)) - mode * 0.5
        blocks.append(
            "MODEL %d\\nREMARK VINA RESULT:  %.3f      0.000      0.000\\n%sENDMDL\\n"
            % (mode + 1, score, text)
        )
    (output_dir / (path.stem + "_out.pdbqt")).write_text("".join(blocks))

record_path.write_text(json.dumps(record))
'''


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    """Return a factory that writes the fake ``unidock`` and its behaviour."""

    executable = tmp_path / "unidock"
    record_path = tmp_path / "record.json"

    def configure(**behaviour: Any) -> Path:
        behaviour.setdefault("record_path", str(record_path))
        executable.write_text(
            f"#!{sys.executable}\n_BEHAVIOUR = {json.dumps(json.dumps(behaviour))}\n{_FAKE_ENGINE}",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    configure.record_path = record_path  # type: ignore[attr-defined]
    return configure


def _record(engine: Any) -> dict[str, Any]:
    path = Path(engine.record_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------- the inputs


class _Target(NamedTuple):
    """The structure the operator supplied, its digest, and its conversion."""

    receptor: Path
    digest: str
    prepared: Path


def _target(tmp_path: Path, *, prepared_text: str = "REMARK prepared\n") -> _Target:
    """Write a receptor PDB and the PDBQT derived from it.

    Two files on purpose. ``receptor_id`` is the digest of what the operator
    handed over, not of what this engine happened to open -- otherwise the same
    protein would carry one identifier under Uni-Dock, which reads a PDBQT, and
    a different one under KarmaDock, which reads the PDB directly, and a
    consensus across the tier would be comparing two targets.
    """

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("REMARK target\nATOM      1  N   ALA A   1\n", encoding="utf-8")
    prepared = tmp_path / "receptor.pdbqt"
    prepared.write_text(f"{prepared_text}ATOM      1  N   ALA A   1\n", encoding="utf-8")
    return _Target(receptor, hashlib.sha256(receptor.read_bytes()).hexdigest(), prepared)


def _parent_row(index: int, smiles: str) -> dict[str, object]:
    return {
        "parent_id": f"parent-{index:03d}",
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{index:03d}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _molblock(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 20_260_823
    rdDistGeom.EmbedMolecule(molecule, params)
    return str(Chem.MolToMolBlock(molecule))


def _parents(root: Path, smiles: tuple[str, ...]) -> StageInput:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_parent_row(index, value) for index, value in enumerate(smiles)],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
        row_group_size=2,
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "a" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _conformers(
    root: Path,
    smiles: tuple[str, ...],
    *,
    without_geometry: frozenset[int] = frozenset(),
    extra: tuple[str, ...] = (),
    reverse: bool = False,
) -> StageInput:
    """Write the geometry side input the adapter joins against.

    ``extra`` and ``reverse`` exist because the conformer table is a *superset*
    in a real cascade -- a gate between preparation and docking removes parents,
    and nothing promises the two tables stay in the same order.
    """

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, value in enumerate(smiles):
        blocked = index in without_geometry
        rows.append(
            {
                "parent_id": f"parent-{index:03d}",
                "conformer_index": 0,
                "molblock": None if blocked else _molblock(value),
                "energy_kcal_mol": None,
                "embed_status": "EMBED_FAILED" if blocked else "MINIMIZED",
                "method_id": "ligand-conformer-etkdgv3:sha256:" + "b" * 64,
            }
        )
    for offset, value in enumerate(extra):
        rows.append(
            {
                "parent_id": f"gated-{offset:03d}",
                "conformer_index": 0,
                "molblock": _molblock(value),
                "energy_kcal_mol": None,
                "embed_status": "MINIMIZED",
                "method_id": "ligand-conformer-etkdgv3:sha256:" + "b" * 64,
            }
        )
    if reverse:
        rows.reverse()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=LIGAND_CONFORMER_V1.schema),
        root / "conformers.parquet",
    )
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + "e" * 64, kind=LIGAND_CONFORMER_V1.id
        ),
        root=root,
        contract_id=LIGAND_CONFORMER_V1.id,
    )


def _settings(executable: Path, target: _Target, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "schema_version": 1,
        "executable": str(executable),
        "receptor_path": str(target.receptor),
        "receptor_pdbqt_path": str(target.prepared),
        "center_x": 10.0,
        "center_y": -4.5,
        "center_z": 22.0,
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
        "batch_size": 64,
        "seed": 4_242,
    }
    settings.update(overrides)
    return settings


def _run(
    tmp_path: Path,
    name: str,
    settings: dict[str, object],
    *,
    smiles: tuple[str, ...] = _SMILES,
    resources: StageResources | None = None,
    conformers: StageInput | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    staging = tmp_path / name
    response = UniDockPlugin().execute(
        StageRequest(
            stage_id="docking",
            inputs={
                "parents": _parents(tmp_path / f"{name}-parents", smiles),
                "conformers": conformers
                or _conformers(tmp_path / f"{name}-conformers", smiles),
            },
            config=settings,
        ),
        StageContext(
            staging,
            resources or StageResources(devices=("cuda:0",), shard_rows=1_000),
        ),
    )
    scores = pa.concat_tables(
        [pq.read_table(staging / path) for path in response.outputs["scores"].file_paths]
    )
    parents = pa.concat_tables(
        [pq.read_table(staging / path) for path in response.outputs["primary"].file_paths]
    )
    DOCKING_SCORE_V1.validate(scores)
    PARENT_V1.validate(parents)
    return (
        scores.to_pylist(),
        [str(row["parent_id"]) for row in parents.to_pylist()],
        dict(response.metadata),
    )


# ------------------------------------------------------------------- the tests


@_needs_meeko
def test_a_score_belongs_to_the_molecule_it_was_computed_for(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(tmp_path, "basic", _settings(engine(), target))

    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert [row["parent_id"] for row in rows] == parents
    assert {row["engine_id"] for row in rows} == {"unidock"}
    assert {row["receptor_id"] for row in rows} == {target.digest}
    assert {row["score_kind"] for row in rows} == {"VINA_KCAL_MOL"}
    assert {row["direction"] for row in rows} == {"LOWER_STRONGER"}
    assert metadata["receptor_id"] == target.digest
    assert metadata["scored_pose_count"] == len(_SMILES)
    assert metadata["missing_geometry_count"] == 0

    # The join, checked against the engine's own view of what it was handed:
    # ligand n's atom count must be molecule n's score.
    ligands = _record(engine)["ligands"]
    assert [row["score"] for row in rows] == [-float(item["atom_count"]) for item in ligands]
    # ...and those atom counts have to differ, or the assertion above proves
    # nothing about ordering.
    assert len({item["atom_count"] for item in ligands}) == len(_SMILES)


@_needs_meeko
def test_the_receptor_and_the_box_reach_the_command_line(tmp_path: Path, engine: Any) -> None:
    target = _target(tmp_path)
    _rows, _parents, metadata = _run(
        tmp_path,
        "command",
        _settings(engine(), target, scoring="vinardo", search_mode="detail"),
    )

    call = _record(engine)["calls"][0]
    # The engine opens the conversion; the score is recorded against the target.
    assert call["receptor"] == [str(target.prepared)]
    assert metadata["receptor_id"] == target.digest
    assert metadata["receptor_pdbqt_path"] == str(target.prepared)
    assert call["center_x"] == ["10.0"]
    assert call["center_y"] == ["-4.5"]
    assert call["size_z"] == ["20.0"]
    assert call["scoring"] == ["vinardo"]
    assert call["search_mode"] == ["detail"]
    assert call["seed"] == ["4242"]
    assert call["num_modes"] == ["1"]
    assert Path(call["ligand_index"][0]).name == "ligand_index.txt"
    assert metadata["scoring"] == "vinardo"
    assert metadata["center_x"] == 10.0


@_needs_meeko
def test_poses_come_back_ranked_best_first(tmp_path: Path, engine: Any) -> None:
    """``docking_score/v1`` states best-first as an invariant, so it is enforced.

    The fake writes its ``MODEL`` blocks worst-first on purpose. Real Vina does
    not, which is exactly why inheriting the ordering from the output format
    rather than establishing it here would be a promise held by someone else.
    """

    target = _target(tmp_path)
    rows, _parents, metadata = _run(
        tmp_path,
        "poses",
        _settings(engine(), target, num_modes=3, keep_poses=True),
        smiles=(_SMILES[0], _SMILES[1]),
    )

    assert len(rows) == 6
    assert metadata["scored_pose_count"] == 6
    for parent_id in ("parent-000", "parent-001"):
        poses = [row for row in rows if row["parent_id"] == parent_id]
        assert [row["pose_rank"] for row in poses] == [0, 1, 2]
        scores = [row["score"] for row in poses]
        assert scores == sorted(scores)
    # keep_poses is the only reason a molblock should be present at all.
    assert all(row["pose_molblock"] for row in rows)


@_needs_meeko
def test_poses_are_not_stored_unless_they_were_asked_for(tmp_path: Path, engine: Any) -> None:
    target = _target(tmp_path)
    rows, _parents, _metadata = _run(
        tmp_path, "no-poses", _settings(engine(), target), smiles=(_SMILES[0],)
    )

    assert [row["pose_molblock"] for row in rows] == [None]


@_needs_meeko
def test_geometry_is_joined_by_identity_not_by_position(tmp_path: Path, engine: Any) -> None:
    """A gate between preparation and docking is allowed to change the table.

    Reading the conformers positionally would dock every survivor as whichever
    molecule happened to sit at its index -- a failure that produces a complete,
    plausible, entirely wrong score table.
    """

    target = _target(tmp_path)
    shuffled = _conformers(
        tmp_path / "shuffled",
        _SMILES,
        extra=("CCN(CC)CC", "OCC1OC(O)C(O)C(O)C1O"),
        reverse=True,
    )
    rows, _parents, _metadata = _run(
        tmp_path, "identity", _settings(engine(), target), conformers=shuffled
    )

    ligands = _record(engine)["ligands"]
    assert [row["parent_id"] for row in rows] == [
        f"parent-{index:03d}" for index in range(len(_SMILES))
    ]
    assert [row["score"] for row in rows] == [-float(item["atom_count"]) for item in ligands]


@_needs_meeko
def test_a_molecule_without_geometry_keeps_its_row_and_gets_no_score(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    blocked = _conformers(
        tmp_path / "blocked", _SMILES, without_geometry=frozenset({1})
    )
    rows, parents, metadata = _run(
        tmp_path, "no-geometry", _settings(engine(), target), conformers=blocked
    )

    # Docking declares no filter: rejecting for missing evidence is the score
    # gate's job, and it does it in the open.
    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert "parent-001" not in {row["parent_id"] for row in rows}
    assert metadata["missing_geometry_count"] == 1
    assert metadata["scored_pose_count"] == len(_SMILES) - 1


@_needs_meeko
def test_a_ligand_the_engine_returned_nothing_for_is_counted(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(
        tmp_path, "unscored", _settings(engine(skip=[2]), target)
    )

    assert len(parents) == len(_SMILES)
    assert [row["parent_id"] for row in rows] == [
        "parent-000",
        "parent-001",
        "parent-003",
    ]
    assert metadata["unscored_count"] == 1


@_needs_meeko
def test_splitting_the_work_does_not_move_a_score(tmp_path: Path, engine: Any) -> None:
    target = _target(tmp_path)
    serial, _parents, serial_metadata = _run(
        tmp_path,
        "serial",
        _settings(engine(), target),
        resources=StageResources(devices=("cuda:0",), shard_rows=1_000),
    )
    split, _parents, split_metadata = _run(
        tmp_path,
        "split",
        _settings(engine(), target),
        resources=StageResources(
            workers=2, devices=("cuda:0", "cuda:1"), shard_rows=2, batch_size=2
        ),
    )

    assert serial_metadata["shard_count"] == 1
    assert split_metadata["shard_count"] == 2
    assert serial == split
    assert serial_metadata["method_id"] == split_metadata["method_id"]


@_needs_meeko
def test_a_non_zero_exit_arrives_as_the_engines_own_message(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "failure",
            _settings(engine(exit_code=3, message="CUDA error: out of memory"), target),
        )

    assert raised.value.code == "DOCKING_UNIDOCK_RUN_FAILED"
    assert "out of memory" in raised.value.context["output_tail"]
    assert raised.value.context["returncode"] == 3


def test_a_cpu_lane_is_refused_before_any_ligand_is_written(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "cpu",
            _settings(engine(), target),
            resources=StageResources(devices=("cpu",), shard_rows=1_000),
        )

    assert raised.value.code == "DOCKING_GPU_REQUIRED"
    assert _record(engine) == {}


def test_a_receptor_that_changed_since_the_cascade_was_written_stops_the_run(
    tmp_path: Path, engine: Any
) -> None:
    """The digest is what makes ``receptor_id`` mean something.

    Without this check a re-prepared or re-protonated structure would produce
    scores that sit in the same column as the old ones and are not comparable
    with them.
    """

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "mismatch",
            _settings(engine(), target, receptor_sha256="f" * 64),
        )

    assert raised.value.code == "DOCKING_RECEPTOR_DIGEST_MISMATCH"
    assert _record(engine) == {}


@_needs_meeko
def test_re_preparing_the_receptor_changes_the_method_not_the_target(
    tmp_path: Path, engine: Any
) -> None:
    """Two preparations of one protein are one target computed two ways.

    ``receptor_id`` is what says a score is about this protein, so it must not
    move when the conversion is redone; ``method_id`` is what says how the
    number was produced, so it must, because a differently protonated PDBQT is
    a different computation and its scores are not interchangeable.
    """

    first = _target(tmp_path)
    _rows, _parents, before = _run(tmp_path, "prep-a", _settings(engine(), first))
    second = _target(tmp_path, prepared_text="REMARK prepared again\n")
    _rows, _parents, after = _run(tmp_path, "prep-b", _settings(engine(), second))

    assert before["receptor_id"] == after["receptor_id"] == first.digest
    assert before["receptor_pdbqt_id"] != after["receptor_pdbqt_id"]
    assert before["method_id"] != after["method_id"]


def test_a_receptor_that_is_not_pdbqt_is_refused(tmp_path: Path, engine: Any) -> None:
    """Uni-Dock reads one format, and it is not the one the operator supplied."""

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "pdb",
            _settings(engine(), target, receptor_pdbqt_path=str(target.receptor)),
        )

    assert raised.value.code == "DOCKING_RECEPTOR_FORMAT_INVALID"
    assert "--receptor-pdbqt" in (raised.value.hint or "")
    assert _record(engine) == {}


def test_a_prepared_receptor_that_is_not_there_stops_the_run(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    target.prepared.unlink()

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "unprepared", _settings(engine(), target))

    assert raised.value.code == "DOCKING_RECEPTOR_UNREADABLE"
    assert _record(engine) == {}


def test_a_missing_executable_names_the_setting_that_fixes_it(tmp_path: Path) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "missing",
            _settings(tmp_path / "not-installed" / "unidock", target),
        )

    assert raised.value.code == "DOCKING_EXECUTABLE_MISSING"
    assert "executable" in (raised.value.hint or "")


def test_the_population_cap_stops_the_run_before_the_first_ligand(
    tmp_path: Path, engine: Any
) -> None:
    """A cap noticed after the work has started is not a cap."""

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "cap", _settings(engine(), target, max_molecules=2))

    assert raised.value.code == "DOCKING_POPULATION_TOO_LARGE"
    assert raised.value.context["molecule_count"] == len(_SMILES)
    assert _record(engine) == {}


def test_a_relative_executable_is_rejected_by_configuration(tmp_path: Path) -> None:
    """``PATH`` is the wrong place to find an engine that lives in its own env."""

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "relative", _settings(Path("unidock"), target))

    assert raised.value.code == "PLUGIN_CONFIG_INVALID"


def test_the_engine_is_registered_as_a_docking_stage() -> None:
    descriptor = create_builtin_registry().get("docking.unidock@0.1.0").descriptor

    assert descriptor.kind is PluginKind.DOCK
    assert descriptor.inputs == (PARENT_V1.id, LIGAND_CONFORMER_V1.id)
    assert descriptor.output_ports["scores"] == DOCKING_SCORE_V1.id
