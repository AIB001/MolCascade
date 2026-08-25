"""The GNINA adapter, tested at the process boundary rather than at the network.

GNINA is a CUDA binary that is not installed here and could not run if it were.
So what is under test is the part this project owns: that a score belongs to the
molecule it was computed for, that the choice of *which* of GNINA's three
numbers is the score reaches both the rows and the method identity, that the
lane this process was pinned to arrives as ``--device 0`` and a CPU lane is
refused unless somebody asked for it, and that an engine failure comes back as
the engine's own words.

The stand-in binary is a fake of the *protocol*.  It parses the real flags --
including GNINA's single-dash ``-r``/``-l``/``-o`` -- reads the real ligand SDF
and writes the real ``minimizedAffinity``/``CNNscore``/``CNNaffinity`` tags.
Every number it writes is a function of the ligand's atom count, which is what
makes the identity join checkable: a table of invented scores would hide an
off-by-one join, a table derived from the molecules cannot.  It also copies what
it was handed into a record file outside the scratch directory, because the
adapter deletes that directory before any test can look inside it.

The fake writes its poses *worst*-first on purpose, and worst-first on all three
scales at once.  Real GNINA does not, which is exactly why inheriting the
ordering from the output file rather than establishing it here would leave a
stated invariant in another program's hands.
"""

from __future__ import annotations

import hashlib
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
from molcascade.plugins.builtin.docking import GninaPlugin
from molcascade.plugins.manifest import PluginKind
from molcascade.plugins.registry import create_builtin_registry

#: Distinct heavy-atom counts on purpose: every score the fake writes is a
#: function of atom count, so a score landing on the wrong molecule shows up as
#: a number rather than as a subtly reordered table.
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


def is_flag(token):
    # GNINA mixes '-r' with '--cnn_scoring', and a box centre is allowed to be
    # negative -- so a leading dash is not enough to make a token a flag.
    return token.startswith("-") and token.lstrip("-")[:1].isalpha()


arguments = {}
key = None
for token in sys.argv[1:]:
    if is_flag(token):
        key = token.lstrip("-")
        arguments[key] = []
    else:
        arguments[key].append(token)
record["calls"].append(arguments)

if behaviour.get("exit_code"):
    record_path.write_text(json.dumps(record))
    sys.stdout.write(behaviour.get("message", "could not open receptor"))
    sys.exit(behaviour["exit_code"])

modes = int(arguments["num_modes"][0])
skip = set(behaviour.get("skip", []))
drop_tags = set(behaviour.get("drop_tags", []))
blocks = []
for chunk in Path(arguments["l"][0]).read_text().split("$$$$"):
    body = chunk.lstrip("\\n")
    if not body.strip():
        continue
    lines = body.splitlines()
    name = lines[0].strip()
    # The V2000 counts line holds the atom count in its first three columns.
    atoms = int(lines[3][:3])
    ordinal = len(record["ligands"])
    record["ligands"].append({"name": name, "atom_count": atoms, "line_count": len(lines)})
    if ordinal in skip:
        continue
    molblock = "\\n".join(lines[: lines.index("M  END") + 1])
    for mode in range(modes):
        tags = {
            "minimizedAffinity": "%.3f" % (-atoms + mode * 0.5),
            "CNNscore": "%.4f" % (atoms / 100.0 + mode * 0.01),
            "CNNaffinity": "%.3f" % (atoms / 2.0 + mode * 0.1),
        }
        for absent in drop_tags:
            tags.pop(absent, None)
        rendered = "".join("> <%s>\\n%s\\n\\n" % (k, v) for k, v in tags.items())
        blocks.append("%s\\n%s\\n$$$$\\n" % (molblock, rendered.rstrip("\\n")))

Path(arguments["o"][0]).write_text("".join(blocks))
record_path.write_text(json.dumps(record))
'''


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    """Return a factory that writes the fake ``gnina`` and its behaviour."""

    executable = tmp_path / "gnina"
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
    receptor: Path
    digest: str


def _target(tmp_path: Path, *, suffix: str = ".pdb") -> _Target:
    """The operator's structure, handed to ``-r`` unchanged.

    GNINA is the one engine in the tier that opens exactly the file the cascade
    named, so here ``receptor_id`` is the digest of the bytes the engine read.
    """

    receptor = tmp_path / f"receptor{suffix}"
    receptor.write_text("REMARK target\nATOM      1  N   ALA A   1\n", encoding="utf-8")
    return _Target(receptor, hashlib.sha256(receptor.read_bytes()).hexdigest())


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


def _molblock(smiles: str, *, trailing: str = "") -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 20_260_823
    rdDistGeom.EmbedMolecule(molecule, params)
    return str(Chem.MolToMolBlock(molecule)) + trailing


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
    trailing: str = "",
) -> StageInput:
    """Write the geometry side input the adapter joins against.

    ``extra`` and ``reverse`` exist because in a real cascade this table is a
    *superset* -- a gate between preparation and docking removes parents, and
    nothing promises the two tables stay in the same order.
    """

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, value in enumerate(smiles):
        blocked = index in without_geometry
        rows.append(
            {
                "parent_id": f"parent-{index:03d}",
                "conformer_index": 0,
                "molblock": None if blocked else _molblock(value, trailing=trailing),
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
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "e" * 64, kind=LIGAND_CONFORMER_V1.id),
        root=root,
        contract_id=LIGAND_CONFORMER_V1.id,
    )


def _settings(executable: Path, target: _Target, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "schema_version": 1,
        "executable": str(executable),
        "receptor_path": str(target.receptor),
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
    response = GninaPlugin().execute(
        StageRequest(
            stage_id="docking",
            inputs={
                "parents": _parents(tmp_path / f"{name}-parents", smiles),
                "conformers": conformers or _conformers(tmp_path / f"{name}-conformers", smiles),
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


def test_a_score_belongs_to_the_molecule_it_was_computed_for(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(tmp_path, "basic", _settings(engine(), target))

    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert [row["parent_id"] for row in rows] == parents
    assert {row["engine_id"] for row in rows} == {"gnina"}
    assert {row["receptor_id"] for row in rows} == {target.digest}
    # The shipped default is the CNN affinity, and it gets better going up --
    # the opposite of the Vina-family scores sitting in the same column from the
    # engine next to it, which is why every row carries its own direction.
    assert {row["score_kind"] for row in rows} == {"CNN_AFFINITY"}
    assert {row["direction"] for row in rows} == {"HIGHER_STRONGER"}
    assert {row["pose_rank"] for row in rows} == {0}
    assert metadata["receptor_id"] == target.digest
    assert metadata["scored_pose_count"] == len(_SMILES)
    assert metadata["missing_geometry_count"] == 0

    # The join, checked against the engine's own view of what it was handed.
    ligands = _record(engine)["ligands"]
    assert [row["score"] for row in rows] == [item["atom_count"] / 2.0 for item in ligands]
    # The companion number comes from the same pose, so it must track too.
    assert [row["secondary_score"] for row in rows] == [
        -float(item["atom_count"]) for item in ligands
    ]
    # ...and the atom counts have to differ, or the assertions above prove
    # nothing about ordering.
    assert len({item["atom_count"] for item in ligands}) == len(_SMILES)


def test_the_receptor_and_the_box_reach_the_command_line(tmp_path: Path, engine: Any) -> None:
    target = _target(tmp_path)
    _rows, _parents, metadata = _run(
        tmp_path,
        "command",
        _settings(
            engine(),
            target,
            cnn_scoring="refinement",
            cnn="crossdock_default2018",
            exhaustiveness=16,
        ),
    )

    call = _record(engine)["calls"][0]
    # GNINA reads the operator's PDB itself: no conversion, one digest.
    assert call["r"] == [str(target.receptor)]
    assert Path(call["l"][0]).name == "ligands.sdf"
    assert Path(call["o"][0]).name == "poses.sdf"
    assert call["center_x"] == ["10.0"]
    assert call["center_y"] == ["-4.5"]
    assert call["size_z"] == ["20.0"]
    assert call["seed"] == ["4242"]
    assert call["num_modes"] == ["1"]
    assert call["exhaustiveness"] == ["16"]
    assert call["cnn_scoring"] == ["refinement"]
    assert call["cnn"] == ["crossdock_default2018"]
    assert metadata["cnn_scoring"] == "refinement"
    assert metadata["exhaustiveness"] == 16
    assert metadata["center_x"] == 10.0


def test_the_assigned_card_is_always_device_zero(tmp_path: Path, engine: Any) -> None:
    """The pool pinned this process before it started, so index 0 is the card.

    Passing the lane's own global index would name a device outside the
    ``CUDA_VISIBLE_DEVICES`` mask, which GNINA reports as a missing GPU rather
    than as the misconfiguration it is.
    """

    target = _target(tmp_path)
    _run(
        tmp_path,
        "lane",
        _settings(engine(), target),
        resources=StageResources(devices=("cuda:3",), shard_rows=1_000),
    )

    call = _record(engine)["calls"][0]
    assert call["device"] == ["0"]
    assert "no_gpu" not in call


def test_a_cpu_lane_is_refused_unless_it_was_asked_for(tmp_path: Path, engine: Any) -> None:
    """GNINA's CPU path works, which is exactly why it needs a deliberate switch.

    An engine that simply crashed on CPU would announce itself. This one runs,
    thousands of times slower, and the run looks fine until it does not finish.
    """

    target = _target(tmp_path)
    cpu = StageResources(devices=("cpu",), shard_rows=1_000)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "cpu-refused", _settings(engine(), target), resources=cpu)

    assert raised.value.code == "DOCKING_GPU_REQUIRED"
    assert "allow_cpu" in (raised.value.hint or "")
    assert _record(engine) == {}

    rows, _parents, _metadata = _run(
        tmp_path,
        "cpu-allowed",
        _settings(engine(), target, allow_cpu=True, cpu_threads=4),
        resources=cpu,
    )

    call = _record(engine)["calls"][0]
    assert call["no_gpu"] == []
    assert call["cpu"] == ["4"]
    assert "device" not in call
    assert len(rows) == len(_SMILES)


def test_which_of_the_three_numbers_is_the_score_is_a_setting(
    tmp_path: Path, engine: Any
) -> None:
    """GNINA reports an affinity and two CNN numbers; the choice is recorded."""

    target = _target(tmp_path)
    cnn, _parents, cnn_metadata = _run(tmp_path, "by-cnn", _settings(engine(), target))
    empirical, _parents, empirical_metadata = _run(
        tmp_path, "by-affinity", _settings(engine(), target, rank_by="affinity")
    )

    assert {row["score_kind"] for row in empirical} == {"VINA_KCAL_MOL"}
    assert {row["direction"] for row in empirical} == {"LOWER_STRONGER"}
    assert all(row["score"] < 0 for row in empirical)
    # The two rankings swap which number is primary and which is the companion.
    assert [row["secondary_score"] for row in empirical] == [row["score"] for row in cnn]
    assert [row["score"] for row in empirical] == [row["secondary_score"] for row in cnn]
    # A different question about the same poses is a different computation.
    assert empirical_metadata["method_id"] != cnn_metadata["method_id"]
    assert empirical_metadata["score_kind"] == "VINA_KCAL_MOL"


def test_poses_come_back_ranked_best_first_on_either_scale(
    tmp_path: Path, engine: Any
) -> None:
    """Best-first is an invariant of the contract, so it is established here."""

    target = _target(tmp_path)
    cnn, _parents, metadata = _run(
        tmp_path,
        "poses-cnn",
        _settings(engine(), target, num_modes=3, keep_poses=True),
        smiles=(_SMILES[0], _SMILES[1]),
    )
    empirical, _parents, _metadata = _run(
        tmp_path,
        "poses-affinity",
        _settings(engine(), target, num_modes=3, rank_by="affinity"),
        smiles=(_SMILES[0], _SMILES[1]),
    )

    assert len(cnn) == 6
    assert metadata["scored_pose_count"] == 6
    for parent_id in ("parent-000", "parent-001"):
        higher = [row["score"] for row in cnn if row["parent_id"] == parent_id]
        lower = [row["score"] for row in empirical if row["parent_id"] == parent_id]
        assert [row["pose_rank"] for row in cnn if row["parent_id"] == parent_id] == [0, 1, 2]
        assert higher == sorted(higher, reverse=True)
        assert lower == sorted(lower)
    # keep_poses is the only reason a molblock should be present at all.
    assert all("M  END" in str(row["pose_molblock"]) for row in cnn)
    assert [row["pose_molblock"] for row in empirical] == [None] * 6


def test_ranking_by_a_number_the_settings_never_produce_is_refused(
    tmp_path: Path, engine: Any
) -> None:
    """With the network off there are no CNN tags, so the shortlist would be empty.

    Not mis-ranked -- empty, every molecule counted as unscored, which reads as a
    library that failed to dock rather than as a setting that contradicts itself.
    """

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "no-cnn", _settings(engine(), target, cnn_scoring="none"))

    assert raised.value.code == "PLUGIN_CONFIG_INVALID"
    assert _record(engine) == {}

    rows, _parents, _metadata = _run(
        tmp_path,
        "no-cnn-affinity",
        _settings(engine(), target, cnn_scoring="none", rank_by="affinity"),
    )
    assert len(rows) == len(_SMILES)


def test_a_pose_missing_the_ranked_tag_is_not_a_score(tmp_path: Path, engine: Any) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(
        tmp_path, "no-tag", _settings(engine(drop_tags=["CNNaffinity"]), target)
    )

    assert len(parents) == len(_SMILES)
    assert rows == []
    assert metadata["unscored_count"] == len(_SMILES)


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
    assert [row["score"] for row in rows] == [item["atom_count"] / 2.0 for item in ligands]


def test_properties_left_on_an_upstream_molblock_do_not_reach_the_engine(
    tmp_path: Path, engine: Any
) -> None:
    """An inherited SDF tag block would arrive back as a tag this module reads.

    Worse, its ``$$$$`` would split one ligand into two records under one name,
    so the ligand count is the assertion: four molecules must be four ligands.
    """

    target = _target(tmp_path)
    contaminated = _conformers(
        tmp_path / "tagged",
        _SMILES,
        trailing="> <CNNaffinity>\n999.0\n\n$$$$\n",
    )
    rows, _parents, _metadata = _run(
        tmp_path, "tagged", _settings(engine(), target), conformers=contaminated
    )

    ligands = _record(engine)["ligands"]
    assert len(ligands) == len(_SMILES)
    assert [row["score"] for row in rows] == [item["atom_count"] / 2.0 for item in ligands]
    assert all(row["score"] != 999.0 for row in rows)


def test_a_molecule_without_geometry_keeps_its_row_and_gets_no_score(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    blocked = _conformers(tmp_path / "blocked", _SMILES, without_geometry=frozenset({1}))
    rows, parents, metadata = _run(
        tmp_path, "no-geometry", _settings(engine(), target), conformers=blocked
    )

    # Docking declares no filter: rejecting for missing evidence is the score
    # gate's job, and it does it in the open.
    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert "parent-001" not in {row["parent_id"] for row in rows}
    assert metadata["missing_geometry_count"] == 1
    assert metadata["scored_pose_count"] == len(_SMILES) - 1


def test_a_ligand_the_engine_returned_nothing_for_is_counted(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(tmp_path, "unscored", _settings(engine(skip=[2]), target))

    assert len(parents) == len(_SMILES)
    assert [row["parent_id"] for row in rows] == ["parent-000", "parent-001", "parent-003"]
    assert metadata["unscored_count"] == 1


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


def test_a_non_zero_exit_arrives_as_the_engines_own_message(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "failure",
            _settings(engine(exit_code=4, message="CUDA error: out of memory"), target),
        )

    assert raised.value.code == "DOCKING_GNINA_RUN_FAILED"
    assert "out of memory" in raised.value.context["output_tail"]
    assert raised.value.context["returncode"] == 4


def test_a_receptor_that_changed_since_the_cascade_was_written_stops_the_run(
    tmp_path: Path, engine: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "mismatch", _settings(engine(), target, receptor_sha256="f" * 64))

    assert raised.value.code == "DOCKING_RECEPTOR_DIGEST_MISMATCH"
    assert _record(engine) == {}


def test_a_receptor_that_is_not_pdb_is_refused(tmp_path: Path, engine: Any) -> None:
    """Its reader answers a format it does not know with no atoms, not an error."""

    target = _target(tmp_path, suffix=".cif")

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "cif", _settings(engine(), target))

    assert raised.value.code == "DOCKING_RECEPTOR_FORMAT_INVALID"
    assert _record(engine) == {}


def test_a_missing_executable_names_the_setting_that_fixes_it(tmp_path: Path) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "missing", _settings(tmp_path / "not-installed" / "gnina", target))

    assert raised.value.code == "DOCKING_EXECUTABLE_MISSING"
    assert "executable" in (raised.value.hint or "")


def test_the_population_cap_stops_the_run_before_the_first_ligand(
    tmp_path: Path, engine: Any
) -> None:
    """A cap noticed after the work has started is not a cap.

    It matters most here: this is the engine where the difference between a cap
    that holds and one that does not is measured in days.
    """

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
        _run(tmp_path, "relative", _settings(Path("gnina"), target))

    assert raised.value.code == "PLUGIN_CONFIG_INVALID"


def test_the_engine_is_registered_as_a_docking_stage() -> None:
    descriptor = create_builtin_registry().get("docking.gnina@0.1.0").descriptor

    assert descriptor.kind is PluginKind.DOCK
    assert descriptor.inputs == (PARENT_V1.id, LIGAND_CONFORMER_V1.id)
    assert descriptor.output_ports["scores"] == DOCKING_SCORE_V1.id
