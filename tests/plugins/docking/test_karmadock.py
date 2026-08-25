"""The KarmaDock adapter, tested at the process boundary rather than at the model.

KarmaDock is a Torch model in an environment that cannot coexist with this one,
so what is under test is the contract between MolCascade and that process: which
molecule each returned score belongs to, that the site definition and the seed
reach the command line, that nothing is written beside the operator's receptor,
and that a lane or a format this engine cannot use is refused before any work
starts.

The stand-in script is a fake of the *protocol*. It parses the real flags, reads
the real ``.smi`` file, derives a pocket next to ``--protein_file`` exactly as
KarmaDock does, and writes the real ``score.csv``. Its scores are a function of
each SMILES, which is what makes the identity mapping checkable -- KarmaDock
reports results under a name MolCascade invented, so an off-by-one in the way
those names are mapped back would otherwise produce a complete and entirely
wrong table.
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
from molcascade.contracts import DOCKING_SCORE_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.docking import KarmaDockPlugin
from molcascade.plugins.manifest import PluginKind
from molcascade.plugins.registry import create_builtin_registry

#: Distinct lengths on purpose: the fake's score is a function of the SMILES it
#: was handed, so a score landing on the wrong molecule shows up as a number.
_SMILES = (
    "CCO",
    "Clc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCCCCCCCCCCCCCCCC(=O)O",
)

#: One carbon at the origin. Enough for RDKit to read a pose back; the pose's
#: chemistry is KarmaDock's business, not this adapter's.
_MOLBLOCK = """lig
  molcascade-fake

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""


# ------------------------------------------------------------- the fake screener

_FAKE_SCREENING = '''\
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

# KarmaDock derives its pocket beside the protein it was given, and only when
# that exact file is absent. The fake does the same so the adapter's staging can
# be observed from the outside.
protein = Path(arguments["protein_file"][0])
pocket = Path(str(protein).replace(".pdb", "_pocket.pdb"))
call = dict(arguments)
call["pocket_existed"] = pocket.exists()
call["protein_dir"] = str(protein.parent)
record["calls"].append(call)
if not pocket.exists():
    pocket.write_text("REMARK derived pocket\\n")

if behaviour.get("exit_code"):
    record_path.write_text(json.dumps(record))
    sys.stdout.write(behaviour.get("message", "CUDA out of memory"))
    sys.exit(behaviour["exit_code"])

out_dir = Path(arguments["out_dir"][0])
out_dir.mkdir(parents=True, exist_ok=True)
skip = set(behaviour.get("skip", []))
nan_ff = set(behaviour.get("nan_ff", []))
rows = []
for line in Path(arguments["ligand_smi"][0]).read_text().split("\\n"):
    if not line.strip():
        continue
    smiles, name = line.split()
    ordinal = len(record["ligands"])
    record["ligands"].append({"name": name, "smiles": smiles})
    if ordinal in skip:
        continue
    score = float(len(smiles))
    corrected = "nan" if ordinal in nan_ff else "%.3f" % (score + 0.5)
    rows.append("%s,%.3f,%s,%.3f" % (name, score, corrected, score + 0.25))
    if "out_corrected" in arguments:
        (out_dir / (name + "_pred_ff_corrected.sdf")).write_text(_MOLBLOCK)

header = "pdb_id,karma_score,karma_score_ff,karma_score_aligned"
(out_dir / "score.csv").write_text("\\n".join([header, *rows]) + "\\n")
record_path.write_text(json.dumps(record))
'''


@pytest.fixture
def screener(tmp_path: Path) -> Any:
    """Return a factory that writes the fake checkout and returns its root."""

    repo = tmp_path / "KarmaDock"
    script = repo / "utils" / "virtual_screening.py"
    record_path = tmp_path / "record.json"

    def configure(**behaviour: Any) -> Path:
        behaviour.setdefault("record_path", str(record_path))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            f"_BEHAVIOUR = {json.dumps(json.dumps(behaviour))}\n"
            f"_MOLBLOCK = {json.dumps(_MOLBLOCK)}\n"
            f"{_FAKE_SCREENING}",
            encoding="utf-8",
        )
        return repo

    configure.record_path = record_path  # type: ignore[attr-defined]
    return configure


def _record(screener: Any) -> dict[str, Any]:
    path = Path(screener.record_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------- the inputs


class _Target(NamedTuple):
    receptor: Path
    digest: str
    reference: Path


def _target(tmp_path: Path, *, reference_suffix: str = ".mol2") -> _Target:
    """The receptor and the bound ligand that says where its site is.

    They live in their own directory so a test can assert that a run left it
    exactly as it found it.
    """

    directory = tmp_path / "target"
    directory.mkdir(parents=True, exist_ok=True)
    receptor = directory / "receptor.pdb"
    receptor.write_text("REMARK target\nATOM      1  N   ALA A   1\n", encoding="utf-8")
    reference = directory / f"ligand{reference_suffix}"
    reference.write_text("@<TRIPOS>ATOM\n1 C1 1.0 2.0 3.0 C.3\n", encoding="utf-8")
    return _Target(receptor, hashlib.sha256(receptor.read_bytes()).hexdigest(), reference)


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


def _settings(repo: Path, target: _Target, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "schema_version": 1,
        "executable": sys.executable,
        "repo_path": str(repo),
        "receptor_path": str(target.receptor),
        "reference_ligand_path": str(target.reference),
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
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    staging = tmp_path / name
    response = KarmaDockPlugin().execute(
        StageRequest(
            stage_id="docking",
            inputs={"parents": _parents(tmp_path / f"{name}-parents", smiles)},
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
    tmp_path: Path, screener: Any
) -> None:
    """KarmaDock answers under a name MolCascade invented, so the map back matters."""

    target = _target(tmp_path)
    rows, parents, metadata = _run(tmp_path, "basic", _settings(screener(), target))

    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert [row["parent_id"] for row in rows] == parents
    assert {row["engine_id"] for row in rows} == {"karmadock"}
    assert {row["receptor_id"] for row in rows} == {target.digest}
    assert {row["score_kind"] for row in rows} == {"KARMADOCK_MDN"}
    # Higher is better here, which is the opposite of every Vina-family score in
    # the same tier -- recording it per row is what keeps a gate honest.
    assert {row["direction"] for row in rows} == {"HIGHER_STRONGER"}
    assert {row["pose_rank"] for row in rows} == {0}
    assert metadata["scored_pose_count"] == len(_SMILES)

    # The engine's own view of what it was handed, checked against the answers.
    handed = _record(screener)["ligands"]
    assert [row["score"] for row in rows] == [float(len(item["smiles"])) for item in handed]
    assert [row["secondary_score"] for row in rows] == [
        float(len(item["smiles"])) + 0.5 for item in handed
    ]
    assert len({item["smiles"] for item in handed}) == len(_SMILES)


def test_the_target_and_the_seed_reach_the_command_line(tmp_path: Path, screener: Any) -> None:
    target = _target(tmp_path)
    _rows, _parents, metadata = _run(
        tmp_path, "command", _settings(screener(), target, engine_batch_size=8)
    )

    call = _record(screener)["calls"][0]
    assert call["crystal_ligand_file"] == [str(target.reference)]
    assert call["random_seed"] == ["4242"]
    assert call["batch_size"] == ["8"]
    assert Path(call["ligand_smi"][0]).name == "ligands.smi"
    # No box, ever. KarmaDock locates the site from the reference ligand, and a
    # box it silently ignored would look like a setting that had been applied.
    assert not any(key.startswith(("center_", "size_")) for key in call)
    assert metadata["reference_ligand_path"] == str(target.reference)
    assert metadata["pocket_source"] == "derived"


def test_the_pocket_is_derived_in_scratch_and_never_beside_the_receptor(
    tmp_path: Path, screener: Any
) -> None:
    """Two shards on two cards would otherwise race to write one derived file.

    KarmaDock writes its pocket next to whatever protein it was pointed at, so
    pointing it at the operator's receptor would both litter their directory and
    let one shard read a pocket another was halfway through writing.
    """

    target = _target(tmp_path)
    _run(tmp_path, "scratch", _settings(screener(), target))

    call = _record(screener)["calls"][0]
    assert call["pocket_existed"] is False
    assert call["protein_dir"] != str(target.receptor.parent)
    assert sorted(item.name for item in target.receptor.parent.iterdir()) == [
        "ligand.mol2",
        "receptor.pdb",
    ]


def test_a_supplied_pocket_replaces_the_one_karmadock_would_derive(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)
    pocket = tmp_path / "chosen_pocket.pdb"
    pocket.write_text("REMARK chosen pocket\nATOM      1  N   ALA A   1\n", encoding="utf-8")

    _rows, _parents, derived = _run(tmp_path, "derived", _settings(screener(), target))
    _rows, _parents, supplied = _run(
        tmp_path, "supplied", _settings(screener(), target, pocket_pdb_path=str(pocket))
    )

    assert _record(screener)["calls"][-1]["pocket_existed"] is True
    assert supplied["pocket_source"] == "configured"
    # Same target, different site definition: one protein, two computations.
    assert supplied["receptor_id"] == derived["receptor_id"]
    assert supplied["method_id"] != derived["method_id"]


def test_a_molecule_karmadock_returned_nothing_for_is_counted(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)
    rows, parents, metadata = _run(tmp_path, "unscored", _settings(screener(skip=[2]), target))

    # Docking declares no filter: rejecting for missing evidence is the score
    # gate's job, and it does it in the open.
    assert parents == [f"parent-{index:03d}" for index in range(len(_SMILES))]
    assert [row["parent_id"] for row in rows] == ["parent-000", "parent-001", "parent-003"]
    assert metadata["unscored_count"] == 1


def test_a_force_field_score_that_did_not_converge_is_absent_rather_than_nan(
    tmp_path: Path, screener: Any
) -> None:
    """A NaN compares false against every threshold, so it filters silently."""

    target = _target(tmp_path)
    rows, _parents, _metadata = _run(tmp_path, "nan", _settings(screener(nan_ff=[1]), target))

    assert [row["secondary_score"] is None for row in rows] == [False, True, False, False]
    # The primary score is untouched: only the correction failed.
    assert all(row["score"] is not None for row in rows)


def test_poses_are_not_stored_unless_they_were_asked_for(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)
    quiet, _parents, _metadata = _run(tmp_path, "quiet", _settings(screener(), target))
    kept, _parents, metadata = _run(
        tmp_path,
        "kept",
        _settings(screener(), target, keep_poses=True, pose_score_threshold=0.0),
    )

    assert [row["pose_molblock"] for row in quiet] == [None] * len(_SMILES)
    assert all(row["pose_molblock"] for row in kept)
    assert metadata["pose_missing_count"] == 0
    assert "out_corrected" not in _record(screener)["calls"][0]
    assert "out_corrected" in _record(screener)["calls"][-1]


def test_splitting_the_work_does_not_move_a_score(tmp_path: Path, screener: Any) -> None:
    target = _target(tmp_path)
    serial, _parents, serial_metadata = _run(
        tmp_path,
        "serial",
        _settings(screener(), target),
        resources=StageResources(devices=("cuda:0",), shard_rows=1_000),
    )
    split, _parents, split_metadata = _run(
        tmp_path,
        "split",
        _settings(screener(), target),
        resources=StageResources(workers=2, devices=("cuda:0", "cuda:1"), shard_rows=2),
    )

    assert serial_metadata["shard_count"] == 1
    assert split_metadata["shard_count"] == 2
    assert serial == split
    assert serial_metadata["method_id"] == split_metadata["method_id"]


def test_a_shard_is_one_call_however_small_the_read_batches_are(
    tmp_path: Path, screener: Any
) -> None:
    """Loading the model per batch would make the fastest engine the slowest."""

    target = _target(tmp_path)
    _run(tmp_path, "one-call", _settings(screener(), target, batch_size=1))

    assert len(_record(screener)["calls"]) == 1


def test_a_smiles_that_is_not_one_token_is_refused_rather_than_renamed(
    tmp_path: Path, screener: Any
) -> None:
    """``.smi`` is whitespace-separated, so a space would invent a molecule."""

    target = _target(tmp_path)
    rows, parents, metadata = _run(
        tmp_path,
        "spaced",
        _settings(screener(), target),
        smiles=(_SMILES[0], "CC O", _SMILES[2]),
    )

    assert len(parents) == 3
    assert [row["parent_id"] for row in rows] == ["parent-000", "parent-002"]
    assert metadata["unusable_smiles_count"] == 1


def test_a_non_zero_exit_arrives_as_the_engines_own_message(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "failure",
            _settings(screener(exit_code=2, message="CUDA error: out of memory"), target),
        )

    assert raised.value.code == "DOCKING_KARMADOCK_RUN_FAILED"
    assert "out of memory" in raised.value.context["output_tail"]
    assert raised.value.context["returncode"] == 2


def test_a_cpu_lane_is_refused_before_any_ligand_is_written(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            "cpu",
            _settings(screener(), target),
            resources=StageResources(devices=("cpu",), shard_rows=1_000),
        )

    assert raised.value.code == "DOCKING_GPU_REQUIRED"
    assert _record(screener) == {}


def test_a_receptor_that_changed_since_the_cascade_was_written_stops_the_run(
    tmp_path: Path, screener: Any
) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path, "mismatch", _settings(screener(), target, receptor_sha256="f" * 64)
        )

    assert raised.value.code == "DOCKING_RECEPTOR_DIGEST_MISMATCH"
    assert _record(screener) == {}


def test_a_reference_ligand_that_is_not_mol2_is_refused(
    tmp_path: Path, screener: Any
) -> None:
    """Its reader answers an unreadable format with no atoms, not with an error.

    An empty coordinate array becomes a pocket centred on the origin, which is a
    complete run against the wrong site -- the one failure this tier cannot see
    in its own output.
    """

    target = _target(tmp_path, reference_suffix=".sdf")

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "sdf-reference", _settings(screener(), target))

    assert raised.value.code == "DOCKING_REFERENCE_LIGAND_FORMAT_INVALID"
    assert _record(screener) == {}


def test_a_checkout_without_the_screening_script_names_what_is_missing(
    tmp_path: Path
) -> None:
    target = _target(tmp_path)
    empty = tmp_path / "not-karmadock"
    empty.mkdir()

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "no-script", _settings(empty, target))

    assert raised.value.code == "DOCKING_KARMADOCK_SCRIPT_MISSING"
    assert "virtual_screening.py" in raised.value.context["expected"]


def test_the_population_cap_stops_the_run_before_the_first_ligand(
    tmp_path: Path, screener: Any
) -> None:
    """A cap noticed after the work has started is not a cap."""

    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "cap", _settings(screener(), target, max_molecules=2))

    assert raised.value.code == "DOCKING_POPULATION_TOO_LARGE"
    assert raised.value.context["molecule_count"] == len(_SMILES)
    assert _record(screener) == {}


def test_a_relative_checkout_is_rejected_by_configuration(tmp_path: Path) -> None:
    target = _target(tmp_path)

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, "relative", _settings(Path("KarmaDock"), target))

    assert raised.value.code == "PLUGIN_CONFIG_INVALID"


def test_the_engine_is_registered_without_a_conformer_binding() -> None:
    """It builds its own geometry, so binding the shared table would be a fiction."""

    descriptor = create_builtin_registry().get("docking.karmadock@0.1.0").descriptor

    assert descriptor.kind is PluginKind.DOCK
    assert descriptor.inputs == (PARENT_V1.id,)
    assert descriptor.output_ports["scores"] == DOCKING_SCORE_V1.id
