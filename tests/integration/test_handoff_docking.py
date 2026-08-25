"""The docking sidecar written beside a shortlist.

A shortlist that came out of a docking tier should carry the evidence that
selected it, and the file formats a shortlist travels in have nowhere to put a
table.  So the export grows a sibling directory instead: one parquet per docking
stage, plus a merged best-pose table.

The engines themselves are not involved.  Uni-Dock, KarmaDock and GNINA are
external binaries that cannot run here, and none of what is under test in this
file depends on them: a stand-in plugin emits real ``docking_score/v1`` rows
through the real runner, and everything after that -- which rows are carried,
how they are named, what the merged table says, what happens on a second export
-- is MolCascade's own code.  Testing it against the real engines would test
CUDA, not this.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel

from molcascade.chemistry.datasets import iter_contract_batches, require_single_input
from molcascade.config import PipelineConfig
from molcascade.contracts import DOCKING_SCORE_V1, PARENT_V1
from molcascade.errors import PipelineError
from molcascade.handoff import materialize_shortlist
from molcascade.io.parquet import write_parquet_table
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PendingOutput,
    PluginDescriptor,
    PluginKind,
    StageContext,
    StageRequest,
    StageResponse,
    create_builtin_registry,
)
from molcascade.runtime import LocalRunner

_LIBRARY = "id,smiles\na,CCO\nb,c1ccccc1\nc,CC(=O)O\n"

#: A parent that no library row produced.  It stands in for a molecule that was
#: docked in this tier and then dropped by a later one: its scores exist in the
#: run and must not reach a bundle that describes the shortlist.
_GHOST = "parent:sha256:" + "0" * 64

_PARENT_PATH = "datasets/parents/part-00000.parquet"
_SCORE_PATH = "datasets/docking_scores/part-00000.parquet"


def _aligned(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Re-project a batch into a schema's exact column order and metadata."""

    return pa.RecordBatch.from_arrays(
        [batch.column(field.name) for field in schema], schema=schema
    )


class _FakeDockConfig(BaseModel):
    engine_id: str
    receptor_id: str = "sha256:" + "f" * 64


class _FakeDockPlugin:
    """Emit two ranked poses per parent, plus one for a molecule already gone.

    Scores are a function of the parent id rather than invented constants, so a
    row landing under the wrong molecule in the merged table shows up as a
    number that does not belong to it.
    """

    descriptor = PluginDescriptor(
        id="test.fake_dock",
        version="0.1.0",
        kind=PluginKind.DOCK,
        inputs=(PARENT_V1.id,),
        outputs=(PARENT_V1.id, DOCKING_SCORE_V1.id),
        output_ports={"primary": PARENT_V1.id, "scores": DOCKING_SCORE_V1.id},
        cardinality=Cardinality.ONE_TO_MANY,
        determinism=Determinism.DETERMINISTIC,
        display_name="Stand-in docking engine",
    )
    config_model = _FakeDockConfig

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        config = _FakeDockConfig.model_validate(dict(request.config))
        stage_input = require_single_input(request.inputs, contract_id=PARENT_V1.id)
        batches = list(iter_contract_batches(stage_input, PARENT_V1, batch_size=1024))
        parents = pa.Table.from_batches(
            [_aligned(batch, PARENT_V1.schema) for batch in batches],
            schema=PARENT_V1.schema,
        )
        parent_ids = [*parents.column("parent_id").to_pylist(), _GHOST]

        rows: dict[str, list[object]] = {name: [] for name in DOCKING_SCORE_V1.schema.names}
        for parent_id in parent_ids:
            for rank in (0, 1):
                seed = int(parent_id[-6:], 16) % 1000 / 100.0
                rows["parent_id"].append(parent_id)
                rows["engine_id"].append(config.engine_id)
                rows["receptor_id"].append(config.receptor_id)
                rows["pose_rank"].append(rank)
                rows["score"].append(-(seed + rank))
                rows["score_kind"].append("VINA_KCAL_MOL")
                rows["direction"].append("LOWER_STRONGER")
                rows["secondary_score"].append(None)
                rows["pose_molblock"].append(None)
                rows["method_id"].append(f"{config.engine_id}:test")

        scores = pa.Table.from_pydict(rows, schema=DOCKING_SCORE_V1.schema)
        write_parquet_table(parents, context.staging_root / _PARENT_PATH)
        write_parquet_table(scores, context.staging_root / _SCORE_PATH)
        return StageResponse(
            outputs={
                "primary": PendingOutput(PARENT_V1.id, (_PARENT_PATH,)),
                "scores": PendingOutput(DOCKING_SCORE_V1.id, (_SCORE_PATH,)),
            },
            metadata={"input_count": parents.num_rows, "output_count": parents.num_rows},
        )


def _run(
    tmp_path: Path,
    *,
    engines: tuple[str, ...],
    run_id: str = "docking-handoff",
) -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    stages: list[dict[str, object]] = [
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
            "config": {},
        },
    ]
    stages.extend(
        {
            "id": f"dock.{engine}",
            "slot": "dock",
            "plugin": "test.fake_dock@0.1.0",
            "config": {"engine_id": engine},
        }
        for engine in engines
    )
    stages.append(
        {
            "id": "shortlist",
            "slot": "export",
            "plugin": "export.native_smiles_shortlist@0.1.0",
            "config": {},
        }
    )
    config = PipelineConfig.model_validate(
        {"schema_version": 1, "name": "docking-handoff-test", "stages": stages}
    )
    registry = create_builtin_registry()
    registry.register(_FakeDockPlugin(), origin="test", trusted=True)
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    return runner, runner.run(config, run_id=run_id).run_id


def test_a_run_without_docking_writes_no_sidecar_at_all(tmp_path: Path) -> None:
    """The common case has to stay exactly as it was.

    Most cascades have no docking tier, and an empty directory appearing beside
    every shortlist would be a change to output that nobody asked for.
    """

    runner, run_id = _run(tmp_path, engines=())
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    assert result.docking is None
    assert sorted(item.name for item in tmp_path.iterdir() if item.is_dir()) == ["workspace"]


def test_the_sidecar_carries_one_file_per_docking_stage(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, engines=("unidock", "karmadock"))
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    assert result.docking is not None
    sidecar = result.docking
    assert sidecar.path == tmp_path / "shortlist-docking"
    # Named for the stage, not for the engine: one cascade may run the same
    # engine twice with different boxes, and the stage id is the name that
    # appears in the run report.
    assert sorted(item.name for item in sidecar.engine_files) == [
        "dock.karmadock.parquet",
        "dock.unidock.parquet",
    ]
    assert sidecar.engines == ("karmadock", "unidock")
    assert all(path.exists() for path in sidecar.engine_files)
    assert sidecar.best_pose_path.exists()

    # Three shortlisted molecules, two poses each, two stages.  The ghost's six
    # rows are the ones that must not be here.
    assert sidecar.pose_count == 12
    assert sidecar.best_pose_count == 6

    table = pq.read_table(sidecar.engine_files[0])
    DOCKING_SCORE_V1.validate(table)


def test_scores_for_molecules_that_left_the_funnel_are_not_carried(tmp_path: Path) -> None:
    """The bundle describes the shortlist, not the run.

    A molecule docked in one tier and dropped by a later one has real scores in
    the run's artifacts.  Carrying them here would make the sidecar disagree
    with the file it sits next to, and anyone joining the two on ``parent_id``
    would find rows on one side that are absent on the other.
    """

    runner, run_id = _run(tmp_path, engines=("unidock",))
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")
    assert result.docking is not None

    shortlisted = {
        line.split("\t")[1]
        for line in result.path.read_text(encoding="utf-8").splitlines()[1:]
    }
    carried = set(pq.read_table(result.docking.engine_files[0]).column("parent_id").to_pylist())
    assert carried == shortlisted
    assert _GHOST not in carried


def test_the_merged_table_is_one_best_pose_per_molecule_per_stage(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, engines=("unidock", "karmadock"))
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")
    assert result.docking is not None

    merged = pq.read_table(result.docking.best_pose_path)
    assert merged.num_rows == 6
    assert merged.column_names == [
        "parent_id",
        "stage_id",
        "engine_id",
        "receptor_id",
        "score",
        "score_kind",
        "direction",
        "secondary_score",
        "method_id",
    ]
    rows = merged.to_pylist()
    assert {row["stage_id"] for row in rows} == {"dock.unidock", "dock.karmadock"}
    # One row per (molecule, stage), and it is the rank-zero pose: the fake
    # engine makes rank 1 exactly one unit weaker, so the best score for a
    # molecule is the one whose fractional part matches its own seed.
    keys = [(row["parent_id"], row["stage_id"]) for row in rows]
    assert len(set(keys)) == len(keys)
    for row in rows:
        seed = int(row["parent_id"][-6:], 16) % 1000 / 100.0
        assert row["score"] == pytest.approx(-seed)
    # Without this column a consumer sorting ascending would rank KarmaDock's
    # MDN scores backwards, so it travels per row rather than being assumed.
    assert {row["direction"] for row in rows} == {"LOWER_STRONGER"}


def test_a_second_export_refuses_the_sidecar_unless_overwrite_is_given(
    tmp_path: Path,
) -> None:
    runner, run_id = _run(tmp_path, engines=("unidock",))
    first = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")
    assert first.docking is not None

    with pytest.raises(PipelineError) as caught:
        materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")
    assert caught.value.code == "HANDOFF_DOCKING_SIDECAR_EXISTS"

    again = materialize_shortlist(
        runner, run_id, tmp_path / "shortlist.smi", overwrite=True
    )
    assert again.docking is not None
    assert again.docking.pose_count == first.docking.pose_count


def test_an_undocked_export_clears_the_sidecar_it_writes_over(tmp_path: Path) -> None:
    """Two destinations, one export: they may never describe different runs.

    Writing an undocked shortlist over a docked one and leaving the directory
    alone would leave scores beside a file they have nothing to do with -- and
    every one of them would look, to anything joining on ``parent_id``, like
    evidence for the molecules now in that file.
    """

    runner, docked = _run(tmp_path, engines=("unidock",), run_id="docked")
    materialize_shortlist(runner, docked, tmp_path / "shortlist.smi")
    assert (tmp_path / "shortlist-docking").is_dir()

    later, plain = _run(tmp_path, engines=(), run_id="undocked")
    result = materialize_shortlist(
        later, plain, tmp_path / "shortlist.smi", overwrite=True
    )

    assert result.docking is None
    assert not (tmp_path / "shortlist-docking").exists()


def test_overwrite_will_not_delete_a_directory_molcascade_did_not_write(
    tmp_path: Path,
) -> None:
    """Permission to replace a bundle is not permission to empty a folder.

    The sidecar path is derived from the output name, so it can collide with a
    directory the user already had there.  Refusing is the only safe answer:
    the alternative is a flag that silently deletes whatever it finds.
    """

    runner, run_id = _run(tmp_path, engines=("unidock",))
    intruder = tmp_path / "shortlist-docking"
    intruder.mkdir()
    (intruder / "notes.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(PipelineError) as caught:
        materialize_shortlist(
            runner, run_id, tmp_path / "shortlist.smi", overwrite=True
        )
    assert caught.value.code == "HANDOFF_DOCKING_SIDECAR_FOREIGN"
    assert (intruder / "notes.txt").read_text(encoding="utf-8") == "mine"
    # And it refuses before writing anything, so the shortlist beside it is
    # still whatever it was -- here, absent.
    assert not (tmp_path / "shortlist.smi").exists()
