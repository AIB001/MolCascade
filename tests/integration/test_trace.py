"""Flattening a run into per-stage CSVs and docked-pose SDFs.

``molcascade trace`` answers a question no other command answers: not "what
survived" -- that is the export -- but "what did each block remove".  So what is
under test here is a reader, and the ways a reader of this shape goes wrong are
specific: it can attribute rows to the wrong stage, measure retention against
the wrong denominator, carry molecules that already left the funnel, or produce
a structure file whose coordinates are not the ones the engine computed.

The docking engines are not involved.  Uni-Dock, KarmaDock and GNINA are
external binaries that cannot run here, and nothing under test depends on them:
a stand-in plugin emits real ``docking_score/v1`` rows through the real runner
and the real threshold gate, and everything after that is MolCascade's own code.
The stand-in *does* build its poses with RDKit, which is exactly what a trace
must never do -- the point of the file is that its coordinates came from an
engine -- but here it is the engine, and the two ranks it emits are 100 A apart
so that a trace reading the wrong pose is visible as a coordinate rather than
inferred from a score.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from molcascade.chemistry.datasets import iter_contract_batches, require_single_input
from molcascade.config import PipelineConfig
from molcascade.contracts import BUILTIN_CONTRACTS, DOCKING_SCORE_V1, PARENT_V1, PREDICTION_V1
from molcascade.errors import ExecutionError, PipelineError
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
from molcascade.runtime import LocalRunner, RunStatus, StageRunStatus
from molcascade.trace import _STRUCTURE_COLUMNS, TracedRun, trace_run

#: Six molecules chosen so that every stage below removes a different subset,
#: and so that no two stages keep the same count -- a trace that attributed one
#: stage's rows to another would otherwise still add up.
_LIBRARY = (
    "id,smiles\n"
    "benzene,c1ccccc1\n"
    "acetic,CC(=O)O\n"
    "phenylalanine,NC(Cc1ccccc1)C(=O)O\n"
    "ethanol,CCO\n"
    "aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
    "triethylamine,CCN(CC)CC\n"
)

#: Standardized parent SMILES, which is what the stand-in engine is keyed on.
#: These molecules are neutral and already canonical, so the standardizer
#: returns them unchanged.
_BENZENE = "c1ccccc1"
_ACETIC = "CC(=O)O"
_PHENYLALANINE = "NC(Cc1ccccc1)C(=O)O"
_ETHANOL = "CCO"
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_TRIETHYLAMINE = "CCN(CC)CC"

#: Scores in kcal/mol, stronger when lower.  ``_ACETIC`` is scored and rejected
#: on the number; ``_ETHANOL`` is absent, so the gate rejects it for having no
#: evidence at all -- the two different ways a molecule leaves a docking tier.
_SCORES = {
    _BENZENE: -9.5,
    _ACETIC: -7.0,
    _PHENYLALANINE: -9.0,
    _ASPIRIN: -8.5,
    _TRIETHYLAMINE: -10.5,
}

#: Inclusive cutoff for the gate.  Four of the six molecules clear it.
_CUTOFF = -8.0

_GATE_SURVIVORS = {_BENZENE, _PHENYLALANINE, _ASPIRIN, _TRIETHYLAMINE}

#: Benzene weighs 78, so a 100-dalton floor after the docking gate leaves three.
#: The shortlist is then a strict subset of what the gate passed, which is the
#: only arrangement in which the two SDFs can be told apart.
_SHORTLIST = {_PHENYLALANINE, _ASPIRIN, _TRIETHYLAMINE}

#: A parent no library row produced, standing in for a molecule docked in this
#: tier and dropped by a later one.  Its score is good enough to pass, so
#: anything that carries it is carrying evidence for a molecule that is gone.
_GHOST = "parent:sha256:" + "0" * 64

#: Distance the rank-1 pose is translated along x.  Far enough that a centroid
#: says unambiguously which pose a record came from.
_DECOY_OFFSET = 100.0

#: Every parent the library produces, in the order the predictions below are
#: numbered from.  Position, not length or hash, so editing a SMILES here does
#: not silently renumber the values a test compares against.
_PARENTS = (_BENZENE, _ACETIC, _PHENYLALANINE, _ETHANOL, _ASPIRIN, _TRIETHYLAMINE)

#: Endpoints for the stand-in predictor, deliberately out of alphabetical order:
#: the trace sorts its key values, so the header must come out the same whatever
#: order the model emitted its rows in.
_ENDPOINTS = ("herg_blocking", "ames_mutagenicity", "aqueous_solubility")

# The default cascade's ADMET tier asks two models the same question in two
# units, so these two travel together through one stage and must not merge.
_HERG_ENDPOINTS = ("herg_blocking", "herg_pic50")

# Appended rather than inserted, so every value the three original endpoints
# produce is the number it was before.
_KNOWN_ENDPOINTS = (*_ENDPOINTS, "herg_pic50")

_MODEL_ID = "test-predictor:sha256:" + "a" * 64

_PARENT_PATH = "datasets/parents/part-00000.parquet"
_SCORE_PATH = "datasets/docking_scores/part-00000.parquet"
_PREDICTION_PATH = "datasets/predictions/part-00000.parquet"


def _prediction(smiles: str, endpoint: str) -> float:
    """A value unique to one (molecule, endpoint) pair.

    Twenty-four endpoints collapsed into one column is not a crash; it is a
    number in the right place that belongs to a different question.  So every
    cell here is distinguishable from every other, and eighths keep the values
    exact in binary so a CSV round trip compares equal rather than close.
    """

    return _PARENTS.index(smiles) + _KNOWN_ENDPOINTS.index(endpoint) / 8


def _aligned(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Re-project a batch into a schema's exact column order and metadata."""

    return pa.RecordBatch.from_arrays([batch.column(field.name) for field in schema], schema=schema)


def _molblock(smiles: str, *, offset: float) -> str:
    """One deterministic 3D conformer, optionally translated along x."""

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0xC0FFEE
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, Point3D(position.x + offset, position.y, position.z))
    return str(Chem.MolToMolBlock(molecule))


class _PoseDockConfig(BaseModel):
    engine_id: str = "unidock"
    receptor_id: str = "sha256:" + "f" * 64
    #: Off, every pose is emitted without geometry -- which is what a real run
    #: looks like when the engine was not asked to keep its poses.
    keep_poses: bool = True
    fail: bool = False


class _PoseDockPlugin:
    """Emit two ranked poses per scored parent, plus one for a molecule gone.

    Rank 1 is the same conformer translated by ``_DECOY_OFFSET``, so a consumer
    that took the wrong pose lands 100 A away rather than merely reporting a
    slightly different number.
    """

    descriptor = PluginDescriptor(
        id="test.pose_dock",
        version="0.1.0",
        kind=PluginKind.DOCK,
        inputs=(PARENT_V1.id,),
        outputs=(PARENT_V1.id, DOCKING_SCORE_V1.id),
        output_ports={"primary": PARENT_V1.id, "scores": DOCKING_SCORE_V1.id},
        cardinality=Cardinality.ONE_TO_MANY,
        determinism=Determinism.DETERMINISTIC,
        display_name="Stand-in docking engine with poses",
    )
    config_model = _PoseDockConfig

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        config = _PoseDockConfig.model_validate(dict(request.config))
        if config.fail:
            raise RuntimeError("simulated engine crash inside the docking tier")
        stage_input = require_single_input(request.inputs, contract_id=PARENT_V1.id)
        batches = list(iter_contract_batches(stage_input, PARENT_V1, batch_size=1024))
        parents = pa.Table.from_batches(
            [_aligned(batch, PARENT_V1.schema) for batch in batches],
            schema=PARENT_V1.schema,
        )
        scored = [
            *zip(
                parents.column("parent_id").to_pylist(),
                parents.column("parent_smiles").to_pylist(),
                strict=True,
            ),
            (_GHOST, _BENZENE),
        ]

        rows: dict[str, list[object]] = {name: [] for name in DOCKING_SCORE_V1.schema.names}
        for parent_id, smiles in scored:
            score = _SCORES.get(smiles)
            if score is None:
                continue
            for rank in (0, 1):
                rows["parent_id"].append(parent_id)
                rows["engine_id"].append(config.engine_id)
                rows["receptor_id"].append(config.receptor_id)
                rows["pose_rank"].append(rank)
                # Rank 1 is weaker, as a ranked pose list is.
                rows["score"].append(score + rank)
                rows["score_kind"].append("VINA_KCAL_MOL")
                rows["direction"].append("LOWER_STRONGER")
                rows["secondary_score"].append(None)
                rows["pose_molblock"].append(
                    _molblock(smiles, offset=rank * _DECOY_OFFSET) if config.keep_poses else None
                )
                rows["method_id"].append(f"{config.engine_id}:test")

        scores = pa.Table.from_pydict(rows, schema=DOCKING_SCORE_V1.schema)
        write_parquet_table(parents, context.staging_root / _PARENT_PATH)
        write_parquet_table(scores, context.staging_root / _SCORE_PATH)
        return StageResponse(
            outputs={
                "primary": PendingOutput(PARENT_V1.id, (_PARENT_PATH,)),
                "scores": PendingOutput(DOCKING_SCORE_V1.id, (_SCORE_PATH,)),
            },
            metadata={
                "input_count": parents.num_rows,
                "output_count": parents.num_rows,
            },
        )


class _PredictConfig(BaseModel):
    endpoints: tuple[str, ...] = _ENDPOINTS


class _PredictPlugin:
    """Emit one prediction row per (molecule, endpoint), as a real predictor does.

    ``prediction/v1`` is keyed on ``(parent_id, endpoint_id, model_id)`` and its
    own invariant forbids collapsing endpoints into one column, so a table of
    this shape is the case a one-row-per-molecule reader gets wrong.  The four
    optional columns are left empty throughout, which is what ADMET-AI actually
    writes: a model that reports no spread and no interval.
    """

    descriptor = PluginDescriptor(
        id="test.predict",
        version="0.1.0",
        kind=PluginKind.PREDICTOR,
        inputs=(PARENT_V1.id,),
        outputs=(PARENT_V1.id, PREDICTION_V1.id),
        output_ports={"primary": PARENT_V1.id, "predictions": PREDICTION_V1.id},
        cardinality=Cardinality.ONE_TO_ONE,
        determinism=Determinism.DETERMINISTIC,
        display_name="Stand-in multi-endpoint predictor",
    )
    config_model = _PredictConfig

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        config = _PredictConfig.model_validate(dict(request.config))
        stage_input = require_single_input(request.inputs, contract_id=PARENT_V1.id)
        batches = list(iter_contract_batches(stage_input, PARENT_V1, batch_size=1024))
        parents = pa.Table.from_batches(
            [_aligned(batch, PARENT_V1.schema) for batch in batches],
            schema=PARENT_V1.schema,
        )

        rows: dict[str, list[object]] = {name: [] for name in PREDICTION_V1.schema.names}
        for parent_id, smiles in zip(
            parents.column("parent_id").to_pylist(),
            parents.column("parent_smiles").to_pylist(),
            strict=True,
        ):
            for endpoint in config.endpoints:
                rows["parent_id"].append(parent_id)
                rows["endpoint_id"].append(endpoint)
                rows["model_id"].append(_MODEL_ID)
                rows["prediction_mean"].append(_prediction(smiles, endpoint))
                rows["prediction_std"].append(None)
                rows["interval_lower"].append(None)
                rows["interval_upper"].append(None)
                rows["calibration_id"].append(None)

        predictions = pa.Table.from_pydict(rows, schema=PREDICTION_V1.schema)
        write_parquet_table(parents, context.staging_root / _PARENT_PATH)
        write_parquet_table(predictions, context.staging_root / _PREDICTION_PATH)
        return StageResponse(
            outputs={
                "primary": PendingOutput(PARENT_V1.id, (_PARENT_PATH,)),
                "predictions": PendingOutput(PREDICTION_V1.id, (_PREDICTION_PATH,)),
            },
            metadata={
                "input_count": parents.num_rows,
                "output_count": parents.num_rows,
            },
        )


def _stages(
    source: Path,
    *,
    docking: bool,
    keep_poses: bool,
    fail: bool,
    id_column: bool,
) -> list[dict[str, object]]:
    """Source, standardize, two sibling gates, then optionally a docking tier.

    ``ro5_gate`` deliberately reads ``standardize`` rather than ``mass_gate``,
    which is what a parallel tier of criteria does: both siblings are handed the
    same population.  It sits *after* a stage that removed rows, so a report
    that measured retention against the previous line of itself would print
    120% here.
    """

    source_config: dict[str, object] = {
        "path": str(source),
        "smiles_column": "smiles",
    }
    if id_column:
        source_config["candidate_id_column"] = "id"
    stages: list[dict[str, object]] = [
        {
            "id": "source",
            "slot": "source",
            "plugin": "source.delimited_smiles@0.1.0",
            "config": source_config,
        },
        {
            "id": "standardize",
            "slot": "standardize",
            "plugin": "chemistry.rdkit_standardize@0.1.0",
            "inputs": [{"request_port": "primary", "stage": "source", "port": "primary"}],
        },
        {
            "id": "mass_gate",
            "slot": "hard_gate",
            "plugin": "chemistry.rdkit_property_range_gate@0.1.0",
            "inputs": [{"request_port": "primary", "stage": "standardize", "port": "primary"}],
            "config": {"mw_min": 50.0},
        },
        {
            "id": "ro5_gate",
            "slot": "hard_gate",
            "plugin": "chemistry.rdkit_drug_likeness@0.1.0",
            "inputs": [{"request_port": "primary", "stage": "standardize", "port": "primary"}],
            "config": {
                "maximum_rule_of_five_violations": 0,
                "minimum_qed": None,
                "failure_action": "reject",
            },
        },
    ]
    tail = "ro5_gate"
    if docking:
        stages.extend(
            (
                {
                    "id": "docking_score",
                    "slot": "dock",
                    "plugin": "test.pose_dock@0.1.0",
                    "inputs": [
                        {
                            "request_port": "primary",
                            "stage": "ro5_gate",
                            "port": "primary",
                        }
                    ],
                    "config": {"keep_poses": keep_poses, "fail": fail},
                },
                {
                    "id": "docking_score__gate",
                    "slot": "dock",
                    "plugin": "docking.numeric_evidence_gate@0.1.0",
                    "inputs": [
                        {
                            "request_port": "parents",
                            "stage": "docking_score",
                            "port": "primary",
                        },
                        {
                            "request_port": "evidence",
                            "stage": "docking_score",
                            "port": "scores",
                        },
                    ],
                    "config": {
                        "engine_id": "unidock",
                        "expected_score_kind": "VINA_KCAL_MOL",
                        "expected_direction": "LOWER_STRONGER",
                        "maximum": _CUTOFF,
                    },
                },
                {
                    "id": "weight_floor",
                    "slot": "hard_gate",
                    "plugin": "chemistry.rdkit_property_range_gate@0.1.0",
                    "inputs": [
                        {
                            "request_port": "primary",
                            "stage": "docking_score__gate",
                            "port": "primary",
                        }
                    ],
                    "config": {"mw_min": 100.0},
                },
            )
        )
        tail = "weight_floor"
    stages.append(
        {
            "id": "shortlist",
            "slot": "export",
            "plugin": "export.native_smiles_shortlist@0.1.0",
            "inputs": [{"request_port": "primary", "stage": tail, "port": "primary"}],
        }
    )
    return stages


def _run(
    tmp_path: Path,
    *,
    docking: bool = True,
    keep_poses: bool = True,
    fail: bool = False,
    id_column: bool = True,
    run_id: str = "trace-run",
) -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "trace-test",
            "stages": _stages(
                source,
                docking=docking,
                keep_poses=keep_poses,
                fail=fail,
                id_column=id_column,
            ),
        }
    )
    registry = create_builtin_registry()
    registry.register(_PoseDockPlugin(), origin="test", trusted=True)
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    if fail:
        with pytest.raises(ExecutionError):
            runner.run(config, run_id=run_id)
        return runner, run_id
    return runner, runner.run(config, run_id=run_id).run_id


#: The gate's window on ``herg_blocking``, whose values here are 0 through 5.
#: Four molecules clear it, so the survivors are a strict subset and the gate's
#: own CSV has something to be wrong about.
_HERG_CEILING = 3.0


def _predict_traced(tmp_path: Path, *, endpoints: tuple[str, ...] = _ENDPOINTS) -> TracedRun:
    """Source, standardize, predict, then one gate reading a single endpoint."""

    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "trace-prediction-test",
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
                    "inputs": [{"request_port": "primary", "stage": "source", "port": "primary"}],
                },
                {
                    "id": "admet",
                    "slot": "predictor",
                    "plugin": "test.predict@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "standardize", "port": "primary"}
                    ],
                    "config": {"endpoints": list(endpoints)},
                },
                {
                    "id": "admet__gate",
                    "slot": "predictor",
                    "plugin": "prediction.numeric_evidence_gate@0.1.0",
                    "inputs": [
                        {"request_port": "parents", "stage": "admet", "port": "primary"},
                        {"request_port": "evidence", "stage": "admet", "port": "predictions"},
                    ],
                    "config": {
                        "endpoint_id": endpoints[0],
                        "semantics_label": "stand-in endpoint, lower is safer",
                        "maximum": _HERG_CEILING,
                    },
                },
            ],
        }
    )
    registry = create_builtin_registry()
    registry.register(_PredictPlugin(), origin="test", trusted=True)
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    run_id = runner.run(config, run_id="trace-prediction").run_id
    return trace_run(runner, run_id, tmp_path / "trace")


def _traced(tmp_path: Path, **keywords: object) -> tuple[LocalRunner, TracedRun]:
    runner, run_id = _run(tmp_path, **keywords)  # type: ignore[arg-type]
    return runner, trace_run(runner, run_id, tmp_path / "trace")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parent_smiles(runner: LocalRunner, run_id: str, stage_id: str) -> list[str]:
    """The stage's own surviving molecules, read straight from its artifact."""

    state = runner.load_run(run_id)
    stage = next(item for item in state.stages if item.stage_id == stage_id)
    assert stage.output_ref is not None
    manifest = runner.store.get_manifest(stage.output_ref.artifact_id)
    port = next(output.port for output in manifest.outputs if output.contract_id == PARENT_V1.id)
    ref = manifest.dataset_ref(port)
    root = runner.store.resolve_dataset(ref, verify=True)
    smiles: list[str] = []
    for name in ref.file_paths:
        smiles.extend(
            pq.read_table(root / name, columns=["parent_smiles"])
            .column("parent_smiles")
            .to_pylist()
        )
    return smiles


def _sdf_records(path: Path) -> list[Chem.Mol]:
    with Chem.SDMolSupplier(str(path), removeHs=False) as supplier:
        molecules = [molecule for molecule in supplier if molecule is not None]
    return molecules


def _centroid_x(molecule: Chem.Mol) -> float:
    conformer = molecule.GetConformer()
    positions = [conformer.GetAtomPosition(index).x for index in range(molecule.GetNumAtoms())]
    return sum(positions) / len(positions)


# --------------------------------------------------------------------------
# One CSV per stage, holding that stage's own molecules


def test_every_stage_that_ran_leaves_one_csv_of_what_it_kept(tmp_path: Path) -> None:
    _, result = _traced(tmp_path)

    assert result.status == "SUCCEEDED"
    assert result.skipped == ()
    named = {stage.stage_id: stage for stage in result.stages}
    assert list(named) == [
        "source",
        "standardize",
        "mass_gate",
        "ro5_gate",
        "docking_score",
        "docking_score__gate",
        "weight_floor",
        "shortlist",
    ]
    assert result.csv_count == len(result.stages)

    for stage in result.stages:
        assert stage.csv_name is not None
        path = result.path / stage.csv_name
        assert path.is_file()
        rows = _rows(path)
        assert len(rows) == stage.row_count
        # The prefix is the execution order, so the directory reads top to
        # bottom in the order the funnel ran.
        assert stage.csv_name.startswith(f"{stage.order:02d}-")

    assert [named[key].row_count for key in ("source", "standardize")] == [6, 6]
    assert named["mass_gate"].row_count == 5
    assert named["ro5_gate"].row_count == 6
    assert named["docking_score"].row_count == 6
    assert named["docking_score__gate"].row_count == 4
    assert named["weight_floor"].row_count == 3
    assert named["shortlist"].row_count == 3


def test_the_csv_holds_the_stages_own_rows_in_their_own_order(tmp_path: Path) -> None:
    """A row-for-row comparison, not a set comparison.

    Parents are ordered by digest rather than by position in the library, and a
    reader that resolved the wrong dataset -- an upstream stage's, or one file of
    a sharded stage -- would still produce plausible SMILES in a plausible
    count.  Only the sequence rules that out.
    """

    runner, result = _traced(tmp_path)

    for stage in result.stages:
        if stage.stage_id == "source":
            continue
        assert stage.csv_name is not None
        traced = [row["smiles"] for row in _rows(result.path / stage.csv_name)]
        assert traced == _parent_smiles(runner, result.run_id, stage.stage_id)


def test_the_source_csv_is_the_library_before_parents_existed(tmp_path: Path) -> None:
    """The first stage runs before ``parent/v1`` exists, so it is traced as raw.

    Its key is the source record rather than a parent, and there is no parent
    SMILES to show -- only what the file said.
    """

    _, result = _traced(tmp_path)
    stage = result.stages[0]
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    assert list(rows[0]) == ["source_record_id", "smiles", "raw_format", "source_index"]
    assert [row["smiles"] for row in rows] == [
        line.split(",")[1] for line in _LIBRARY.splitlines()[1:]
    ]
    assert [row["source_index"] for row in rows] == [str(index) for index in range(6)]
    assert {row["raw_format"] for row in rows} == {"SMILES"}


# --------------------------------------------------------------------------
# Retention, measured against the population the stage was actually handed
#
# This is the arithmetic the command exists to get right, and the way it goes
# wrong is not subtle: siblings in a parallel tier all consume the tier's input,
# so a report that divides each line by the line above it prints ratios above
# 100% and stops being readable at all.


def test_retention_is_measured_against_the_stages_own_input(tmp_path: Path) -> None:
    _, result = _traced(tmp_path)
    named = {stage.stage_id: stage for stage in result.stages}

    # ``ro5_gate`` is ``mass_gate``'s sibling: both were handed the six
    # standardized parents, and ro5 rejected none of them.
    assert named["ro5_gate"].entering == 6
    assert named["ro5_gate"].kept_pct == pytest.approx(100.0)
    assert named["mass_gate"].entering == 6
    assert named["mass_gate"].kept_pct == pytest.approx(5 / 6 * 100.0)

    # Chaining down the list instead would have made this 6/5.
    assert all(stage.kept_pct is None or stage.kept_pct <= 100.0 for stage in result.stages)
    # The source stage has no population input, so it has no ratio to report
    # rather than a made-up one.
    assert named["source"].entering is None
    assert named["source"].kept_pct is None


def test_the_index_names_every_file_the_trace_wrote(tmp_path: Path) -> None:
    """Nothing on disk may be unaccounted for.

    The index covers the per-stage files; the run-level shortlist SDF is
    reported on the result instead, since it belongs to no single stage.
    Between the two, every file written must be named somewhere.
    """

    _, result = _traced(tmp_path)
    rows = _rows(result.index_path)

    assert [row["stage_id"] for row in rows] == [stage.stage_id for stage in result.stages]
    listed = {row["csv_file"] for row in rows} | {row["sdf_file"] for row in rows}
    listed.discard("")
    listed.add(result.index_path.name)
    assert result.shortlist_sdf is not None
    listed.add(result.shortlist_sdf.name)
    assert {item.name for item in result.path.iterdir()} == listed

    docking = next(row for row in rows if row["stage_id"] == "docking_score__gate")
    assert docking["rows"] == "4"
    assert docking["entering"] == "6"
    assert docking["kept_pct"] == "66.67"
    assert docking["plugin"] == "docking.numeric_evidence_gate@0.1.0"
    assert docking["status"] == "SUCCEEDED"
    assert docking["slot"] == "dock"
    # A hand-assembled pipeline has no tiers, so those columns stay empty
    # rather than inventing a grouping the run does not have.
    assert docking["tier_id"] == ""
    assert docking["tier_title"] == ""


# --------------------------------------------------------------------------
# The docking layer: the score beside the molecule it selected


def test_the_docking_csv_carries_the_score_that_decided_each_molecule(
    tmp_path: Path,
) -> None:
    _, result = _traced(tmp_path)
    named = {stage.stage_id: stage for stage in result.stages}

    scored_name = named["docking_score"].csv_name
    assert scored_name is not None
    scored = {row["smiles"]: row for row in _rows(result.path / scored_name)}
    assert set(scored) == {*_SCORES, _ETHANOL}
    for column in ("engine_id", "receptor_id", "score", "score_kind", "direction"):
        assert column in scored[_BENZENE]
    assert float(scored[_BENZENE]["score"]) == pytest.approx(_SCORES[_BENZENE])
    assert scored[_BENZENE]["score_kind"] == "VINA_KCAL_MOL"
    # Without this a consumer sorting ascending ranks a KarmaDock run backwards,
    # so it travels per row rather than being assumed.
    assert scored[_BENZENE]["direction"] == "LOWER_STRONGER"
    # Only the best pose: the stand-in makes rank 1 exactly one unit weaker.
    assert scored[_BENZENE]["pose_rank"] == "0"
    # A molecule the engine never scored is present and blank, not absent.  It
    # is why the next stage rejected it, and dropping the row would hide that.
    assert scored[_ETHANOL]["score"] == ""
    assert scored[_ETHANOL]["direction"] == ""

    gate_name = named["docking_score__gate"].csv_name
    assert gate_name is not None
    gated = _rows(result.path / gate_name)
    # The gate's own output is only molecules; the score is on the port it was
    # handed to decide with, and following that binding is what puts the two in
    # one file.
    assert {row["smiles"] for row in gated} == _GATE_SURVIVORS
    assert all(float(row["score"]) <= _CUTOFF for row in gated)


def test_a_molecule_that_left_the_funnel_reaches_no_file(tmp_path: Path) -> None:
    """The trace describes the run's populations, not everything scored in it.

    The stand-in scores a parent no library row produced, well enough to clear
    the cutoff.  Anything carrying it is carrying evidence for a molecule that
    is not there -- and joined on ``parent_id`` it would look like a hit.
    """

    _, result = _traced(tmp_path)

    for path in sorted(result.path.iterdir()):
        assert _GHOST not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Evidence keyed on more than the molecule: one column per key value
#
# Most evidence contracts are keyed on ``parent_id`` plus something -- an
# endpoint, a calculator, a fingerprint spec, a conformer index -- and whether
# that something takes one value or twenty-four is a property of the run rather
# than of the schema.  A reader that assumes one row per molecule does not fail
# on the second row; it keeps the first and drops the rest, which is a CSV that
# looks complete and answers a question nobody asked.


def _endpoint_columns(header: list[str]) -> list[str]:
    return [name for name in header if name in _ENDPOINTS]


def test_every_endpoint_reaches_the_csv_as_its_own_column(tmp_path: Path) -> None:
    result = _predict_traced(tmp_path)

    stage = next(item for item in result.stages if item.stage_id == "admet")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    assert len(rows) == len(_PARENTS)
    header = list(rows[0])
    # Sorted, not emission order: the plugin emits hERG first.
    assert _endpoint_columns(header) == sorted(_ENDPOINTS)
    for row in rows:
        for endpoint in _ENDPOINTS:
            assert float(row[endpoint]) == pytest.approx(_prediction(row["smiles"], endpoint))


def test_the_two_herg_endpoints_stay_two_columns(tmp_path: Path) -> None:
    """The default ADMET tier's two arms, and the merge that would go unnoticed.

    ``t6_admet`` runs ADMET-AI and OpenADMET on the same liability: the first
    answers with a blocking probability in 0-1, the second with a pIC50 around
    4-8.  Both are hERG, and naming both endpoints ``herg`` would have been the
    obvious thing to do -- which is why this is worth an explicit test.  One
    column holding both would not raise anywhere; it would hold whichever value
    the writer reached last, and the tier's own thresholds (0.95 for one arm,
    a pIC50 for the other) would then be applied to numbers from the wrong
    scale, silently.

    This also settles that the trace needed no change to carry the new tier:
    the fan-out is keyed on ``endpoint_id``, so two endpoints are two columns
    without anything being taught about hERG.
    """

    result = _predict_traced(tmp_path, endpoints=_HERG_ENDPOINTS)

    stage = next(item for item in result.stages if item.stage_id == "admet")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    header = list(rows[0])
    assert [name for name in header if name in _HERG_ENDPOINTS] == sorted(_HERG_ENDPOINTS)
    for row in rows:
        blocking = float(row["herg_blocking"])
        pic50 = float(row["herg_pic50"])
        assert blocking != pic50
        assert blocking == pytest.approx(_prediction(row["smiles"], "herg_blocking"))
        assert pic50 == pytest.approx(_prediction(row["smiles"], "herg_pic50"))


def test_a_key_that_takes_one_value_stays_an_ordinary_column(tmp_path: Path) -> None:
    """``model_id`` is part of the same primary key and does not fan out here.

    One model produced every row, so turning it into a column axis would give
    three columns named after a digest.  It stays a column of its own, after the
    ones that did fan out.
    """

    result = _predict_traced(tmp_path)

    stage = next(item for item in result.stages if item.stage_id == "admet")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    header = list(rows[0])
    assert header[-1] == "model_id"
    assert {row["model_id"] for row in rows} == {_MODEL_ID}


def test_a_value_column_no_row_filled_is_not_multiplied_into_empty_ones(
    tmp_path: Path,
) -> None:
    """Three endpoints times four unused columns is twelve columns of nothing."""

    result = _predict_traced(tmp_path)

    stage = next(item for item in result.stages if item.stage_id == "admet")
    assert stage.csv_name is not None
    header = list(_rows(result.path / stage.csv_name)[0])

    for empty in ("prediction_std", "interval_lower", "interval_upper", "calibration_id"):
        assert not [name for name in header if empty in name]


def test_the_gate_csv_shows_the_endpoint_it_decided_on(tmp_path: Path) -> None:
    """The failure this replaces: the gate's own basis was the missing column.

    Endpoints arrive alphabetically, so on a real ADMET run the one column that
    survived was ``human_intestinal_absorption`` while the gate was filtering on
    ``herg_blocking``.  The CSV reported neither the result nor the reason.
    """

    result = _predict_traced(tmp_path)

    stage = next(item for item in result.stages if item.stage_id == "admet__gate")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    assert len(rows) == 4
    assert _endpoint_columns(list(rows[0])) == sorted(_ENDPOINTS)
    assert all(float(row[_ENDPOINTS[0]]) <= _HERG_CEILING for row in rows)
    # And the molecules it dropped are the ones above the ceiling, so the column
    # explains the population rather than merely accompanying it.
    survivors = {row["smiles"] for row in rows}
    assert survivors == {
        smiles for smiles in _PARENTS if _prediction(smiles, _ENDPOINTS[0]) <= _HERG_CEILING
    }


def test_evidence_that_does_not_fan_out_keeps_the_header_it_always_had(
    tmp_path: Path,
) -> None:
    """One endpoint is one row per molecule, and must look exactly as it did.

    The rule is measured against the run, not read off the schema, so a
    ``prediction/v1`` table with a single endpoint takes the same path as
    ``property/v1`` or ``fingerprint/v1``: contract column order, empty columns
    and all.
    """

    result = _predict_traced(tmp_path, endpoints=("herg_blocking",))

    stage = next(item for item in result.stages if item.stage_id == "admet")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)

    assert list(rows[0])[-7:] == [
        "endpoint_id",
        "model_id",
        "prediction_mean",
        "prediction_std",
        "interval_lower",
        "interval_upper",
        "calibration_id",
    ]
    assert {row["endpoint_id"] for row in rows} == {"herg_blocking"}
    assert rows[0]["prediction_std"] == ""


def test_the_docking_csv_is_not_pivoted_by_the_keys_it_is_filtered_on(
    tmp_path: Path,
) -> None:
    """Docking evidence is keyed on engine, receptor and pose rank too.

    It reaches one row per molecule a different way -- the trace keeps only rank
    0 while scanning, and a stage runs one engine against one receptor -- so
    nothing varies by the time the header is decided.  Stated as a test because
    the alternative is a docking CSV whose columns are named after pose ranks.
    """

    _, result = _traced(tmp_path)

    for stage_id in ("docking_score", "docking_score__gate"):
        stage = next(item for item in result.stages if item.stage_id == stage_id)
        assert stage.csv_name is not None
        header = list(_rows(result.path / stage.csv_name)[0])
        assert header[-8:] == [
            "engine_id",
            "receptor_id",
            "pose_rank",
            "score",
            "score_kind",
            "direction",
            "secondary_score",
            "method_id",
        ]


# --------------------------------------------------------------------------
# The SDF: real coordinates, or no file


def test_the_gate_sdf_is_one_top_ranked_pose_per_surviving_molecule(
    tmp_path: Path,
) -> None:
    _, result = _traced(tmp_path)
    stage = next(item for item in result.stages if item.stage_id == "docking_score__gate")
    assert stage.sdf_name is not None
    path = result.path / stage.sdf_name

    content = path.read_text(encoding="utf-8")
    assert content.count("$$$$") == 4
    molecules = _sdf_records(path)
    assert len(molecules) == 4
    assert {
        molecule.GetProp("MOLCASCADE_PARENT_SMILES") for molecule in molecules
    } == _GATE_SURVIVORS
    assert all(
        molecule.GetProp("MOLCASCADE_PARENT_ID").startswith("parent:sha256:")
        for molecule in molecules
    )
    assert {molecule.GetProp("MOLCASCADE_DOCKING_STAGE") for molecule in molecules} == {
        "docking_score"
    }
    assert {molecule.GetProp("MOLCASCADE_SOURCE_ID") for molecule in molecules} == {
        "benzene",
        "phenylalanine",
        "aspirin",
        "triethylamine",
    }
    for molecule in molecules:
        smiles = molecule.GetProp("MOLCASCADE_PARENT_SMILES")
        assert float(molecule.GetProp("MOLCASCADE_DOCKING_SCORE")) == pytest.approx(_SCORES[smiles])
        assert molecule.GetProp("MOLCASCADE_DOCKING_DIRECTION") == "LOWER_STRONGER"

    # The coordinates are the engine's own, and they are the rank-zero pose:
    # rank 1 sits 100 A away, so reading the wrong row is not a rounding
    # difference.
    assert all(_centroid_x(molecule) < _DECOY_OFFSET / 2 for molecule in molecules)
    assert result.pose_count == 4 + 3


def test_the_shortlist_sdf_is_the_final_population_not_the_docked_one(
    tmp_path: Path,
) -> None:
    """Two SDFs, two questions.

    "What cleared the docking threshold" and "what came out of the cascade" are
    different sets whenever anything filters after the gate, and a trace that
    conflated them would answer the first while appearing to answer the second.
    """

    _, result = _traced(tmp_path)
    assert result.shortlist_sdf is not None

    molecules = _sdf_records(result.shortlist_sdf)
    assert len(molecules) == 3
    assert {molecule.GetProp("MOLCASCADE_PARENT_SMILES") for molecule in molecules} == _SHORTLIST
    assert _BENZENE not in {molecule.GetProp("MOLCASCADE_PARENT_SMILES") for molecule in molecules}
    # Same geometry as the gate's file for the molecules they share: the pose is
    # read once from the engine's output, not recomputed per file.
    gate = next(item for item in result.stages if item.sdf_name is not None)
    assert gate.sdf_name is not None
    by_smiles = {
        molecule.GetProp("MOLCASCADE_PARENT_SMILES"): Chem.MolToMolBlock(molecule)
        for molecule in _sdf_records(result.path / gate.sdf_name)
    }
    for molecule in molecules:
        smiles = molecule.GetProp("MOLCASCADE_PARENT_SMILES")
        assert Chem.MolToMolBlock(molecule) == by_smiles[smiles]


def test_a_run_without_docking_writes_no_sdf_and_says_why(tmp_path: Path) -> None:
    """Most cascades have no docking tier, and silence would be ambiguous."""

    _, result = _traced(tmp_path, docking=False)

    assert result.shortlist_sdf is None
    assert result.sdf_paths == ()
    assert result.pose_count == 0
    assert not list(result.path.glob("*.sdf"))
    assert any("No docking stage" in note for note in result.notes)
    assert result.csv_count == 5


def test_a_run_that_kept_no_poses_traces_its_csvs_and_explains_the_absence(
    tmp_path: Path,
) -> None:
    """Without ``keep_poses`` there is no geometry in the run to write.

    The only alternative would be embedding a conformer from SMILES, which
    produces a file that looks like a docking result, is not one, and cannot be
    told apart from one afterwards.  So: no file, and a reason.
    """

    _, result = _traced(tmp_path, keep_poses=False)

    assert result.shortlist_sdf is None
    assert result.sdf_paths == ()
    assert result.pose_count == 0
    # No empty file either -- an empty SDF is indistinguishable from a run that
    # docked nothing.
    assert not list(result.path.glob("*.sdf"))
    assert any("keep_poses" in note for note in result.notes)
    # The scores are still there; it is only the coordinates that are not.
    stage = next(item for item in result.stages if item.stage_id == "docking_score")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)
    assert len([row for row in rows if row["score"]]) == 5
    assert "pose_molblock" not in rows[0]


def test_no_molblock_is_ever_written_into_a_csv_cell(tmp_path: Path) -> None:
    """A quoted 3D structure keeps the file valid and makes it unreadable."""

    _, result = _traced(tmp_path)

    for path in sorted(result.path.glob("*.csv")):
        rows = _rows(path)
        assert "M  END" not in path.read_text(encoding="utf-8")
        assert all("pose_molblock" not in row for row in rows)


def test_every_contract_column_holding_a_structure_is_excluded_by_name() -> None:
    """The exclusion list is by name, so a new contract can be missed in silence.

    ``ligand_conformer/v1`` was: its column is plain ``molblock`` rather than
    ``pose_molblock``, nothing listed it, and every conformer CSV of every real
    run carried a quoted multi-line structure on each of its rows.  The pipeline
    this file builds has no conformer stage -- those need an engine -- so the
    only thing that catches it is walking the contracts themselves.
    """

    geometry = {
        field.name
        for contract in BUILTIN_CONTRACTS
        for field in contract.schema
        if field.name.endswith("molblock") or field.name == "structure_record"
    }

    assert geometry
    assert geometry <= _STRUCTURE_COLUMNS


# --------------------------------------------------------------------------
# Identifiers


def test_the_library_identifier_is_carried_into_every_stages_csv(
    tmp_path: Path,
) -> None:
    _, result = _traced(tmp_path)

    stage = next(item for item in result.stages if item.stage_id == "shortlist")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)
    assert list(rows[0])[:3] == ["parent_id", "smiles", "source_id"]
    assert {row["source_id"] for row in rows} == {
        "phenylalanine",
        "aspirin",
        "triethylamine",
    }


def test_a_library_without_identifiers_gets_no_identifier_column(
    tmp_path: Path,
) -> None:
    """Not a column of empty cells: there is nothing to put in it."""

    _, result = _traced(tmp_path, id_column=False)

    stage = next(item for item in result.stages if item.stage_id == "shortlist")
    assert stage.csv_name is not None
    rows = _rows(result.path / stage.csv_name)
    assert list(rows[0])[:2] == ["parent_id", "smiles"]
    assert "source_id" not in rows[0]
    assert result.shortlist_sdf is not None
    assert all(
        not molecule.HasProp("MOLCASCADE_SOURCE_ID")
        for molecule in _sdf_records(result.shortlist_sdf)
    )


# --------------------------------------------------------------------------
# Writing over things


def test_an_existing_directory_is_refused_until_overwrite_is_given(
    tmp_path: Path,
) -> None:
    runner, run_id = _run(tmp_path)
    destination = tmp_path / "trace"
    first = trace_run(runner, run_id, destination)
    stamp = (first.path / first.index_path.name).read_bytes()

    with pytest.raises(PipelineError) as caught:
        trace_run(runner, run_id, destination)
    assert caught.value.code == "TRACE_OUTPUT_EXISTS"
    assert first.index_path.read_bytes() == stamp

    again = trace_run(runner, run_id, destination, overwrite=True)
    assert again.index_path.read_bytes() == stamp
    assert again.csv_count == first.csv_count


def test_overwrite_will_not_delete_files_molcascade_did_not_write(
    tmp_path: Path,
) -> None:
    """Permission to replace a trace is not permission to empty a folder.

    The output path is whatever the user typed, so it can name a directory they
    already had.  Refusing is the only safe answer; the alternative is a flag
    that deletes whatever it finds.
    """

    runner, run_id = _run(tmp_path)
    destination = tmp_path / "trace"
    destination.mkdir()
    (destination / "notes.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(PipelineError) as caught:
        trace_run(runner, run_id, destination, overwrite=True)
    assert caught.value.code == "TRACE_OUTPUT_FOREIGN"
    assert caught.value.context["entry"] == "notes.txt"
    assert (destination / "notes.txt").read_text(encoding="utf-8") == "mine"
    # It refused before writing anything, and left no staging directory behind.
    assert list(destination.iterdir()) == [destination / "notes.txt"]
    assert not list(tmp_path.glob(".trace.*"))


def test_a_default_output_directory_is_named_for_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, run_id = _run(tmp_path, docking=False)
    monkeypatch.chdir(tmp_path)

    result = trace_run(runner, run_id)

    # Resolved, so the path it reports still names the directory once the
    # process that wrote it is gone.
    assert result.path == (tmp_path / f"{run_id}-trace").resolve()
    assert result.path.is_absolute()
    assert result.index_path.is_file()


# --------------------------------------------------------------------------
# The interrupted run, which is when this is most worth having


def test_an_interrupted_run_still_traces_the_layers_that_finished(
    tmp_path: Path,
) -> None:
    """The export refuses a run that is not ``SUCCEEDED``; a trace must not.

    A cascade that died in its docking tier is precisely the case where the
    layers above it are worth reading -- that is where the answer to "why did
    only six molecules reach the engine" lives.  So each stage is skipped on its
    own merits, and the ones that finished are written out.
    """

    runner, run_id = _run(tmp_path, fail=True)
    state = runner.load_run(run_id)
    assert state.status is RunStatus.FAILED

    result = trace_run(runner, run_id, tmp_path / "trace")

    assert result.status == "FAILED"
    traced = {stage.stage_id for stage in result.stages if stage.csv_name is not None}
    assert traced == {"source", "standardize", "mass_gate", "ro5_gate"}
    named = {stage.stage_id: stage for stage in result.stages}
    assert named["mass_gate"].row_count == 5
    assert named["ro5_gate"].kept_pct == pytest.approx(100.0)

    # The stage that died is reported as having nothing to trace rather than
    # being left out of the index, so the report says where the run stopped.
    assert dict(result.skipped)["docking_score"] == "stage produced no output"
    assert named["docking_score"].status == StageRunStatus.FAILED.value
    assert named["docking_score"].csv_name is None
    assert any("nothing to trace" in note for note in result.notes)

    # No poses were ever written, so no SDF claims otherwise.
    assert result.shortlist_sdf is None
    assert not list(result.path.glob("*.sdf"))
