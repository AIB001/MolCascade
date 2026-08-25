"""A screening run killed in the middle of a tier must restart, not repeat.

This is the promise the whole sharding change was made for: a cascade that dies
at molecule 1.9M of 2M has to pick up where it stopped.  ``tests/parallel`` and
``tests/plugins/test_sharded_execution.py`` already cover the two halves in
isolation -- that ``run_sharded`` banks completed shards, and that an adapter
actually reaches the checkpoint directory it was handed.  What neither can show
is that the *runner* wires those halves together: that a real failure inside a
middle stage leaves the earlier shards where the next attempt will find them,
that ``resume=True`` says in the audit log how much work it skipped, and -- the
part that matters most -- that resuming produces the same bytes as never having
been interrupted.

The failure is injected into a real adapter rather than a stub, because the
thing worth proving is that a stage half of whose shards came from disk and half
from this process commits exactly what a single uninterrupted pass would have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.artifacts import ArtifactDatasetRef
from molcascade.artifacts.hashing import sha256_file
from molcascade.config.models import PipelineConfig
from molcascade.parallel import ShardError, ShardOutcome, ShardTask, StageResources
from molcascade.parallel.shards import count_completed_shards
from molcascade.plugins import create_builtin_registry
from molcascade.plugins.builtin import property_gate
from molcascade.runtime import LocalRunner, RunStatus, StageRunStatus

#: Eight molecules, so two rows per shard cuts the gate under test into four and
#: "shard 2" is genuinely its middle.  Two of them are rejected -- ethanol on
#: mass, the sugar on lipophilicity -- which matters: a shard that emits fewer
#: rows than it read is one a reused checkpoint could get wrong, and a
#: pass-through population would hide that.
_MOLECULES = (
    "CCO",
    "c1ccccc1",
    "CCN(CC)CC",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Clc1ccccc1",
    "OCC1OC(O)C(O)C(O)C1O",
    "c1ccc2ccccc2c1",
    "NC(Cc1ccccc1)C(=O)O",
)

#: Two rows per shard, so eight molecules become four shards and "shard 2"
#: is a real middle of the stage rather than its last piece.
_SHARD_ROWS = 2

_KILLED_SHARD = 2


def _source_csv(tmp_path: Path) -> Path:
    path = tmp_path / "molecules.csv"
    path.write_text("smiles\n" + "\n".join(_MOLECULES) + "\n", encoding="utf-8")
    return path


def _pipeline(source: Path) -> PipelineConfig:
    """Source, standardize, the sharded gate under test, and one stage after it.

    ``property_gate`` has to be a *middle* stage for the test to mean anything:
    a resumed final stage proves only that the artifact was written, while a
    resumed middle stage has to hand its recombined output on to a successor
    that never noticed the interruption.
    """

    return PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "shard-resume",
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
                    "id": "property_gate",
                    "slot": "hard_gate",
                    "plugin": "chemistry.rdkit_property_range_gate@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "standardize", "port": "primary"}
                    ],
                    "config": {"mw_min": 50.0},
                },
                {
                    "id": "drug_likeness",
                    "slot": "hard_gate",
                    "plugin": "chemistry.rdkit_drug_likeness@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "property_gate", "port": "primary"}
                    ],
                    "config": {
                        "maximum_rule_of_five_violations": 0,
                        "minimum_qed": None,
                        "failure_action": "reject",
                    },
                },
            ],
        }
    )


def _digests(runner: LocalRunner, ref: ArtifactDatasetRef) -> dict[str, str]:
    """Every committed file of a dataset, by name, hashed.

    Bytes rather than row sets: shard outputs are concatenated by shard index
    precisely so that order survives parallel execution, and comparing the
    molecules alone would let a resume that reassembled them in completion
    order pass.
    """

    root = runner.store.resolve_dataset(ref, verify=True)
    return {path: sha256_file(root / path) for path in ref.file_paths}


def _stage_output(runner: LocalRunner, result: object, stage_id: str) -> ArtifactDatasetRef:
    stage = next(
        stage
        for stage in result.stages  # type: ignore[attr-defined]
        if stage.stage_id == stage_id
    )
    assert stage.output_ref is not None
    return stage.output_ref


def _events(runner: LocalRunner, run_id: str, event_type: str) -> list[dict[str, object]]:
    return [
        {"stage_id": event.stage_id, **event.details}
        for event in runner.read_events(run_id)
        if event.event_type == event_type
    ]


def test_a_stage_killed_on_shard_two_resumes_from_shard_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_csv(tmp_path)
    registry = create_builtin_registry()
    resources = StageResources(shard_rows=_SHARD_ROWS)

    # The control: the same cascade, the same shard plan, never interrupted.
    reference_runner = LocalRunner(
        tmp_path / "reference-workspace",
        plugins=registry,
        resources=resources,
    )
    reference = reference_runner.run(_pipeline(source), run_id="reference-run")
    assert reference.status is RunStatus.SUCCEEDED
    assert _events(reference_runner, "reference-run", "STAGE_SHARDS_RESUMED") == []
    assert _events(reference_runner, "reference-run", "STAGE_SHARDS_PLANNED") == [
        {
            "stage_id": "property_gate",
            "shard_count": 4,
            "shards_reused": 0,
            "execution_lanes": ["cpu"],
        },
        {
            # Six of the eight parents clear the mass gate, so the stage below
            # it is cut into three shards rather than four.
            "stage_id": "drug_likeness",
            "shard_count": 3,
            "shards_reused": 0,
            "execution_lanes": ["cpu"],
        },
    ]

    runner = LocalRunner(
        tmp_path / "interrupted-workspace",
        plugins=registry,
        resources=resources,
    )
    intact = property_gate._run_shard

    def killed(task: ShardTask) -> ShardOutcome:
        if task.index == _KILLED_SHARD:
            raise RuntimeError("simulated interruption inside shard 2")
        return intact(task)

    monkeypatch.setattr(property_gate, "_run_shard", killed)

    with pytest.raises(ShardError) as caught:
        runner.run(_pipeline(source), run_id="killed-run")

    assert caught.value.code == "SHARD_EXECUTION_FAILED"
    assert caught.value.context["shard_index"] == _KILLED_SHARD
    # The hint is the instruction the user acts on; if it stops naming resume,
    # the machinery below is unreachable in practice.
    assert "--resume" in (caught.value.hint or "")

    failed = runner.load_run("killed-run")
    assert failed.status is RunStatus.FAILED
    assert [stage.status for stage in failed.stages] == [
        StageRunStatus.SUCCEEDED,
        StageRunStatus.SUCCEEDED,
        StageRunStatus.FAILED,
        StageRunStatus.PENDING,
    ]

    # Shards 0 and 1 finished before the kill and were banked; shard 2 got as
    # far as its directory and left nothing in it.  Nothing was committed, so
    # the failure cost exactly the two shards that had not run yet.
    checkpoints = sorted(path for path in runner.checkpoints_root.iterdir() if path.is_dir())
    assert len(checkpoints) == 1
    assert count_completed_shards(checkpoints[0]) == (_KILLED_SHARD, _KILLED_SHARD + 1)
    assert list(runner.store.staging_root.iterdir()) == []

    monkeypatch.undo()
    resumed = runner.run(_pipeline(source), run_id="killed-run", resume=True)
    assert resumed.status is RunStatus.SUCCEEDED

    # The two earlier stages resume at stage granularity, the gate at shard
    # granularity, and the log says so for both.
    assert _events(runner, "killed-run", "STAGE_SHARDS_RESUMED") == [
        {"stage_id": "property_gate", "shards_reused": 2, "cache_key": checkpoints[0].name}
    ]
    assert _events(runner, "killed-run", "STAGE_SHARDS_PLANNED") == [
        {
            "stage_id": "property_gate",
            "shard_count": 4,
            "shards_reused": 2,
            "execution_lanes": ["cpu"],
        },
        {
            "stage_id": "drug_likeness",
            "shard_count": 3,
            "shards_reused": 0,
            "execution_lanes": ["cpu"],
        },
    ]

    # The claim the rest of this test exists to support.
    assert _digests(runner, _stage_output(runner, resumed, "property_gate")) == _digests(
        reference_runner, _stage_output(reference_runner, reference, "property_gate")
    )
    assert resumed.output_ref is not None
    assert reference.output_ref is not None
    assert _digests(runner, resumed.output_ref) == _digests(
        reference_runner, reference.output_ref
    )

    # A committed stage owns its shards no longer: leaving them would let a
    # later run pick pieces of a superseded computation back up.
    assert list(runner.checkpoints_root.iterdir()) == []
