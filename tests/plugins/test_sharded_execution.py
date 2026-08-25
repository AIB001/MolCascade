"""A stage split across processes must decide exactly what one process decided.

Sharding changes how a stage runs and must change nothing about what it
produces.  Two properties carry that claim: the same molecules survive in the
same order regardless of worker count, and a stage that dies part-way leaves
enough behind that the retry does not repeat the work that already succeeded.

Both are asserted against a real adapter rather than a stub worker, because the
thing that can quietly break is not the shard machinery -- ``tests/parallel``
covers that -- but an adapter that turns out to have carried state across
batches without anyone noticing.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.parallel import StageResources, count_completed_shards
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.property_gate import RDKitPropertyRangeGatePlugin

_SMILES = (
    "CCO",
    "c1ccccc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCN(CC)CC",
    "OCC1OC(O)C(O)C(O)C1O",
    "Clc1ccccc1",
    "CCCCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)cccc2",
    "NC(Cc1ccccc1)C(=O)O",
)


def _row(index: int, smiles: str) -> dict[str, object]:
    return {
        "parent_id": f"parent-{index:03d}",
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{index:03d}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(root: Path, *, count: int = 40) -> StageInput:
    root.mkdir(parents=True, exist_ok=True)
    rows = [_row(index, _SMILES[index % len(_SMILES)]) for index in range(count)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARENT_V1.schema),
        root / "parents.parquet",
        # Several row groups, so a shard boundary genuinely falls inside one.
        row_group_size=7,
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "b" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(
    tmp_path: Path,
    name: str,
    resources: StageResources,
    *,
    stage_input: StageInput,
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    staging = tmp_path / name
    plugin = RDKitPropertyRangeGatePlugin()
    response = plugin.execute(
        StageRequest(
            stage_id="property_gate",
            inputs={"parents": stage_input},
            config={"mw_max": 200.0},
        ),
        StageContext(staging, resources),
    )
    parents = pa.concat_tables(
        [pq.read_table(staging / name) for name in response.outputs["primary"].file_paths]
    )
    decisions = pa.concat_tables(
        [pq.read_table(staging / name) for name in response.outputs["decisions"].file_paths]
    )
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    return (
        [str(row["parent_id"]) for row in parents.to_pylist()],
        decisions.to_pylist(),
        dict(response.metadata),
    )


def test_four_workers_reach_the_same_verdict_in_the_same_order(tmp_path: Path) -> None:
    """The claim that makes sharding safe to turn on by default.

    Order matters as much as membership: ``Cardinality.ONE_TO_ONE`` stages and
    every positional join downstream assume row order survives, and shard
    outputs are concatenated by shard index precisely so that it does.
    """

    stage_input = _input(tmp_path / "input")
    serial, serial_decisions, serial_metadata = _run(
        tmp_path,
        "serial",
        StageResources(shard_rows=1_000),
        stage_input=stage_input,
    )
    parallel, parallel_decisions, parallel_metadata = _run(
        tmp_path,
        "parallel",
        StageResources(workers=4, devices=("cpu",) * 4, shard_rows=7),
        stage_input=stage_input,
    )

    assert serial == parallel
    assert serial_decisions == parallel_decisions
    assert serial_metadata["policy_id"] == parallel_metadata["policy_id"]
    assert serial_metadata["input_count"] == parallel_metadata["input_count"] == 40
    assert serial_metadata["shard_count"] == 1
    assert parallel_metadata["shard_count"] == 6


def test_a_second_attempt_reuses_the_shards_the_first_one_finished(tmp_path: Path) -> None:
    """What the user asked for: an interrupted screening run restarts, not repeats.

    The first attempt is allowed to complete here -- a mid-flight kill is what
    ``tests/parallel/test_shards.py`` covers -- because the question this test
    asks is narrower and just as easy to get wrong: does an *adapter* built on
    ``run_sharded`` actually reach the checkpoint directory the runner handed
    it, and does a later attempt find that work rather than redo it?
    """

    stage_input = _input(tmp_path / "input")
    checkpoints = tmp_path / "checkpoints" / "cache-key"
    resources = StageResources(shard_rows=7, checkpoint_dir=checkpoints)

    first, first_decisions, _ = _run(tmp_path, "first", resources, stage_input=stage_input)
    assert count_completed_shards(checkpoints) == (6, 6)

    # Replacing every molecule with one the gate would reject, while keeping the
    # row count and therefore the shard plan identical, is what makes the reuse
    # observable: a second attempt that recomputed would pass nothing at all.
    pq.write_table(
        pa.Table.from_pylist(
            [_row(index, "CCCCCCCCCCCCCCCCCC(=O)O") for index in range(40)],
            schema=PARENT_V1.schema,
        ),
        tmp_path / "input" / "parents.parquet",
        row_group_size=7,
    )
    second, second_decisions, metadata = _run(
        tmp_path, "second", resources, stage_input=stage_input
    )

    assert second == first
    assert second_decisions == first_decisions
    assert metadata["shards_reused"] == 6
    assert metadata["resume_supported"] is True


def test_a_scratch_run_leaves_no_shard_tree_in_the_artifact(tmp_path: Path) -> None:
    """Without a checkpoint directory the shards are scratch, and must not commit."""

    stage_input = _input(tmp_path / "input", count=20)
    staging = tmp_path / "scratch"
    RDKitPropertyRangeGatePlugin().execute(
        StageRequest(
            stage_id="property_gate",
            inputs={"parents": stage_input},
            config={"mw_max": 200.0},
        ),
        StageContext(staging, StageResources(shard_rows=7)),
    )

    assert not (staging / ".shards").exists()
    assert sorted(path.name for path in staging.rglob("*.parquet")) == [
        "part-00000.parquet",
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
        "part-00002.parquet",
    ]
