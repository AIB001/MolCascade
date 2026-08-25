"""Sharding must be deterministic first and fast second.

The claim these tests defend is that splitting a stage's work changes *when*
rows are computed and never *what* is computed or where it lands.  A one-worker
run and an eight-worker run over the same input have to produce byte-identical
files in the same order, because ``Cardinality.ONE_TO_ONE`` stages and the
positional joins downstream would silently mis-pair molecules otherwise -- and a
mis-paired ADMET prediction is not an error anyone would notice.

The second claim is that an interrupted stage keeps what it finished.  A tier
that dies at molecule 1.9M of 2M and restarts from molecule 0 is the failure
mode this whole module exists to remove, so the resume tests deliberately kill
in the middle rather than at the end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from molcascade.errors import PluginError
from molcascade.parallel.models import ShardOutcome, ShardTask, StageResources
from molcascade.parallel.shards import (
    DONE_SUFFIX,
    SCRATCH_DIRNAME,
    ShardError,
    count_completed_shards,
    iter_shard_batches,
    plan_shards,
    run_sharded,
)

SCHEMA = pa.schema([("parent_id", pa.string()), ("value", pa.int64())])

#: Set by a test through the environment so the module-level worker can be
#: spawned by name.  A closure would be the natural way to say "fail on shard 3"
#: and is exactly what ``spawn`` cannot pickle.
FAIL_SHARD_ENV = "MOLCASCADE_TEST_FAIL_SHARD"


def _write_input(path: Path, rows: int, *, row_group_size: int = 100) -> Path:
    table = pa.table(
        {
            "parent_id": [f"P{index:06d}" for index in range(rows)],
            "value": list(range(rows)),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def double_values(task: ShardTask) -> ShardOutcome:
    """A worker with no cross-row state, which is the class this module serves.

    Module-level and importable by name because ``spawn`` re-imports it in a
    fresh interpreter; this is the shape every converted adapter has to take.
    """

    failing = os.environ.get(FAIL_SHARD_ENV)
    if failing is not None and int(failing) == task.index:
        raise RuntimeError(f"deliberate failure on shard {task.index}")

    rows = 0
    with pq.ParquetWriter(task.output_paths["primary"], SCHEMA, compression="zstd") as writer:
        for batch in iter_shard_batches(task):
            rows += batch.num_rows
            writer.write_table(
                pa.table(
                    {
                        "parent_id": batch.column("parent_id"),
                        "value": pc.multiply(batch.column("value"), 2),
                    },
                    schema=SCHEMA,
                )
            )
    return ShardOutcome(rows_in=rows, rows_out={"primary": rows})


#: Same trick as :data:`FAIL_SHARD_ENV`, for the worker that fails the way an
#: adapter does: with a verdict about a molecule rather than a crash.
DIAGNOSE_SHARD_ENV = "MOLCASCADE_TEST_DIAGNOSE_SHARD"


def diagnose_values(task: ShardTask) -> ShardOutcome:
    """Reject one shard's data the way a real adapter rejects a bad parent."""

    diagnosing = os.environ.get(DIAGNOSE_SHARD_ENV)
    if diagnosing is not None and int(diagnosing) == task.index:
        raise PluginError(
            "registered parent cannot be parsed",
            code="TEST_PARENT_INVALID",
            hint="Standardize the library before this stage.",
            context={"parent_id": "P000042"},
        )
    return double_values(task)


def _read_all(staging: Path, result_paths: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in result_paths:
        rows.extend(pq.read_table(staging / name).to_pylist())
    return rows


def _resources(checkpoint_dir: Path, **changes: object) -> StageResources:
    base = StageResources(shard_rows=100, batch_size=32, checkpoint_dir=checkpoint_dir)
    return base.replace(**changes)


def test_shard_ranges_are_contiguous_disjoint_and_complete(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input.parquet", 1000)
    ranges = plan_shards((source,), 300)

    assert ranges == ((0, 300), (300, 300), (600, 300), (900, 100))
    covered = [index for offset, count in ranges for index in range(offset, offset + count)]
    assert covered == list(range(1000))


def test_an_adapters_shard_size_is_a_ceiling_not_an_override(tmp_path: Path) -> None:
    """An expensive engine may only ever shard *more* finely than asked.

    A docking adapter declares a small constant so an interruption cannot cost
    hours; that reasoning gives it no standing to overrule a caller who wants
    finer shards still, since the only cost of a smaller unit is a few more
    parquet footers and the only cost of a larger one is lost work.
    """

    source = _write_input(tmp_path / "input.parquet", 300)
    staging = tmp_path / "staging"
    staging.mkdir()

    def _shards(*, adapter: int, requested: int) -> int:
        target = staging / f"a{adapter}-r{requested}"
        target.mkdir()
        result = run_sharded(
            worker=double_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=target,
            resources=StageResources(shard_rows=requested, batch_size=32),
            shard_rows=adapter,
        )
        return len(result.file_paths["primary"])

    assert _shards(adapter=100, requested=1_000) == 3
    assert _shards(adapter=100, requested=50) == 6


def test_an_empty_input_still_produces_one_shard(tmp_path: Path) -> None:
    """A stage that received nothing still has to write its schema.

    Zero shards would mean zero output files, and the next stage would fail on a
    missing dataset rather than on an empty one -- which is a different, and
    much less legible, error.
    """

    source = _write_input(tmp_path / "empty.parquet", 0)
    assert plan_shards((source,), 100) == ((0, 0),)


def test_shards_span_input_files_as_one_concatenation(tmp_path: Path) -> None:
    """Row ranges address the concatenation, not each file separately.

    An upstream stage that itself sharded hands down several files; treating
    them as independent would make shard size depend on how the *previous*
    stage happened to split, and the layout would drift down the funnel.
    """

    first = _write_input(tmp_path / "a.parquet", 150)
    second = _write_input(tmp_path / "b.parquet", 150)
    assert plan_shards((first, second), 100) == ((0, 100), (100, 100), (200, 100))

    crossing = ShardTask(
        index=1,
        input_files=(first, second),
        row_offset=100,
        row_count=100,
        output_paths={"primary": tmp_path / "out.parquet"},
        batch_size=32,
    )
    values = [
        row["value"] for batch in iter_shard_batches(crossing) for row in batch.to_pylist()
    ]
    # 50 from the tail of the first file, then 50 from the head of the second.
    assert values == list(range(100, 150)) + list(range(0, 50))


def test_a_shard_reads_only_its_own_rows_across_row_group_boundaries(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input.parquet", 1000, row_group_size=137)
    task = ShardTask(
        index=0,
        input_files=(source,),
        row_offset=250,
        row_count=310,
        output_paths={"primary": tmp_path / "out.parquet"},
        batch_size=64,
    )
    values = [row["value"] for batch in iter_shard_batches(task) for row in batch.to_pylist()]
    assert values == list(range(250, 560))


def test_one_worker_and_many_workers_produce_identical_bytes(tmp_path: Path) -> None:
    """The determinism claim, stated as bytes rather than as intent.

    Equal row *sets* would not be enough: the funnel joins positionally in
    several places, so the order across files and the split between them both
    have to be reproducible, which is what comparing file-by-file digests
    actually tests.
    """

    source = _write_input(tmp_path / "input.parquet", 1000)

    def run(label: str, workers: int) -> tuple[tuple[str, ...], list[bytes]]:
        staging = tmp_path / f"staging-{label}"
        staging.mkdir()
        result = run_sharded(
            worker=double_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=staging,
            resources=_resources(
                tmp_path / label,
                workers=workers,
                devices=tuple("cpu" for _ in range(workers)),
            ),
        )
        names = result.file_paths["primary"]
        return names, [(staging / name).read_bytes() for name in names]

    serial_names, serial_bytes = run("serial", 1)
    parallel_names, parallel_bytes = run("parallel", 8)

    assert serial_names == parallel_names
    assert len(serial_names) == 10
    assert serial_bytes == parallel_bytes


def test_output_rows_stay_in_input_order(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input.parquet", 500)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=staging,
        resources=_resources(tmp_path / "ck", workers=4, devices=("cpu",) * 4),
    )
    rows = _read_all(staging, result.file_paths["primary"])
    assert [row["parent_id"] for row in rows] == [f"P{index:06d}" for index in range(500)]
    assert [row["value"] for row in rows] == [index * 2 for index in range(500)]


def test_a_single_shard_keeps_the_unsuffixed_file_name(tmp_path: Path) -> None:
    """Small runs must commit exactly what they committed before this change.

    Most of the suite screens a handful of molecules.  If sharding renamed their
    single output to ``parent-part-00000.parquet``, every one of those artifacts
    would change identity for no computational reason.
    """

    source = _write_input(tmp_path / "input.parquet", 3)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=staging,
        resources=_resources(tmp_path / "ck"),
    )
    assert result.file_paths["primary"] == ("parent.parquet",)
    assert (staging / "parent.parquet").is_file()


def test_a_failed_shard_keeps_its_finished_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of checkpointing, tested at the moment it has to pay off."""

    source = _write_input(tmp_path / "input.parquet", 1000)
    checkpoints = tmp_path / "ck"
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv(FAIL_SHARD_ENV, "3")

    with pytest.raises(ShardError) as raised:
        run_sharded(
            worker=double_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=staging,
            resources=_resources(checkpoints),
        )
    assert raised.value.code == "SHARD_EXECUTION_FAILED"
    assert raised.value.hint is not None and "--resume" in raised.value.hint

    done = sorted(path.name for path in checkpoints.glob(f"*{DONE_SUFFIX}"))
    assert done == [f"shard-{index:05d}{DONE_SUFFIX}" for index in range(3)]


@pytest.mark.parametrize("workers", [1, 4])
def test_an_adapters_own_diagnosis_survives_the_shard_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """Splitting the work must not blur what the work concluded.

    Sixteen adapters answer with a precise code -- an unparseable parent, a
    non-finite score -- and callers and tests both key on it.  Relabelling that
    as ``SHARD_EXECUTION_FAILED`` would make every one of those diagnoses
    dependent on how many cores the machine had.  The pool case matters
    separately from the inline one because the error is pickled there, and the
    default reduction of an exception drops everything but its message.
    """

    source = _write_input(tmp_path / "input.parquet", 1000)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv(DIAGNOSE_SHARD_ENV, "3")

    with pytest.raises(PluginError) as raised:
        run_sharded(
            worker=diagnose_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=staging,
            resources=_resources(
                tmp_path / "ck", workers=workers, devices=("cpu",) * workers
            ),
        )
    assert not isinstance(raised.value, ShardError)
    assert raised.value.code == "TEST_PARENT_INVALID"
    assert raised.value.hint == "Standardize the library before this stage."
    # The molecule that caused it, plus where it was -- neither replaces the other.
    assert raised.value.context["parent_id"] == "P000042"
    assert raised.value.context["shard_index"] == 3


def test_a_retry_recomputes_only_what_was_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_input(tmp_path / "input.parquet", 1000)
    checkpoints = tmp_path / "ck"
    first_staging = tmp_path / "staging-1"
    first_staging.mkdir()
    monkeypatch.setenv(FAIL_SHARD_ENV, "3")
    with pytest.raises(ShardError):
        run_sharded(
            worker=double_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=first_staging,
            resources=_resources(checkpoints),
        )

    monkeypatch.delenv(FAIL_SHARD_ENV)
    second_staging = tmp_path / "staging-2"
    second_staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=second_staging,
        resources=_resources(checkpoints),
    )

    assert result.shard_count == 10
    assert result.reused_count == 3
    rows = _read_all(second_staging, result.file_paths["primary"])
    assert [row["value"] for row in rows] == [index * 2 for index in range(1000)]


def test_a_resumed_stage_equals_one_that_was_never_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is only useful if the answer is the same one."""

    source = _write_input(tmp_path / "input.parquet", 1000)
    clean_staging = tmp_path / "clean"
    clean_staging.mkdir()
    clean = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=clean_staging,
        resources=_resources(tmp_path / "ck-clean"),
    )

    checkpoints = tmp_path / "ck-resumed"
    monkeypatch.setenv(FAIL_SHARD_ENV, "5")
    broken_staging = tmp_path / "broken"
    broken_staging.mkdir()
    with pytest.raises(ShardError):
        run_sharded(
            worker=double_values,
            input_files=(source,),
            output_paths={"primary": "parent.parquet"},
            staging_root=broken_staging,
            resources=_resources(checkpoints),
        )
    monkeypatch.delenv(FAIL_SHARD_ENV)
    resumed_staging = tmp_path / "resumed"
    resumed_staging.mkdir()
    resumed = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=resumed_staging,
        resources=_resources(checkpoints),
    )

    assert resumed.reused_count == 5
    assert clean.file_paths == resumed.file_paths
    for name in clean.file_paths["primary"]:
        assert (clean_staging / name).read_bytes() == (resumed_staging / name).read_bytes()


def test_a_marker_whose_output_was_truncated_is_not_honoured(tmp_path: Path) -> None:
    """A marker is a claim; the digest is the evidence.

    The interruption that makes resume necessary -- a kill, a full disk, a node
    eviction -- is exactly the one that leaves a parquet half written, so
    trusting the marker alone would resume from corrupt data.
    """

    source = _write_input(tmp_path / "input.parquet", 300)
    checkpoints = tmp_path / "ck"
    first = tmp_path / "staging-1"
    first.mkdir()
    run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=first,
        resources=_resources(checkpoints),
    )
    assert count_completed_shards(checkpoints) == (3, 3)

    victim = checkpoints / "shard-00001" / "primary.parquet"
    victim.write_bytes(victim.read_bytes()[:-64])

    second = tmp_path / "staging-2"
    second.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=second,
        resources=_resources(checkpoints),
    )
    assert result.reused_count == 2
    rows = _read_all(second, result.file_paths["primary"])
    assert [row["value"] for row in rows] == [index * 2 for index in range(300)]


def test_a_marker_whose_output_was_deleted_is_not_honoured(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input.parquet", 300)
    checkpoints = tmp_path / "ck"
    first = tmp_path / "staging-1"
    first.mkdir()
    run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=first,
        resources=_resources(checkpoints),
    )
    (checkpoints / "shard-00002" / "primary.parquet").unlink()

    second = tmp_path / "staging-2"
    second.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=second,
        resources=_resources(checkpoints),
    )
    assert result.reused_count == 2


def test_force_ignores_every_valid_checkpoint(tmp_path: Path) -> None:
    """``--force`` already discards the stage cache; checkpoints follow it.

    Otherwise a user re-running to escape a suspect result would get the
    suspect shards back.
    """

    source = _write_input(tmp_path / "input.parquet", 300)
    checkpoints = tmp_path / "ck"
    first = tmp_path / "staging-1"
    first.mkdir()
    run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=first,
        resources=_resources(checkpoints),
    )

    second = tmp_path / "staging-2"
    second.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=second,
        resources=_resources(checkpoints, reuse_checkpoints=False),
    )
    assert result.reused_count == 0


def test_a_checkpoint_written_for_different_ports_is_not_reused(tmp_path: Path) -> None:
    """The stage changed shape, so its old shards answer a different question."""

    source = _write_input(tmp_path / "input.parquet", 300)
    checkpoints = tmp_path / "ck"
    first = tmp_path / "staging-1"
    first.mkdir()
    run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=first,
        resources=_resources(checkpoints),
    )
    marker = checkpoints / f"shard-00000{DONE_SUFFIX}"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["digests"]["decisions"] = record["digests"]["primary"]
    marker.write_text(json.dumps(record), encoding="utf-8")

    second = tmp_path / "staging-2"
    second.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=second,
        resources=_resources(checkpoints),
    )
    assert result.reused_count == 2


def test_without_a_checkpoint_directory_nothing_is_left_in_staging(tmp_path: Path) -> None:
    """A resume-less run must commit the same inventory as a single-threaded one.

    ``ArtifactStore`` inventories every file under staging, so a leftover shard
    tree would be committed into the artifact.
    """

    source = _write_input(tmp_path / "input.parquet", 300)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=staging,
        resources=StageResources(shard_rows=100, batch_size=32),
    )
    assert not (staging / SCRATCH_DIRNAME).exists()
    assert sorted(path.name for path in staging.iterdir()) == sorted(
        result.file_paths["primary"]
    )
    assert result.metadata.get("resume_supported") is None


def test_staged_files_are_ordinary_single_linked_copies(tmp_path: Path) -> None:
    """``ArtifactStore`` refuses a staged file with ``st_nlink != 1``.

    Which is why checkpoints are copied rather than hard-linked, and why moving
    them is also wrong: a commit that then failed would leave the markers
    pointing at files that had already left the checkpoint tree.
    """

    source = _write_input(tmp_path / "input.parquet", 300)
    checkpoints = tmp_path / "ck"
    staging = tmp_path / "staging"
    staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=staging,
        resources=_resources(checkpoints),
    )
    for name in result.file_paths["primary"]:
        assert (staging / name).stat().st_nlink == 1
    # The checkpoint keeps its own copy until the artifact is committed.
    assert (checkpoints / "shard-00000" / "primary.parquet").is_file()


def test_the_result_reports_what_the_run_actually_did(tmp_path: Path) -> None:
    source = _write_input(tmp_path / "input.parquet", 250)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = run_sharded(
        worker=double_values,
        input_files=(source,),
        output_paths={"primary": "parent.parquet"},
        staging_root=staging,
        resources=_resources(
            tmp_path / "ck",
            workers=2,
            devices=("cuda:0", "cuda:1"),
            notes=("one lane per detected NVIDIA device (2)",),
        ),
    )
    metadata = result.response_metadata()
    assert metadata["shard_count"] == 3
    assert metadata["shards_reused"] == 0
    assert metadata["rows_in"] == 250
    assert metadata["execution_lanes"] == ["cuda:0", "cuda:1"]
    assert result.rows_out == {"primary": 250}
    assert "one lane per detected NVIDIA device (2)" in metadata["execution_notes"]
