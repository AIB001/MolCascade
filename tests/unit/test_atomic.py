from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from molcascade.artifacts import ArtifactProducer, LocalArtifactStore
from molcascade.io.atomic import (
    DestinationExistsError,
    atomic_commit_directory,
    atomic_write_bytes,
)
from molcascade.io.parquet import (
    ParquetValidationError,
    iter_parquet_batches,
    read_parquet_table,
    validate_parquet_file,
    write_parquet_batches,
    write_parquet_table,
)


def test_atomic_write_bytes_publishes_complete_content(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"

    returned = atomic_write_bytes(destination, b'{"complete":true}', overwrite=False)

    assert returned == destination
    assert destination.read_bytes() == b'{"complete":true}'
    assert list(destination.parent.glob("*.tmp")) == []


def test_atomic_write_without_overwrite_preserves_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("original", encoding="utf-8")

    with pytest.raises(DestinationExistsError):
        atomic_write_bytes(destination, b"replacement", overwrite=False)

    assert destination.read_text(encoding="utf-8") == "original"


def test_atomic_write_can_explicitly_replace_non_artifact_file(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    destination.write_bytes(b"old")

    atomic_write_bytes(destination, b"new", overwrite=True)

    assert destination.read_bytes() == b"new"


def test_atomic_directory_commit_moves_staging(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "committed"
    staging.mkdir()
    (staging / "payload").write_text("complete", encoding="utf-8")

    atomic_commit_directory(staging, destination)

    assert not staging.exists()
    assert (destination / "payload").read_text(encoding="utf-8") == "complete"


def test_atomic_directory_commit_never_overwrites_existing(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "committed"
    staging.mkdir()
    destination.mkdir()
    (staging / "payload").write_text("new", encoding="utf-8")
    (destination / "payload").write_text("old", encoding="utf-8")

    with pytest.raises(DestinationExistsError):
        atomic_commit_directory(staging, destination)

    assert staging.exists()
    assert (destination / "payload").read_text(encoding="utf-8") == "old"


def test_parquet_table_is_validated_before_atomic_publication(tmp_path: Path) -> None:
    destination = tmp_path / "molecules.parquet"
    table = pa.table(
        {
            "parent_id": pa.array(["p1", "p2"], type=pa.string()),
            "score": pa.array([0.5, 0.8], type=pa.float32()),
        }
    )

    summary = write_parquet_table(
        table,
        destination,
        metadata={"contract_id": "parent-record/v1"},
        row_group_size=1,
    )

    assert summary.row_count == 2
    assert summary.row_group_count == 2
    assert summary.schema_checksum.startswith("sha256:")
    assert read_parquet_table(destination).column_names == ["parent_id", "score"]
    assert sum(batch.num_rows for batch in iter_parquet_batches(destination)) == 2


def test_parquet_streaming_writer_checks_every_batch(tmp_path: Path) -> None:
    schema = pa.schema([pa.field("parent_id", pa.string(), nullable=False)])
    valid = pa.record_batch([pa.array(["p1"])], schema=schema)
    invalid = pa.record_batch([pa.array([1], type=pa.int64())], names=["parent_id"])
    destination = tmp_path / "molecules.parquet"

    with pytest.raises(ParquetValidationError, match="batch 1"):
        write_parquet_batches([valid, invalid], destination, schema=schema)

    assert not destination.exists()


def test_parquet_validation_detects_wrong_contract(tmp_path: Path) -> None:
    destination = tmp_path / "molecules.parquet"
    table = pa.table({"parent_id": ["p1", "p2"]})
    write_parquet_table(table, destination)
    wrong_schema = pa.schema([pa.field("parent_id", pa.int64())])

    with pytest.raises(ParquetValidationError, match="schema mismatch"):
        validate_parquet_file(destination, expected_schema=wrong_schema)


def test_parquet_writer_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "molecules.parquet"
    original = pa.table({"parent_id": ["p1"]})
    replacement = pa.table({"parent_id": ["p2"]})
    write_parquet_table(original, destination)

    with pytest.raises(DestinationExistsError):
        write_parquet_table(replacement, destination)

    assert read_parquet_table(destination)["parent_id"].to_pylist() == ["p1"]


def test_artifact_inventory_records_parquet_rows_and_schema(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    table = pa.table({"parent_id": ["p1", "p2"]})
    write_parquet_table(table, staging / "parents.parquet")

    manifest = store.commit(
        staging,
        kind="parent-record/v1",
        producer=ArtifactProducer(plugin_id="builtin.test", plugin_version="1.0.0"),
    )

    entry = manifest.files[0]
    assert entry.row_count == 2
    assert entry.schema_checksum is not None
    assert store.verify(manifest.artifact_id) == manifest
