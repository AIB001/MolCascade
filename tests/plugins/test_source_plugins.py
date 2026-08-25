from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from rdkit import Chem

import molcascade.plugins.builtin.sources as source_module
from molcascade.contracts import DECISION_V1, RAW_MOLECULE_V1
from molcascade.errors import PluginError
from molcascade.io.molecules import snapshot_regular_file
from molcascade.plugins.api import StageContext, StageRequest
from molcascade.plugins.builtin.sources import (
    DelimitedSmilesSourceConfig,
    DelimitedSmilesSourcePlugin,
    RawMoleculeParquetSourcePlugin,
    SDFSourcePlugin,
)


def _read_outputs(
    root: Path,
    response: object,
) -> tuple[pa.Table, pa.Table]:
    outputs = response.outputs  # type: ignore[attr-defined]
    raw = pq.read_table(root / outputs["primary"].file_paths[0])
    decisions = pq.read_table(root / outputs["decisions"].file_paths[0])
    RAW_MOLECULE_V1.validate(raw)
    DECISION_V1.validate(decisions)
    return raw, decisions


def _execute(
    plugin: object,
    config: dict[str, object],
    output: Path,
) -> tuple[object, pa.Table, pa.Table]:
    response = plugin.execute(  # type: ignore[attr-defined]
        StageRequest(stage_id="source", config=config),
        StageContext(output),
    )
    raw, decisions = _read_outputs(output, response)
    return response, raw, decisions


def test_delimited_source_preserves_utf8_csv_spelling_and_isolates_bad_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "molecules.csv"
    source.write_bytes(
        b"\xef\xbb\xbfsmiles,id,note\r\n"
        b'CCO,id-1,"caf\xc3\xa9, quoted"\r\n'
        b'"C\nC",id-2,"line one\r\nline two"\r\n'
        b",id-empty,missing\r\n"
        b"\r\n"
        b'"N""N",id-3,"escaped ""quote"""\r\n'
    )
    config = {
        "path": str(source),
        "delimiter": ",",
        "has_header": True,
        "smiles_column": "smiles",
        "candidate_id_column": "id",
        "metadata_columns": ["note"],
        "batch_size": 2,
    }

    response, raw, decisions = _execute(
        DelimitedSmilesSourcePlugin(),
        config,
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 5
    assert response.metadata["accepted_record_count"] == 3
    assert response.metadata["rejected_record_count"] == 2
    assert response.metadata["max_buffered_records"] <= 2
    rows = raw.to_pylist()
    assert [row["source_index"] for row in rows] == [0, 1, 4]
    assert [row["raw_smiles"] for row in rows] == ["CCO", "C\nC", 'N"N']
    assert [row["source_candidate_id"] for row in rows] == ["id-1", "id-2", "id-3"]
    metadata = json.loads(rows[0]["source_metadata_json"])
    assert metadata["columns"] == {"note": "café, quoted"}
    assert metadata["source_blob_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    rejected = [row for row in decisions.to_pylist() if row["outcome"] == "REJECT"]
    assert {row["reason_code"] for row in rejected} == {
        "DELIMITED_EMPTY_RECORD",
        "DELIMITED_SMILES_EMPTY",
    }
    for row in rejected:
        detail = json.loads(row["detail"])
        assert detail["record_sha256"]
        assert detail["byte_length"] > 0
        assert detail["byte_start"] >= 0


def test_delimited_ids_survive_file_move_and_distinguish_duplicate_occurrences(
    tmp_path: Path,
) -> None:
    content = b"smiles\tid\nCCO\ta\nCCO\tb\n"
    first = tmp_path / "first" / "input.smi"
    second = tmp_path / "second" / "renamed.smi"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(content)
    shutil.copyfile(first, second)

    def ingest(path: Path, output: Path) -> pa.Table:
        _, raw, _ = _execute(
            DelimitedSmilesSourcePlugin(),
            {
                "path": str(path),
                "delimiter": "\t",
                "has_header": True,
                "smiles_column": "smiles",
                "candidate_id_column": "id",
                "batch_size": 1,
            },
            output,
        )
        return raw

    first_rows = ingest(first, tmp_path / "first-out").to_pylist()
    second_rows = ingest(second, tmp_path / "second-out").to_pylist()
    first_ids = [row["source_record_id"] for row in first_rows]
    second_ids = [row["source_record_id"] for row in second_rows]

    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert all(value.startswith("source-record:sha256:") for value in first_ids)


def test_delimited_unclosed_quote_and_invalid_utf8_are_structured_rejects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.csv"
    source.write_bytes(b"smiles,id\nCCO,ok\n\xff,bad\n\"CC,unterminated")

    response, raw, decisions = _execute(
        DelimitedSmilesSourcePlugin(),
        {
            "path": str(source),
            "delimiter": ",",
            "has_header": True,
            "smiles_column": "smiles",
            "candidate_id_column": "id",
        },
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 3
    assert response.metadata["accepted_record_count"] == raw.num_rows == 1
    assert response.metadata["rejected_record_count"] == 2
    assert {row["reason_code"] for row in decisions.to_pylist()} == {
        "DELIMITED_FRAMING_ERROR",
        "SOURCE_UTF8_INVALID",
    }


def test_delimited_framing_matches_skip_space_escape_and_eof_quote_dialect(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dialect.csv"
    source.write_bytes(
        b"smiles,note\n"
        b'CCO,  "quoted\ncontinuation"\n'
        b"CCN,escaped\\\ncontinuation\n"
        b'"CCC",closing quote at EOF'
    )

    response, raw, decisions = _execute(
        DelimitedSmilesSourcePlugin(),
        {
            "path": str(source),
            "delimiter": ",",
            "has_header": True,
            "smiles_column": "smiles",
            "metadata_columns": ["note"],
            "skipinitialspace": True,
            "escapechar": "\\",
        },
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 3
    assert decisions.num_rows == 0
    rows = raw.to_pylist()
    assert [row["raw_smiles"] for row in rows] == ["CCO", "CCN", "CCC"]
    assert [json.loads(row["source_metadata_json"])["columns"]["note"] for row in rows] == [
        "quoted\ncontinuation",
        "escaped\ncontinuation",
        "closing quote at EOF",
    ]


def test_delimited_record_identity_does_not_depend_on_header_extraction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "header.csv"
    source.write_bytes(b"smiles\nCCO\n")
    _, with_header, _ = _execute(
        DelimitedSmilesSourcePlugin(),
        {
            "path": str(source),
            "delimiter": ",",
            "has_header": True,
            "smiles_column": "smiles",
        },
        tmp_path / "header-out",
    )
    _, without_header, _ = _execute(
        DelimitedSmilesSourcePlugin(),
        {
            "path": str(source),
            "delimiter": ",",
            "has_header": False,
            "smiles_column": 0,
            "expected_column_count": 1,
        },
        tmp_path / "no-header-out",
    )

    header_row = with_header.to_pylist()[0]
    no_header_row = next(
        row for row in without_header.to_pylist() if row["raw_smiles"] == "CCO"
    )
    assert header_row["source_record_id"] == no_header_row["source_record_id"]
    assert header_row["source_index"] == 0
    assert no_header_row["source_index"] == 1


def test_delimited_cr_only_records_and_byte_budget_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "classic-mac.csv"
    source.write_bytes(
        b"smiles,note\r"
        + b"CCO," + b"x" * 700 + b"\r"
        + b"CCN," + b"y" * 700 + b"\r"
        + b"CCC," + b"z" * 700 + b"\r"
    )

    response, raw, decisions = _execute(
        DelimitedSmilesSourcePlugin(),
        {
            "path": str(source),
            "delimiter": ",",
            "has_header": True,
            "smiles_column": "smiles",
            "metadata_columns": ["note"],
            "batch_size": 1_000_000,
            "max_record_bytes": 1024,
            "max_buffered_bytes": 4096,
        },
        tmp_path / "out",
    )

    assert raw.num_rows == 3
    assert decisions.num_rows == 0
    assert response.metadata["input_record_count"] == 3
    assert response.metadata["max_buffered_records"] < 3
    assert response.metadata["max_buffered_bytes"] <= 4096


def test_snapshot_does_not_delete_preexisting_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "snapshot.bin"
    source.write_bytes(b"source")
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        snapshot_regular_file(source, destination)

    assert destination.read_bytes() == b"sentinel"


def _molblock(smiles: str, *, crlf: bool = False) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    block = Chem.MolToMolBlock(molecule)
    return block.replace("\n", "\r\n") if crlf else block


def test_sdf_source_frames_exact_records_and_continues_after_bad_blocks(
    tmp_path: Path,
) -> None:
    first = (
        _molblock("CCO", crlf=True)
        + ">  <ID>\r\nid-1\r\n\r\n"
        + ">  <NOTE>\r\ntext $$$$ inside\r\n\r\n"
        + "$$$$\r\n"
    )
    empty = "$$$$\r\n"
    malformed = "broken mol block\r\n$$$$ trailing-vendor-text\r\n"
    final_unterminated = _molblock("CN") + ">  <ID>\nid-final\n\n"
    source = tmp_path / "records.sdf"
    source.write_bytes((first + empty + malformed + final_unterminated).encode("utf-8"))

    response, raw, decisions = _execute(
        SDFSourcePlugin(),
        {
            "path": str(source),
            "selected_properties": ["ID", "NOTE"],
            "candidate_id_property": "ID",
            "batch_size": 2,
        },
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 4
    assert response.metadata["accepted_record_count"] == 3
    assert response.metadata["rejected_record_count"] == 1
    assert response.metadata["warning_record_count"] == 2
    rows = raw.to_pylist()
    assert [row["source_index"] for row in rows] == [0, 2, 3]
    assert rows[0]["raw_molblock"] == _molblock("CCO", crlf=True)
    assert rows[0]["source_candidate_id"] == "id-1"
    assert "$$$$" not in rows[0]["raw_molblock"]
    assert rows[1]["raw_molblock"] == "broken mol block\r\n"
    assert rows[2]["raw_molblock"] == _molblock("CN")
    first_metadata = json.loads(rows[0]["source_metadata_json"])
    assert first_metadata["sdf_properties"] == {
        "ID": "id-1",
        "NOTE": "text $$$$ inside",
    }

    decision_rows = decisions.to_pylist()
    assert {row["reason_code"] for row in decision_rows} == {
        "SDF_EMPTY_RECORD",
        "SDF_MALFORMED_RECORD_FORWARDED",
    }
    warning_details = [
        json.loads(row["detail"])
        for row in decision_rows
        if row["outcome"] == "WARN"
    ]
    assert any("missing_m_end" in detail["issues"] for detail in warning_details)
    assert any("unterminated_record" in detail["issues"] for detail in warning_details)
    assert any("delimiter_trailing_data" in detail["issues"] for detail in warning_details)


def test_sdf_invalid_utf8_isolated_without_losing_following_record(tmp_path: Path) -> None:
    source = tmp_path / "invalid-utf8.sdf"
    source.write_bytes(b"\xff\n$$$$\n" + _molblock("CC").encode() + b"$$$$\n")

    response, raw, decisions = _execute(
        SDFSourcePlugin(),
        {"path": str(source)},
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 2
    assert raw.num_rows == 1
    assert decisions.to_pylist()[0]["reason_code"] == "SOURCE_UTF8_INVALID"
    assert raw.to_pylist()[0]["source_index"] == 1


def test_sdf_oversized_delimiter_line_is_bounded_and_following_record_survives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized-delimiter.sdf"
    source.write_bytes(
        b"broken\n$$$$" + b"x" * 70_000 + b"\n" + _molblock("CC").encode() + b"$$$$\n"
    )

    response, raw, decisions = _execute(
        SDFSourcePlugin(),
        {"path": str(source), "max_record_bytes": 4096},
        tmp_path / "out",
    )

    assert response.metadata["input_record_count"] == 2
    assert raw.num_rows == 1
    assert raw.to_pylist()[0]["source_index"] == 1
    assert decisions.to_pylist()[0]["reason_code"] == "SOURCE_RECORD_TOO_LARGE"


def _raw_row(record_id: str, index: int, smiles: str = "CCO") -> dict[str, object]:
    return {
        "source_record_id": record_id,
        "source_kind": "existing",
        "source_uri": None,
        "source_index": index,
        "generator_id": None,
        "batch_id": None,
        "source_candidate_id": None,
        "raw_smiles": smiles,
        "raw_molblock": None,
        "source_metadata_json": "{}",
    }


def test_parquet_source_streams_partitions_preserves_ids_and_fills_optional_columns(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    partial_schema = pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("source_kind", pa.string(), nullable=False),
            pa.field("source_index", pa.int64(), nullable=False),
            pa.field("raw_smiles", pa.large_string(), nullable=True),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_record_id": "source-b",
                    "source_kind": "existing",
                    "source_index": 1,
                    "raw_smiles": "CN",
                }
            ],
            schema=partial_schema,
        ),
        source / "part-b.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_record_id": "source-a",
                    "source_kind": "existing",
                    "source_index": 0,
                    "raw_smiles": "CCO",
                }
            ],
            schema=partial_schema,
        ),
        source / "part-a.parquet",
    )

    response, raw, decisions = _execute(
        RawMoleculeParquetSourcePlugin(),
        {"path": str(source), "batch_size": 1},
        tmp_path / "out",
    )

    assert [row["source_record_id"] for row in raw.to_pylist()] == [
        "source-a",
        "source-b",
    ]
    assert raw.schema.equals(RAW_MOLECULE_V1.schema, check_metadata=True)
    assert all(row["source_uri"] is None for row in raw.to_pylist())
    assert decisions.num_rows == 0
    assert response.metadata["input_file_count"] == 2
    assert response.metadata["input_record_count"] == 2
    assert response.metadata["max_buffered_records"] <= 1


def test_parquet_source_detects_directory_inventory_change_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist([_raw_row("first", 0)], schema=RAW_MOLECULE_V1.schema),
        source / "first.parquet",
    )
    original_snapshot = source_module.snapshot_regular_file

    def snapshot_then_add_partition(source_path: object, destination: object) -> object:
        snapshot = original_snapshot(source_path, destination)  # type: ignore[arg-type]
        if not (source / "late.parquet").exists():
            pq.write_table(
                pa.Table.from_pylist([_raw_row("late", 1)], schema=RAW_MOLECULE_V1.schema),
                source / "late.parquet",
            )
        return snapshot

    monkeypatch.setattr(source_module, "snapshot_regular_file", snapshot_then_add_partition)
    output = tmp_path / "out"

    with pytest.raises(PluginError) as caught:
        RawMoleculeParquetSourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(output),
        )

    assert caught.value.code == "PARQUET_SOURCE_DATASET_CHANGED"
    assert not list(output.rglob("*.parquet"))


def test_parquet_source_detects_partition_rewrite_after_its_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    for index, name in enumerate(("first.parquet", "second.parquet")):
        pq.write_table(
            pa.Table.from_pylist(
                [_raw_row(f"source-{index}", index)],
                schema=RAW_MOLECULE_V1.schema,
            ),
            source / name,
        )
    original_snapshot = source_module.snapshot_regular_file
    snapshot_count = 0

    def snapshot_then_rewrite_first(source_path: object, destination: object) -> object:
        nonlocal snapshot_count
        snapshot = original_snapshot(source_path, destination)  # type: ignore[arg-type]
        snapshot_count += 1
        if snapshot_count == 2:
            pq.write_table(
                pa.Table.from_pylist([_raw_row("rewritten", 99)], schema=RAW_MOLECULE_V1.schema),
                source / "first.parquet",
            )
        return snapshot

    monkeypatch.setattr(source_module, "snapshot_regular_file", snapshot_then_rewrite_first)

    with pytest.raises(PluginError) as caught:
        RawMoleculeParquetSourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(tmp_path / "out"),
        )

    assert caught.value.code == "PARQUET_SOURCE_DATASET_CHANGED"


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (
            lambda rows: [*rows, _raw_row("duplicate", 1), _raw_row("duplicate", 2)],
            "PARQUET_SOURCE_DUPLICATE_ID",
        ),
        (
            lambda rows: [*rows, _raw_row("negative", -1)],
            "PARQUET_SOURCE_INDEX_NEGATIVE",
        ),
        (
            lambda rows: [*rows, _raw_row("", 1)],
            "PARQUET_SOURCE_ID_INVALID",
        ),
        (
            lambda rows: [*rows, {**_raw_row("bad-json", 1), "source_metadata_json": "{bad"}],
            "PARQUET_SOURCE_METADATA_JSON_INVALID",
        ),
    ],
)
def test_parquet_source_rejects_dataset_wide_contract_holes_without_partial_output(
    tmp_path: Path,
    mutator: object,
    code: str,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    rows = mutator([])  # type: ignore[operator]
    midpoint = max(1, len(rows) // 2)
    pq.write_table(
        pa.Table.from_pylist(rows[:midpoint], schema=RAW_MOLECULE_V1.schema),
        source / "first.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(rows[midpoint:], schema=RAW_MOLECULE_V1.schema),
        source / "second.parquet",
    )
    output = tmp_path / "out"

    with pytest.raises(PluginError) as caught:
        RawMoleculeParquetSourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(output),
        )

    assert caught.value.code == code
    assert not list(output.rglob("*.parquet"))


def test_source_plugin_configuration_is_strict_and_frozen(tmp_path: Path) -> None:
    config = DelimitedSmilesSourceConfig(
        path=str(tmp_path / "input.csv"),
        delimiter=",",
        has_header=True,
        smiles_column="smiles",
    )
    with pytest.raises(ValidationError, match="frozen"):
        config.batch_size = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DelimitedSmilesSourceConfig.model_validate(
            {
                "path": "input.csv",
                "delimiter": ",",
                "has_header": True,
                "smiles_column": "smiles",
                "batch_size": "2",
            }
        )
    with pytest.raises(ValidationError):
        DelimitedSmilesSourceConfig.model_validate(
            {
                "path": "input.csv",
                "delimiter": ",",
                "has_header": False,
                "smiles_column": "smiles",
            }
        )


def test_source_plugins_reject_unexpected_stage_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("smiles\nCCO\n", encoding="utf-8")
    request = StageRequest(
        stage_id="source",
        inputs={"unexpected": object()},  # type: ignore[arg-type]
        config={
            "path": str(source),
            "delimiter": ",",
            "has_header": True,
            "smiles_column": "smiles",
        },
    )

    with pytest.raises(PluginError) as caught:
        DelimitedSmilesSourcePlugin().execute(request, StageContext(tmp_path / "out"))

    assert caught.value.code == "SOURCE_INPUTS_NOT_ALLOWED"
