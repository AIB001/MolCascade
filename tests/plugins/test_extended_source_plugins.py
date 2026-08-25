from __future__ import annotations

import json
import os
import shutil
import tracemalloc
import zipfile
from datetime import datetime
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import molcascade.plugins.builtin.sources as source_module
from molcascade.contracts import DECISION_V1, RAW_MOLECULE_V2
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageRequest
from molcascade.plugins.builtin.sources import (
    DelimitedSmilesSourceConfig,
    Mol2DirectorySourceConfig,
    Mol2DirectorySourcePlugin,
    XlsxSourceConfig,
    XlsxSourcePlugin,
)


def _execute(
    plugin: object,
    config: dict[str, object],
    output: Path,
) -> tuple[object, pa.Table, pa.Table]:
    response = plugin.execute(  # type: ignore[attr-defined]
        StageRequest(stage_id="source", config=config),
        StageContext(output),
    )
    raw = pq.read_table(output / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(output / response.outputs["decisions"].file_paths[0])
    RAW_MOLECULE_V2.validate(raw)
    DECISION_V1.validate(decisions)
    return response, raw, decisions


def _write_xlsx(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "候选分子"
    sheet.append(["smiles", "编号", "备注", "ignored"])
    sheet.append(["CCO", "候选-α", "café", None])
    sheet.append(["=CONCAT(\"C\",\"CN\")", "formula", "bad", None])
    sheet.append(["CCC", "error", None, None])
    sheet.cell(row=4, column=3, value="#DIV/0!").data_type = "e"
    sheet.append(["", "blank", "missing", None])
    sheet.append(["CCN", 42, datetime(2026, 8, 23, 1, 2, 3), "=1+1"])
    workbook.save(path)
    workbook.close()


def test_xlsx_source_streams_selected_cells_and_isolates_formula_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "molecules.xlsx"
    _write_xlsx(source)
    calls: list[dict[str, object]] = []
    original_load = openpyxl.load_workbook

    def audited_load(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(openpyxl, "load_workbook", audited_load)
    config = {
        "path": str(source),
        "sheet_name": "候选分子",
        "has_header": True,
        "smiles_column": "smiles",
        "candidate_id_column": "编号",
        "metadata_columns": ["备注"],
        "batch_size": 2,
    }

    response, raw, decisions = _execute(
        XlsxSourcePlugin(),
        config,
        tmp_path / "out",
    )

    assert len(calls) == 2
    assert {call["data_only"] for call in calls} == {True, False}
    assert all(call["read_only"] is True for call in calls)
    assert all(call["keep_links"] is False for call in calls)
    assert response.metadata["input_record_count"] == 5
    assert response.metadata["accepted_record_count"] == 2
    assert response.metadata["rejected_record_count"] == 3
    rows = raw.to_pylist()
    assert [row["raw_format"] for row in rows] == ["SMILES", "SMILES"]
    assert [row["raw_structure"] for row in rows] == ["CCO", "CCN"]
    assert [row["source_candidate_id"] for row in rows] == ["候选-α", "42"]
    assert [row["source_index"] for row in rows] == [0, 4]
    metadata = json.loads(rows[1]["source_metadata_json"])
    assert metadata["sheet_name"] == "候选分子"
    assert metadata["excel_row"] == 6
    assert metadata["metadata_cells"]["备注"]["type"] == "datetime"
    assert metadata["metadata_cells"]["备注"]["value"] == "2026-08-23T01:02:03"
    assert {row["reason_code"] for row in decisions.to_pylist()} == {
        "XLSX_CELL_ERROR",
        "XLSX_FORMULA_NOT_ALLOWED",
        "XLSX_SMILES_EMPTY",
    }

    moved = tmp_path / "moved" / "renamed.xlsm"
    moved.parent.mkdir()
    shutil.copyfile(source, moved)
    _, moved_raw, _ = _execute(
        XlsxSourcePlugin(),
        {**config, "path": str(moved)},
        tmp_path / "moved-out",
    )
    assert raw.column("source_record_id").to_pylist() == moved_raw.column(
        "source_record_id"
    ).to_pylist()


def test_xlsx_source_requires_explicit_valid_sheet_and_header(tmp_path: Path) -> None:
    source = tmp_path / "molecules.xlsx"
    _write_xlsx(source)

    with pytest.raises(PluginError) as caught:
        XlsxSourcePlugin().execute(
            StageRequest(
                stage_id="source",
                config={
                    "path": str(source),
                    "sheet_name": "missing",
                    "has_header": True,
                    "smiles_column": "smiles",
                },
            ),
            StageContext(tmp_path / "out"),
        )

    assert caught.value.code == "XLSX_SHEET_NOT_FOUND"


def test_source_configs_have_safe_cli_defaults() -> None:
    delimited = DelimitedSmilesSourceConfig()
    xlsx = XlsxSourceConfig()
    mol2 = Mol2DirectorySourceConfig()

    assert delimited.model_dump()["path"] == "molecules.csv"
    assert delimited.delimiter == ","
    assert delimited.has_header is True
    assert delimited.smiles_column == "smiles"
    assert delimited.candidate_id_column is None
    assert xlsx.model_dump()["path"] == "molecules.xlsx"
    assert xlsx.sheet_name is None
    assert xlsx.has_header is True
    assert xlsx.smiles_column == "smiles"
    assert xlsx.candidate_id_column is None
    assert mol2.model_dump()["path"] == "molecules"
    assert mol2.recursive is True
    assert mol2.candidate_id_mode == "RELATIVE_PATH"
    assert mol2.max_record_bytes <= mol2.max_buffered_bytes


def test_xlsx_default_selects_first_visible_sheet_and_rejects_all_hidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "visible.xlsx"
    workbook = openpyxl.Workbook()
    hidden = workbook.active
    hidden.title = "Hidden first"
    hidden.sheet_state = "hidden"
    visible = workbook.create_sheet("Visible second")
    visible.append(["smiles"])
    visible.append(["CCO"])
    workbook.active = 1
    workbook.save(source)
    workbook.close()

    response, raw, _ = _execute(
        XlsxSourcePlugin(),
        {"path": str(source)},
        tmp_path / "visible-out",
    )
    assert response.metadata["sheet_name"] == "Visible second"
    assert response.metadata["sheet_selection"] == "FIRST_VISIBLE"
    assert raw.column("raw_structure").to_pylist() == ["CCO"]

    all_hidden = tmp_path / "all-hidden.xlsx"
    shutil.copyfile(source, all_hidden)
    with zipfile.ZipFile(all_hidden, "r") as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(all_hidden, "w") as archive:
        for info, payload in members:
            if info.filename == "xl/workbook.xml":
                assert b'state="visible"' in payload
                payload = payload.replace(b'state="visible"', b'state="hidden"')
            archive.writestr(info, payload)

    with pytest.raises(PluginError) as caught:
        XlsxSourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(all_hidden)}),
            StageContext(tmp_path / "hidden-out"),
        )
    assert caught.value.code == "XLSX_VISIBLE_SHEET_NOT_FOUND"


def test_xlsx_rejects_nonempty_columns_beyond_the_declared_shape(tmp_path: Path) -> None:
    source = tmp_path / "extra-column.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["smiles", "candidate_id"])
    sheet.append(["CCO", "candidate-1", "must-not-be-truncated"])
    workbook.save(source)
    workbook.close()

    response, raw, decisions = _execute(
        XlsxSourcePlugin(),
        {
            "path": str(source),
            "has_header": True,
            "smiles_column": "smiles",
            "candidate_id_column": "candidate_id",
            "expected_column_count": 2,
        },
        tmp_path / "extra-column-out",
    )

    assert response.metadata["accepted_record_count"] == 0
    assert raw.num_rows == 0
    assert decisions.column("reason_code").to_pylist() == [
        "XLSX_COLUMN_COUNT_MISMATCH"
    ]


def _mol2(name: str) -> str:
    return (
        "@<TRIPOS>MOLECULE\n"
        f"{name}\n"
        "2 1 0 0 0\n"
        "SMALL\n"
        "NO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 C.3 1 MOL 0.0\n"
        "2 C2 1.5 0.0 0.0 C.3 1 MOL 0.0\n"
        "@<TRIPOS>BOND\n"
        "1 1 2 1\n"
    )


def test_mol2_directory_is_deterministic_move_stable_and_isolates_bad_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mol2-library"
    (source / "nested").mkdir(parents=True)
    (source / "z.mol2").write_text(_mol2("z"), encoding="utf-8")
    (source / "nested" / "a.MOL2").write_text(_mol2("a"), encoding="utf-8")
    (source / "empty.mol2").write_bytes(b"")
    (source / "invalid.mol2").write_bytes(b"\xff\n")
    (source / "oversized.mol2").write_bytes(b"x" * 5000)
    (source / "README.txt").write_text("ignored", encoding="utf-8")
    config = {
        "path": str(source),
        "recursive": True,
        "candidate_id_mode": "RELATIVE_PATH",
        "max_record_bytes": 1024,
        "max_buffered_bytes": 16_384,
        "batch_size": 100,
    }

    response, raw, decisions = _execute(
        Mol2DirectorySourcePlugin(),
        config,
        tmp_path / "out",
    )

    assert response.metadata["manifest_file_count"] == 5
    assert response.metadata["ignored_regular_file_count"] == 1
    assert response.metadata["accepted_record_count"] == 2
    assert response.metadata["rejected_record_count"] == 3
    rows = raw.to_pylist()
    relative_paths = [
        json.loads(row["source_metadata_json"])["relative_path"] for row in rows
    ]
    assert relative_paths == ["nested/a.MOL2", "z.mol2"]
    assert [row["raw_format"] for row in rows] == ["MOL2", "MOL2"]
    assert rows[0]["raw_structure"] == _mol2("a")
    assert [row["source_candidate_id"] for row in rows] == relative_paths
    assert {row["reason_code"] for row in decisions.to_pylist()} == {
        "MOL2_EMPTY_FILE",
        "SOURCE_FILE_TOO_LARGE",
        "SOURCE_UTF8_INVALID",
    }

    moved = tmp_path / "renamed-library"
    shutil.copytree(source, moved)
    _, moved_raw, _ = _execute(
        Mol2DirectorySourcePlugin(),
        {**config, "path": str(moved)},
        tmp_path / "moved-out",
    )
    first_ids = {
        json.loads(row["source_metadata_json"])["relative_path"]: row["source_record_id"]
        for row in rows
    }
    moved_ids = {
        json.loads(row["source_metadata_json"])["relative_path"]: row["source_record_id"]
        for row in moved_raw.to_pylist()
    }
    assert first_ids == moved_ids


def test_mol2_directory_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "mol2-library"
    source.mkdir()
    target = tmp_path / "target.mol2"
    target.write_text(_mol2("target"), encoding="utf-8")
    (source / "linked.mol2").symlink_to(target)

    with pytest.raises(PluginError) as caught:
        Mol2DirectorySourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(tmp_path / "out"),
        )

    assert caught.value.code == "MOL2_DIRECTORY_ENTRY_INVALID"


@pytest.mark.skipif(
    not source_module._DIRECTORY_DESCRIPTOR_TRAVERSAL,
    reason="descriptor-anchored traversal is available on Unix/WSL",
)
def test_mol2_directory_root_swap_cannot_escape_descriptor_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mol2-library"
    source.mkdir()
    (source / "inside.mol2").write_text(_mol2("inside"), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.mol2").write_text(_mol2("secret"), encoding="utf-8")
    displaced = tmp_path / "displaced-library"
    original_build = source_module._build_mol2_manifest

    def swap_then_build(*args: object, **kwargs: object) -> tuple[int, int]:
        source.rename(displaced)
        source.symlink_to(outside, target_is_directory=True)
        return original_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(source_module, "_build_mol2_manifest", swap_then_build)
    with pytest.raises(PluginError) as caught:
        Mol2DirectorySourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(tmp_path / "swap-out"),
        )

    assert caught.value.code == "MOL2_DIRECTORY_CHANGED"
    assert not (tmp_path / "swap-out" / "datasets/raw/part-00000.parquet").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX surrogateescape filename case")
def test_mol2_directory_rejects_non_utf8_filename_structurally(tmp_path: Path) -> None:
    source = tmp_path / "mol2-library"
    source.mkdir()
    invalid_name = os.fsdecode(b"invalid-\xff.mol2")
    directory_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        file_descriptor = os.open(
            invalid_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_descriptor,
        )
        os.close(file_descriptor)
    finally:
        os.close(directory_descriptor)

    with pytest.raises(PluginError) as caught:
        Mol2DirectorySourcePlugin().execute(
            StageRequest(stage_id="source", config={"path": str(source)}),
            StageContext(tmp_path / "invalid-name-out"),
        )

    assert caught.value.code == "MOL2_DIRECTORY_ENTRY_INVALID"
    assert caught.value.context["reason"] == "path_not_utf8"


def test_mol2_directory_thousand_file_order_and_python_memory_bound(tmp_path: Path) -> None:
    source = tmp_path / "mol2-library"
    source.mkdir()
    expected = []
    for index in range(1000):
        name = f"molecule-{999 - index:04d}.mol2"
        expected.append(name)
        (source / name).write_text(_mol2(name), encoding="utf-8")
    expected.sort()

    tracemalloc.start()
    response, raw, _ = _execute(
        Mol2DirectorySourcePlugin(),
        {
            "path": str(source),
            "batch_size": 100_000,
            "max_record_bytes": 16_384,
            "max_buffered_bytes": 32_768,
        },
        tmp_path / "out",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    observed = [
        json.loads(value)["relative_path"]
        for value in raw.column("source_metadata_json").to_pylist()
    ]
    assert observed == expected
    assert raw.column("source_index").to_pylist() == list(range(1000))
    assert response.metadata["max_buffered_records"] < 1000
    assert response.metadata["max_buffered_bytes"] <= 32_768
    assert peak < 64 * 1024 * 1024
