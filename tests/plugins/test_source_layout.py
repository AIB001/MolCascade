"""Reading a library that was laid out for a human reader, not for a parser.

Every test here comes from the shape of one real file: a curated binder
workbook whose first sheet is a README, whose data sheet opens with a merged
section banner, and whose SMILES column is spelled ``SMILES``.  MolCascade rejected it three separate
ways before reaching a molecule, and each rejection named something other than
the thing that was wrong -- a "CSV header" for a spreadsheet, a missing column
without saying what columns were present, a header row without saying which
sheet it came from.

The file itself is not vendored -- it is a user's data, not a fixture -- and
neither is its target.  What is reconstructed here is the layout, with filler
structures, because the layout is the thing under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.contracts import DECISION_V1, RAW_MOLECULE_V1, RAW_MOLECULE_V2
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageRequest
from molcascade.plugins.builtin.sources import (
    DelimitedSmilesSourceConfig,
    DelimitedSmilesSourcePlugin,
    XlsxSourceConfig,
    XlsxSourcePlugin,
    _xlsx_column_letter,
)

_HEADER = ["No", "Compound_Name", "Chemotype", "SMILES", "ChEMBL_ID"]
_ROWS = [
    [1, "Compound-A", "Anilide", "CCCCC(=O)Nc1ccccc1", "CHEMBL000001"],
    [2, "Compound-B", "Biaryl", "CSc1ccc(-c2ccccc2)cc1", "CHEMBL000002"],
]


def _banner_workbook(path: Path, *, sheets: tuple[str, ...] = ("2_Master_Compounds",)) -> None:
    """A sheet that opens with a merged section banner above the real header.

    openpyxl reports a merged span as one populated cell followed by ``None``s,
    which is exactly the shape that used to make the whole workbook unreadable.
    """

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name in sheets:
        sheet = workbook.create_sheet(name)
        sheet.append(["Identity", None, None, "Database IDs", None])
        sheet.append(list(_HEADER))
        for row in _ROWS:
            sheet.append(list(row))
    workbook.save(path)
    workbook.close()


def _run_xlsx(config: dict[str, object], output: Path) -> tuple[object, pa.Table, pa.Table]:
    response = XlsxSourcePlugin().execute(
        StageRequest(stage_id="source", config=config),
        StageContext(output),
    )
    raw = pq.read_table(output / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(output / response.outputs["decisions"].file_paths[0])
    RAW_MOLECULE_V2.validate(raw)
    DECISION_V1.validate(decisions)
    return response, raw, decisions


def _run_delimited(config: dict[str, object], output: Path) -> tuple[object, pa.Table]:
    response = DelimitedSmilesSourcePlugin().execute(
        StageRequest(stage_id="source", config=config),
        StageContext(output),
    )
    raw = pq.read_table(output / response.outputs["primary"].file_paths[0])
    RAW_MOLECULE_V1.validate(raw)
    return response, raw


def _fails(config: dict[str, object], output: Path, plugin: object) -> PluginError:
    with pytest.raises(PluginError) as caught:
        plugin.execute(  # type: ignore[attr-defined]
            StageRequest(stage_id="source", config=config),
            StageContext(output),
        )
    return caught.value


# --------------------------------------------------------------------------
# Skipping a banner


def test_a_banner_row_above_the_header_is_skipped_on_request(tmp_path: Path) -> None:
    source = tmp_path / "binders.xlsx"
    _banner_workbook(source)

    _, raw, _ = _run_xlsx(
        {
            "path": str(source),
            "sheet_name": "2_Master_Compounds",
            "skip_rows": 1,
            "smiles_column": "SMILES",
            "candidate_id_column": "Compound_Name",
        },
        tmp_path / "out",
    )

    rows = raw.to_pylist()
    assert [row["raw_structure"] for row in rows] == [item[3] for item in _ROWS]
    assert [row["source_candidate_id"] for row in rows] == ["Compound-A", "Compound-B"]
    # ``source_index`` counts molecules from zero; ``excel_row`` still points at
    # the physical row, so a rejected record can be found in the spreadsheet.
    assert [row["source_index"] for row in rows] == [0, 1]
    assert [
        json.loads(row["source_metadata_json"])["excel_row"] for row in rows
    ] == [3, 4]


def test_the_banner_row_is_named_along_with_the_flag_that_skips_it(tmp_path: Path) -> None:
    source = tmp_path / "binders.xlsx"
    _banner_workbook(source)

    error = _fails(
        {
            "path": str(source),
            "sheet_name": "2_Master_Compounds",
            "smiles_column": "SMILES",
        },
        tmp_path / "out",
        XlsxSourcePlugin(),
    )

    assert error.code == "SOURCE_HEADER_INVALID"
    # The user has to be told three things to fix this in one attempt: which
    # cell, why it is not a header, and the flag that moves past it.
    assert "cell B1" in error.message
    assert "NoneType" in error.message
    assert "skip_rows=1" in error.message


def test_skipping_past_the_end_of_a_file_says_how_long_the_file_is(tmp_path: Path) -> None:
    source = tmp_path / "molecules.csv"
    source.write_text("smiles,id\nCCO,a\n", encoding="utf-8")

    error = _fails(
        {"path": str(source), "skip_rows": 9, "smiles_column": "smiles"},
        tmp_path / "out",
        DelimitedSmilesSourcePlugin(),
    )

    assert error.code == "SOURCE_HEADER_INVALID"
    assert "skip_rows is 9" in error.message
    assert "only 2 line(s)" in error.message


def test_a_delimited_banner_is_skipped_without_being_parsed(tmp_path: Path) -> None:
    """The line being discarded need not be valid CSV.

    A title row is exactly where an unbalanced quote or a stray comma lives.
    Refusing to skip a line because it does not parse would defeat the only
    reason anyone asks to skip it.
    """

    source = tmp_path / "molecules.csv"
    source.write_text(
        'Known binders, exported 2026-08-25 "draft\nsmiles,id\nCCO,a\nCCN,b\n',
        encoding="utf-8",
    )

    _, raw = _run_delimited(
        {
            "path": str(source),
            "skip_rows": 1,
            "smiles_column": "smiles",
            "candidate_id_column": "id",
        },
        tmp_path / "out",
    )

    rows = raw.to_pylist()
    assert [row["raw_smiles"] for row in rows] == ["CCO", "CCN"]
    assert [row["source_candidate_id"] for row in rows] == ["a", "b"]
    # Molecules are numbered from zero even though they begin on physical line 3.
    assert [row["source_index"] for row in rows] == [0, 1]
    assert [
        json.loads(row["source_metadata_json"])["data_record_index"] for row in rows
    ] == [0, 1]


def test_skipping_rows_changes_the_recorded_extraction(tmp_path: Path) -> None:
    """Two reads that keep different rows must not hash as the same extraction.

    The extraction hash is what a reader of the provenance uses to decide
    whether two runs saw the same molecules.  If ``skip_rows`` were left out of
    it, discarding a banner would be invisible in the audit trail.
    """

    source = tmp_path / "molecules.csv"
    source.write_text("banner\nsmiles,id\nCCO,a\n", encoding="utf-8")
    base = {"path": str(source), "smiles_column": "smiles", "candidate_id_column": "id"}

    skipped, _ = _run_delimited({**base, "skip_rows": 1}, tmp_path / "skipped")
    with pytest.raises(PluginError):
        # Without the skip the banner is the header, so 'smiles' is absent.
        _run_delimited(base, tmp_path / "kept")

    source.write_text("smiles,id\nCCO,a\n", encoding="utf-8")
    unskipped, _ = _run_delimited(base, tmp_path / "plain")
    assert (
        skipped.metadata["extraction_config_sha256"]  # type: ignore[attr-defined]
        != unskipped.metadata["extraction_config_sha256"]  # type: ignore[attr-defined]
    )


# --------------------------------------------------------------------------
# Matching a column name


@pytest.mark.parametrize("configured", ["SMILES", "smiles", "Smiles"])
def test_a_column_name_matches_regardless_of_case(tmp_path: Path, configured: str) -> None:
    """``SMILES`` and ``smiles`` are the same column to everyone but a parser.

    ChEMBL exports one spelling, a notebook exports the other, and the default
    configuration cannot be right for both.  Nobody should have to discover
    which by trial.
    """

    source = tmp_path / "binders.xlsx"
    _banner_workbook(source)

    _, raw, _ = _run_xlsx(
        {
            "path": str(source),
            "sheet_name": "2_Master_Compounds",
            "skip_rows": 1,
            "smiles_column": configured,
        },
        tmp_path / configured.lower(),
    )

    assert [row["raw_structure"] for row in raw.to_pylist()] == [item[3] for item in _ROWS]


def test_two_spellings_of_one_name_are_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Case-insensitive matching stops where the file itself is ambiguous.

    A sheet holding both ``SMILES`` and ``smiles`` has two genuinely different
    columns, and picking the leftmost would silently screen the wrong molecules.
    """

    source = tmp_path / "ambiguous.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["SMILES", "smiles", "id"])
    sheet.append(["CCO", "CCN", "a"])
    workbook.save(source)
    workbook.close()

    error = _fails(
        {"path": str(source), "smiles_column": "Smiles"},
        tmp_path / "out",
        XlsxSourcePlugin(),
    )

    assert error.code == "SOURCE_HEADER_INVALID"
    assert "differ only in case" in error.message
    assert "'SMILES'" in error.message and "'smiles'" in error.message


def test_an_exact_match_wins_over_a_case_insensitive_one(tmp_path: Path) -> None:
    """The ambiguity check must not fire when the user named a column exactly."""

    source = tmp_path / "ambiguous.csv"
    source.write_text("SMILES,smiles\nCCO,CCN\n", encoding="utf-8")

    _, raw = _run_delimited(
        {"path": str(source), "smiles_column": "smiles"},
        tmp_path / "out",
    )

    assert [row["raw_smiles"] for row in raw.to_pylist()] == ["CCN"]


def test_a_missing_column_quotes_the_header_it_actually_read(tmp_path: Path) -> None:
    source = tmp_path / "molecules.csv"
    source.write_text("structure,name\nCCO,a\n", encoding="utf-8")

    error = _fails(
        {"path": str(source), "smiles_column": "smiles"},
        tmp_path / "out",
        DelimitedSmilesSourcePlugin(),
    )

    assert "'smiles'" in error.message
    # Naming the header is what turns "column missing" into a fixable message:
    # it is usually either a spelling or the wrong row entirely.
    assert "'structure', 'name'" in error.message


def test_a_wide_header_is_summarised_rather_than_dumped(tmp_path: Path) -> None:
    names = [f"column_{index}" for index in range(40)]
    source = tmp_path / "wide.csv"
    source.write_text(",".join(names) + "\n" + ",".join("x" * 40) + "\n", encoding="utf-8")

    error = _fails(
        {"path": str(source), "smiles_column": "smiles"},
        tmp_path / "out",
        DelimitedSmilesSourcePlugin(),
    )

    assert "column_0" in error.message
    assert "(40 columns)" in error.message
    # A 40-column dump is not a diagnostic; the count plus a prefix is.
    assert "column_39" not in error.message


# --------------------------------------------------------------------------
# Saying which sheet was read


def test_a_header_failure_names_the_sheet_and_offers_the_others(tmp_path: Path) -> None:
    """The default sheet is the first visible one, which is often a README.

    When that happens every configured column is genuinely absent, and the
    accurate message sends the user hunting for a column that is sitting one tab
    to the right.
    """

    source = tmp_path / "curated.xlsx"
    workbook = openpyxl.Workbook()
    readme = workbook.active
    readme.title = "1_README"
    readme.append(["Project library - known small-molecule binders"])
    data = workbook.create_sheet("2_Master_Compounds")
    data.append(list(_HEADER))
    data.append(list(_ROWS[0]))
    workbook.save(source)
    workbook.close()

    error = _fails(
        {"path": str(source), "smiles_column": "SMILES"},
        tmp_path / "out",
        XlsxSourcePlugin(),
    )

    assert "worksheet '1_README'" in error.message
    assert error.hint is not None
    assert "'2_Master_Compounds'" in error.hint
    assert "--sheet" in error.hint
    assert error.context["sheet_selection"] == "FIRST_VISIBLE"
    assert error.context["worksheets"] == ["1_README", "2_Master_Compounds"]


def test_an_explicitly_chosen_sheet_is_not_second_guessed(tmp_path: Path) -> None:
    """No alternatives are suggested when the user already named a sheet."""

    source = tmp_path / "curated.xlsx"
    _banner_workbook(source, sheets=("2_Master_Compounds", "3_Key_Binders"))

    error = _fails(
        {
            "path": str(source),
            "sheet_name": "3_Key_Binders",
            "skip_rows": 1,
            "smiles_column": "absent",
        },
        tmp_path / "out",
        XlsxSourcePlugin(),
    )

    assert "worksheet '3_Key_Binders'" in error.message
    assert "header row 2" in error.message
    assert error.hint is None


def test_a_spreadsheet_problem_is_never_reported_as_a_csv_problem(tmp_path: Path) -> None:
    """One resolver serves both readers; only one of them reads CSV.

    Telling someone their workbook has a bad "CSV header" sends them looking for
    a format error that does not exist -- the file parsed perfectly and only the
    column name was wrong.
    """

    source = tmp_path / "binders.xlsx"
    _banner_workbook(source)

    error = _fails(
        {
            "path": str(source),
            "sheet_name": "2_Master_Compounds",
            "skip_rows": 1,
            "smiles_column": "absent",
        },
        tmp_path / "out",
        XlsxSourcePlugin(),
    )

    assert "CSV" not in error.message
    assert "XLSX" in error.message


# --------------------------------------------------------------------------
# Small pieces


@pytest.mark.parametrize(
    ("index", "letter"),
    [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (701, "ZZ"), (702, "AAA")],
)
def test_column_indexes_are_reported_as_spreadsheet_letters(index: int, letter: str) -> None:
    """A user looking at Excel sees 'B1', not 'column 1'."""

    assert _xlsx_column_letter(index) == letter


def test_skip_rows_defaults_to_reading_the_file_as_it_is() -> None:
    """The new field must not change what any existing configuration does."""

    assert DelimitedSmilesSourceConfig().skip_rows == 0
    assert XlsxSourceConfig().skip_rows == 0


@pytest.mark.parametrize("value", [-1, -100])
def test_a_negative_skip_is_rejected_by_the_model(value: int) -> None:
    for model in (DelimitedSmilesSourceConfig, XlsxSourceConfig):
        with pytest.raises(ValueError):
            model(skip_rows=value)
