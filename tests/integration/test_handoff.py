from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from molcascade.config import PipelineConfig
from molcascade.errors import PipelineError
from molcascade.handoff import _identifier_index, materialize_shortlist
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner

_DEFAULT_LIBRARY = "id,smiles\na,CCO\nb,c1ccccc1\n"


def _run(
    tmp_path: Path,
    exporter: str,
    *,
    library: str = _DEFAULT_LIBRARY,
    id_column: str | None = "id",
) -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(library, encoding="utf-8")
    source_config: dict[str, object] = {"path": str(source), "smiles_column": "smiles"}
    if id_column is not None:
        source_config["candidate_id_column"] = id_column
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "handoff-test",
            "stages": [
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
                    "config": {},
                },
                {
                    "id": "shortlist",
                    "slot": "export",
                    "plugin": exporter,
                    "config": {},
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    result = runner.run(config, run_id="handoff-run")
    return runner, result.run_id


def test_materialize_verified_smiles_shortlist(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, "export.native_smiles_shortlist@0.1.0")
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    lines = result.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "SMILES\tparent_id\tsource_id"
    assert len(lines) == 3
    assert result.record_format == "SMILES_TSV"
    assert result.row_count == 2
    assert len(result.sha256) == 64

    with pytest.raises(PipelineError) as caught:
        materialize_shortlist(runner, run_id, result.path)
    assert caught.value.code == "HANDOFF_OUTPUT_EXISTS"


def test_materialize_verified_sdf_shortlist(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, "export.rdkit_sdf_shortlist@0.1.0")
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.sdf")

    content = result.path.read_text(encoding="utf-8")
    assert content.count("$$$$") == 2
    assert content.count(">  <MOLCASCADE_PARENT_ID>") == 2
    with Chem.SDMolSupplier(str(result.path)) as supplier:
        molecules = [molecule for molecule in supplier if molecule is not None]
    assert len(molecules) == 2
    assert {molecule.GetProp("MOLCASCADE_PARENT_ID") for molecule in molecules} == {
        molecule.GetProp("_Name") for molecule in molecules
    }


# --------------------------------------------------------------------------
# Carrying the library's own identifier through to the handoff file
#
# A shortlist keyed only by content hash cannot be joined back to the
# spreadsheet it came from, which is the first thing anyone does with it.  The
# identifier is already in the run -- ``source_candidate_id`` on the raw records,
# linked to parents by ``parent_source_map/v1`` -- so this is a join, and these
# tests pin the places a join like that goes wrong: rows it must not widen, rows
# it must not narrow, and values that could break the file's own framing.


def test_the_library_identifier_is_carried_into_the_smiles_shortlist(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, "export.native_smiles_shortlist@0.1.0")
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    rows = [
        line.split("\t") for line in result.path.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert {row[2] for row in rows} == {"a", "b"}
    assert result.identified_count == 2
    # The hash is still there: the identifier is an addition, not a replacement,
    # because only the parent id is stable across libraries.
    assert all(row[1].startswith("parent:sha256:") for row in rows)


def test_a_library_without_identifiers_is_exported_exactly_as_before(tmp_path: Path) -> None:
    """No identifier column means no column, not a column of empty fields.

    Anything already parsing this file positionally keeps working, so adding the
    feature costs nothing to a user who never asked for it.
    """

    runner, run_id = _run(tmp_path, "export.native_smiles_shortlist@0.1.0", id_column=None)
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    lines = result.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "SMILES\tparent_id"
    assert all(line.count("\t") == 1 for line in lines[1:])
    assert result.identified_count == 0


def test_the_identifier_reaches_the_sdf_shortlist_as_a_property(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path, "export.rdkit_sdf_shortlist@0.1.0")
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.sdf")

    with Chem.SDMolSupplier(str(result.path)) as supplier:
        molecules = [molecule for molecule in supplier if molecule is not None]
    assert {molecule.GetProp("MOLCASCADE_SOURCE_ID") for molecule in molecules} == {"a", "b"}


def test_a_blank_identifier_cell_leaves_an_empty_field_not_a_short_row(tmp_path: Path) -> None:
    """A ragged TSV misaligns every parser that reads it.

    One blank cell in an otherwise-named column is ordinary, and it must not
    change how many fields the other rows have.
    """

    runner, run_id = _run(
        tmp_path,
        "export.native_smiles_shortlist@0.1.0",
        library="id,smiles\na,CCO\n,c1ccccc1\n",
    )
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    lines = result.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "SMILES\tparent_id\tsource_id"
    assert all(line.count("\t") == 2 for line in lines[1:])
    assert sorted(line.split("\t")[2] for line in lines[1:]) == ["", "a"]
    # The count reports rows that were actually named, so a partly-populated
    # column is visible before the file is joined to anything.
    assert result.identified_count == 1
    assert result.row_count == 2


def test_a_tab_inside_an_identifier_cannot_add_a_column(tmp_path: Path) -> None:
    """The identifier is whatever the user typed in a cell, including a tab."""

    runner, run_id = _run(
        tmp_path,
        "export.native_smiles_shortlist@0.1.0",
        library='id,smiles\n"lead\tseries 4",CCO\nb,c1ccccc1\n',
    )
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    lines = result.path.read_text(encoding="utf-8").splitlines()
    assert all(line.count("\t") == 2 for line in lines[1:])
    assert "lead series 4" in result.path.read_text(encoding="utf-8")


def test_two_library_rows_for_one_molecule_yield_one_named_row(tmp_path: Path) -> None:
    """Deduplication happens before export, so a parent has one name here.

    Which of the two it is, is deterministic but decided by record digest rather
    than by position in the file.  The occurrence that lost is recorded in the
    run's decision log as ``EXACT_DUPLICATE``; a handoff file is a handoff, not
    an audit trail.
    """

    runner, run_id = _run(
        tmp_path,
        "export.native_smiles_shortlist@0.1.0",
        library="id,smiles\nfirst,CCO\nsecond,CCO\nother,c1ccccc1\n",
    )
    result = materialize_shortlist(runner, run_id, tmp_path / "shortlist.smi")

    rows = [
        line.split("\t") for line in result.path.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert result.row_count == 2
    names = {row[2] for row in rows}
    assert "other" in names
    assert len(names & {"first", "second"}) == 1


def test_a_run_that_records_no_source_mapping_exports_without_identifiers(
    tmp_path: Path,
) -> None:
    """A hand-assembled pipeline need not contain a standardizer at all.

    Missing provenance is not a failure to report; it is a file written the way
    it was written before identifiers existed.
    """

    runner, run_id = _run(tmp_path, "export.native_smiles_shortlist@0.1.0")
    state = runner.load_run(run_id)

    assert _identifier_index(runner, state.model_copy(update={"stages": ()}), {"any"}) == {}
    # An empty shortlist short-circuits before any dataset is opened.
    assert _identifier_index(runner, state, set()) == {}
    # A parent that was never registered matches no mapping row.
    assert _identifier_index(runner, state, {"parent:sha256:" + "0" * 64}) == {}
