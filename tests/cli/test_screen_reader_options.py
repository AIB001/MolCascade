"""``molcascade screen`` reader flags: the library's layout, on the command line.

The builder writes a reusable screening *policy*; the library changes every run,
and so does its layout -- a different sheet, a differently spelled SMILES column,
a banner row above the header.  Before these flags the only way to read such a
file was to hand-edit the exported JSON, which defeats the point of a builder.

These tests hold two things: that a typed flag reaches the reader, and -- the
part that is easy to get wrong -- that an *untyped* flag leaves the stored
configuration alone rather than overwriting it with argparse's ``None``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from molcascade.cascade import LibraryConfig, ScreeningConfig, load_screening_config
from molcascade.cascade.library import XLSX_SOURCE, reader_settings, resolve_library
from molcascade.cli import _molecule_override, _prepare_screen, _reader_overrides, main
from molcascade.config import PipelineConfig
from molcascade.errors import PipelineError
from molcascade.plugins import create_builtin_registry
from molcascade.ui.cascade_builder import build_cascade_builder_payload


def _namespace(**overrides: object) -> argparse.Namespace:
    typed: dict[str, object] = {
        "smiles_column": None,
        "id_column": None,
        "sheet": None,
        "skip_rows": None,
    }
    typed.update(overrides)
    return argparse.Namespace(**typed)


def _cascade_config(tmp_path: Path, **library: object) -> Path:
    payload = build_cascade_builder_payload()["default_cascade"]
    if library:
        payload = {**payload, "library": {**payload.get("library", {}), **library}}
    destination = tmp_path / "cascade.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def _source_settings(config: Path, library: Path, **overrides: object) -> dict[str, object]:
    screening: ScreeningConfig = load_screening_config(config)
    pipeline, _ = _prepare_screen(
        screening,
        library,
        create_builtin_registry(),
        molecule_format="auto",
        reader=_reader_overrides(_namespace(**overrides)),
    )
    return dict(pipeline.stages[0].config)


def test_only_the_flags_that_were_typed_become_overrides() -> None:
    assert _reader_overrides(_namespace()) == {}
    assert _reader_overrides(_namespace(sheet="Sheet2")) == {"sheet_name": "Sheet2"}
    # Zero is a meaningful value and must survive the "was it typed" filter.
    assert _reader_overrides(_namespace(skip_rows=0)) == {"skip_rows": 0}


def test_an_untyped_flag_does_not_erase_a_configured_column(tmp_path: Path) -> None:
    """The failure this prevents is silent: a builder-chosen column disappearing.

    ``--smiles-column`` defaults to ``None`` in argparse, and passing that
    through would have replaced whatever the exported configuration said with
    the reader's own default.
    """

    config = _cascade_config(tmp_path, smiles_column="structure", sheet_name="Compounds")
    library = tmp_path / "library.xlsx"

    settings = _source_settings(config, library)

    assert settings["smiles_column"] == "structure"
    assert settings["sheet_name"] == "Compounds"


def test_a_typed_flag_wins_over_the_configured_value(tmp_path: Path) -> None:
    config = _cascade_config(tmp_path, smiles_column="structure", sheet_name="Compounds")
    library = tmp_path / "library.xlsx"

    settings = _source_settings(
        config,
        library,
        smiles_column="SMILES",
        sheet="2_Master_Compounds",
        skip_rows=1,
        id_column="Compound_Name",
    )

    assert settings["smiles_column"] == "SMILES"
    assert settings["sheet_name"] == "2_Master_Compounds"
    assert settings["skip_rows"] == 1
    assert settings["candidate_id_column"] == "Compound_Name"


def test_reader_flags_reach_a_raw_pipeline_configuration_too(tmp_path: Path) -> None:
    """A hand-written pipeline takes a different code path to the same reader."""

    document = {
        "schema_version": 1,
        "name": "reader-test",
        "stages": [
            {
                "id": "source",
                "slot": "source",
                "plugin": "source.delimited_smiles@0.1.0",
                "config": {"path": "configured.csv", "smiles_column": "smiles"},
                "enabled": True,
            }
        ],
    }

    overridden = _molecule_override(
        PipelineConfig.model_validate(document),
        tmp_path / "library.xlsx",
        create_builtin_registry(),
        reader={"sheet_name": "Compounds", "skip_rows": 2, "smiles_column": "SMILES"},
    )

    source = overridden.stages[0]
    assert source.plugin == XLSX_SOURCE
    assert source.config["sheet_name"] == "Compounds"
    assert source.config["skip_rows"] == 2
    assert source.config["smiles_column"] == "SMILES"


def test_an_option_the_chosen_reader_has_no_field_for_is_dropped(tmp_path: Path) -> None:
    """``--sheet`` on a CSV is meaningless, not an error.

    Refusing the whole run over an inapplicable flag would be worse than
    ignoring it: the flag is often left over from the previous library.
    """

    document = {
        "schema_version": 1,
        "name": "reader-test",
        "stages": [
            {
                "id": "source",
                "slot": "source",
                "plugin": "source.delimited_smiles@0.1.0",
                "config": {"path": "configured.csv", "smiles_column": "smiles"},
                "enabled": True,
            }
        ],
    }

    overridden = _molecule_override(
        PipelineConfig.model_validate(document),
        tmp_path / "library.csv",
        create_builtin_registry(),
        reader={"sheet_name": "Compounds", "skip_rows": 1},
    )

    source = overridden.stages[0]
    assert "sheet_name" not in source.config
    assert source.config["skip_rows"] == 1


def test_the_same_option_is_named_whatever_each_reader_calls_it() -> None:
    """One vocabulary for the user, several for the readers.

    An SD record has no columns, so its identifier is a *property*.  Keeping the
    translation in one function is what stops the command line and a stored
    configuration from disagreeing about what ``--id-column`` means.
    """

    library = LibraryConfig(id_column="Compound_Name", smiles_column="SMILES", skip_rows=1)

    delimited = reader_settings(library, "source.delimited_smiles@0.1.0")
    spreadsheet = reader_settings(library, XLSX_SOURCE)
    sdf = reader_settings(library, "source.sdf@0.1.0")
    parquet = reader_settings(library, "source.raw_molecule_parquet@0.1.0")

    assert delimited["candidate_id_column"] == "Compound_Name"
    assert spreadsheet["candidate_id_column"] == "Compound_Name"
    assert sdf == {"candidate_id_property": "Compound_Name"}
    # Neither a column nor a row layout means anything to a Parquet dataset.
    assert parquet == {}
    assert "skip_rows" in delimited and "skip_rows" in spreadsheet


def test_skip_rows_reaches_only_the_readers_that_have_rows_to_skip(tmp_path: Path) -> None:
    library = LibraryConfig(skip_rows=3)

    spreadsheet = resolve_library(library, override_path=tmp_path / "library.xlsx")
    delimited = resolve_library(library, override_path=tmp_path / "library.csv")
    records = resolve_library(library, override_path=tmp_path / "library.sdf")

    assert spreadsheet.settings["skip_rows"] == 3
    assert delimited.settings["skip_rows"] == 3
    assert "skip_rows" not in records.settings


def test_a_negative_skip_is_rejected_before_anything_is_read() -> None:
    with pytest.raises(PipelineError) as caught:
        _reader_overrides(_namespace(skip_rows=-1))

    assert caught.value.code == "CLI_SKIP_ROWS_INVALID"


def test_the_screen_command_accepts_the_reader_flags(tmp_path: Path) -> None:
    """Wiring check: the flags exist on the parser and reach the plan.

    ``--dry-run`` stops before the file is opened, so this asserts the argument
    surface without needing a library on disk.
    """

    config = _cascade_config(tmp_path)
    exit_code = main(
        [
            "screen",
            "--config",
            str(config),
            "--library",
            str(tmp_path / "library.xlsx"),
            "--sheet",
            "2_Master_Compounds",
            "--skip-rows",
            "1",
            "--smiles-column",
            "SMILES",
            "--id-column",
            "Compound_Name",
            "--workspace",
            str(tmp_path / "ws"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
