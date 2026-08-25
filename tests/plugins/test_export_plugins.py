from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, SHORTLIST_EXPORT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.exporters import (
    NativeSmilesShortlistPlugin,
    RDKitSDFShortlistPlugin,
)


def _input(tmp_path: Path) -> StageInput:
    root = tmp_path / "input"
    root.mkdir()
    rows = [
        {
            "parent_id": "ethanol",
            "identity_policy_id": "identity:test",
            "parent_smiles": "CCO",
            "registration_key": "key:ethanol",
            "stereo_key": None,
            "formula": "C2H6O",
            "duplicate_count": 1,
        },
        {
            "parent_id": "benzene",
            "identity_policy_id": "identity:test",
            "parent_smiles": "c1ccccc1",
            "registration_key": "key:benzene",
            "stereo_key": None,
            "formula": "C6H6",
            "duplicate_count": 1,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=PARENT_V1.schema), root / "parents.parquet")
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "9" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


@pytest.mark.parametrize(
    ("plugin", "expected_format"),
    [
        (NativeSmilesShortlistPlugin(), "SMILES_TSV"),
        (RDKitSDFShortlistPlugin(), "SDF_MOLBLOCK"),
    ],
)
def test_shortlist_export_plugins_preserve_parents_and_emit_typed_records(
    tmp_path: Path,
    plugin: object,
    expected_format: str,
) -> None:
    response = plugin.execute(  # type: ignore[attr-defined]
        StageRequest(stage_id="export", inputs={"parents": _input(tmp_path)}),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    records = pq.read_table(
        tmp_path / "out" / response.outputs["export_records"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    SHORTLIST_EXPORT_V1.validate(records)
    assert parents.num_rows == records.num_rows == 2
    assert {row["record_format"] for row in records.to_pylist()} == {expected_format}
    assert response.metadata["docking_included"] is False
    if expected_format == "SDF_MOLBLOCK":
        assert all("M  END" in row["structure_record"] for row in records.to_pylist())
    else:
        assert all("\t" in row["structure_record"] for row in records.to_pylist())


def test_export_plugin_rejects_nonportable_suggested_filename(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        NativeSmilesShortlistPlugin().execute(
            StageRequest(
                stage_id="export",
                inputs={"parents": _input(tmp_path)},
                config={"filename": "../outside.smi"},
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "PLUGIN_CONFIG_INVALID"

