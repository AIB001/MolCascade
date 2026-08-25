from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.property_gate import RDKitPropertyRangeGatePlugin


def _row(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(tmp_path: Path) -> StageInput:
    root = tmp_path / "input"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row("ethanol", "CCO"),
                _row("benzene", "c1ccccc1"),
                _row("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
            ],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "8" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def test_property_range_gate_filters_and_emits_reasoned_decisions(tmp_path: Path) -> None:
    plugin = RDKitPropertyRangeGatePlugin()
    response = plugin.execute(
        StageRequest(
            stage_id="property_gate",
            inputs={"parents": _input(tmp_path)},
            config={
                "mw_min": 40.0,
                "mw_max": 100.0,
                "heavy_atom_min": None,
                "heavy_atom_max": None,
            },
        ),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(
        tmp_path / "out" / response.outputs["decisions"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert {row["parent_id"] for row in parents.to_pylist()} == {"ethanol", "benzene"}
    by_id = {row["entity_id"]: row for row in decisions.to_pylist()}
    assert by_id["ethanol"]["outcome"] == "PASS"
    assert by_id["caffeine"]["outcome"] == "REJECT"
    assert by_id["caffeine"]["reason_code"] == "PROPERTY_MW_ABOVE_MAX"
    assert response.metadata["input_count"] == 3
    assert response.metadata["output_count"] == 2


def test_property_range_gate_rejects_reversed_ranges(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        RDKitPropertyRangeGatePlugin().execute(
            StageRequest(
                stage_id="property_gate",
                inputs={"parents": _input(tmp_path)},
                config={"mw_min": 800.0, "mw_max": 100.0},
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "PLUGIN_CONFIG_INVALID"

