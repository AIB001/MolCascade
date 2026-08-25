from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import FINGERPRINT_V1, PARENT_V1, PROPERTY_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.features import (
    OpenBabelPropertyPlugin,
    RDKitFingerprintPlugin,
    RDKitPropertyPlugin,
)


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity-policy:test",
        "parent_smiles": smiles,
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(tmp_path: Path) -> StageInput:
    root = tmp_path / "input"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [_parent("ethanol", "CCO"), _parent("benzene", "c1ccccc1")],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "1" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(plugin: object, stage_input: StageInput, output: Path, config: dict[str, object]):
    return plugin.execute(  # type: ignore[attr-defined]
        StageRequest(stage_id="features", inputs={"parents": stage_input}, config=config),
        StageContext(output),
    )


def test_rdkit_property_plugin_preserves_primary_and_emits_typed_properties(
    tmp_path: Path,
) -> None:
    response = _run(RDKitPropertyPlugin(), _input(tmp_path), tmp_path / "out", {})
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    properties = pq.read_table(
        tmp_path / "out" / response.outputs["properties"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    PROPERTY_V1.validate(properties)
    assert parents.num_rows == properties.num_rows == 2
    rows = {row["parent_id"]: row for row in properties.to_pylist()}
    assert rows["ethanol"]["mw"] == pytest.approx(46.069, abs=0.01)
    assert rows["benzene"]["ring_count"] == 1
    assert 1.0 <= rows["ethanol"]["sa_score"] <= 10.0
    assert response.metadata["input_count"] == response.metadata["output_count"] == 2


def test_openbabel_is_explicit_and_cross_checks_common_property_semantics(
    tmp_path: Path,
) -> None:
    stage_input = _input(tmp_path)
    with pytest.raises(PluginError) as caught:
        _run(OpenBabelPropertyPlugin(), stage_input, tmp_path / "blocked", {})
    assert caught.value.code == "BACKEND_LICENSE_BLOCKED"

    response = _run(
        OpenBabelPropertyPlugin(),
        stage_input,
        tmp_path / "allowed",
        {"allow_copyleft_backend": True},
    )
    properties = pq.read_table(
        tmp_path / "allowed" / response.outputs["properties"].file_paths[0]
    )
    PROPERTY_V1.validate(properties)
    rows = {row["parent_id"]: row for row in properties.to_pylist()}
    assert rows["ethanol"]["mw"] == pytest.approx(46.068, abs=0.02)
    assert rows["benzene"]["ring_count"] == 1
    assert "PROPERTY_NOT_SUPPORTED:sa_score" in rows["ethanol"]["warning_codes_json"]
    assert response.metadata["license"] == "GPL-2.0-only"


@pytest.mark.parametrize("kind", ["morgan", "atom_pair", "rdkit_path"])
def test_rdkit_fingerprint_backends_are_packed_and_deterministic(
    tmp_path: Path,
    kind: str,
) -> None:
    stage_input = _input(tmp_path)
    first = _run(
        RDKitFingerprintPlugin(),
        stage_input,
        tmp_path / f"first-{kind}",
        {"kind": kind, "bit_length": 256},
    )
    second = _run(
        RDKitFingerprintPlugin(),
        stage_input,
        tmp_path / f"second-{kind}",
        {"kind": kind, "bit_length": 256},
    )
    first_table = pq.read_table(
        tmp_path / f"first-{kind}" / first.outputs["fingerprints"].file_paths[0]
    )
    second_table = pq.read_table(
        tmp_path / f"second-{kind}" / second.outputs["fingerprints"].file_paths[0]
    )
    FINGERPRINT_V1.validate(first_table)
    assert first_table.equals(second_table)
    for row in first_table.to_pylist():
        assert row["bit_length"] == 256
        assert len(row["packed_bits"]) == 32
        assert row["popcount"] > 0


def test_maccs_requires_its_fixed_declared_length(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _run(
            RDKitFingerprintPlugin(),
            _input(tmp_path),
            tmp_path / "out",
            {"kind": "maccs"},
        )
    assert caught.value.code == "PLUGIN_CONFIG_INVALID"

