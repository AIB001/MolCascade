from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import rdBase

from molcascade.artifacts import ArtifactDatasetRef, ArtifactRef
from molcascade.contracts import (
    DECISION_V1,
    PARENT_SOURCE_MAP_V1,
    PARENT_V1,
    RAW_MOLECULE_V1,
    RAW_MOLECULE_V2,
)
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.hard_gate import RDKitHardGatePlugin
from molcascade.plugins.builtin.standardize import RDKitStandardizePlugin
from molcascade.plugins.manifest import Cardinality


def _artifact_ref(contract_id: str, digit: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact:sha256:{digit * 64}",
        kind=contract_id,
    )


def _dataset_ref(
    contract_id: str,
    digit: str,
    *file_paths: str,
) -> ArtifactDatasetRef:
    return ArtifactDatasetRef(
        artifact_id=f"artifact:sha256:{digit * 64}",
        artifact_kind="stage-output-bundle",
        port="primary",
        contract_id=contract_id,
        file_paths=file_paths,
    )


def _write_table(root: Path, name: str, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)


def _raw_row(source_id: str, index: int, smiles: str | None) -> dict[str, object]:
    return {
        "source_record_id": source_id,
        "source_kind": "test",
        "source_uri": None,
        "source_index": index,
        "generator_id": "generator-a",
        "batch_id": "batch-a",
        "source_candidate_id": source_id,
        "raw_smiles": smiles,
        "raw_molblock": None,
        "source_metadata_json": None,
    }


def _raw_v2_row(
    source_id: str,
    index: int,
    raw_format: str,
    raw_structure: str,
) -> dict[str, object]:
    return {
        "source_record_id": source_id,
        "source_kind": "test",
        "source_uri": None,
        "source_index": index,
        "generator_id": "generator-a",
        "batch_id": "batch-a",
        "source_candidate_id": source_id,
        "raw_format": raw_format,
        "raw_structure": raw_structure,
        "source_metadata_json": None,
    }


ETHANOL_MOL2 = """@<TRIPOS>MOLECULE
ethanol
 9 8 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 ETH       0.0000
      2 C2          1.5000    0.0000    0.0000 C.3       1 ETH       0.0000
      3 O1          2.1000    1.2000    0.0000 O.3       1 ETH       0.0000
      4 H1         -0.5000    0.9000    0.0000 H         1 ETH       0.0000
      5 H2         -0.5000   -0.9000    0.0000 H         1 ETH       0.0000
      6 H3          0.0000    0.0000    1.0000 H         1 ETH       0.0000
      7 H4          2.0000   -0.8000    0.0000 H         1 ETH       0.0000
      8 H5          1.5000    0.0000    1.0000 H         1 ETH       0.0000
      9 H6          3.0000    1.2000    0.0000 H         1 ETH       0.0000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 1
     3    1    4 1
     4    1    5 1
     5    1    6 1
     6    2    7 1
     7    2    8 1
     8    3    9 1
"""


def _parent_row(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity-policy:test",
        "parent_smiles": smiles,
        "registration_key": f"registration-{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def test_standardizer_plugin_globally_deduplicates_and_conserves_sources(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw-artifact"
    rows = [
        _raw_row("source-b", 0, "CCO"),
        _raw_row("source-a", 1, "OCC"),
        _raw_row("source-parse", 2, "C1"),
        _raw_row("source-tie", 3, "CC.CN"),
        _raw_row("source-empty", 4, ""),
        _raw_row("source-valence", 5, "[CH5]"),
    ]
    _write_table(input_root, "raw.parquet", RAW_MOLECULE_V1.schema, rows)
    stage_input = StageInput(
        ref=_artifact_ref(RAW_MOLECULE_V1.id, "1"),
        root=input_root,
        contract_id=RAW_MOLECULE_V1.id,
    )
    output_root = tmp_path / "standardized"
    response = RDKitStandardizePlugin().execute(
        StageRequest(
            stage_id="standardize",
            inputs={"molecules": stage_input},
            config={"batch_size": 2},
        ),
        StageContext(output_root),
    )

    parents = pq.read_table(output_root / response.outputs["primary"].file_paths[0])
    mapping = pq.read_table(output_root / response.outputs["mapping"].file_paths[0])
    decisions = pq.read_table(output_root / response.outputs["decisions"].file_paths[0])
    PARENT_V1.validate(parents)
    PARENT_SOURCE_MAP_V1.validate(mapping)
    DECISION_V1.validate(decisions)

    assert response.metadata["input_count"] == 6
    assert response.metadata["input_contract_id"] == RAW_MOLECULE_V1.id
    assert response.metadata["format_counts"] == {
        "SMILES": 5,
        "UNSPECIFIED": 1,
    }
    assert response.metadata["reject_count"] == 4
    assert response.metadata["map_count"] == 2
    assert response.metadata["unique_parent_count"] == 1
    assert response.metadata["duplicate_occurrence_count"] == 1
    assert parents.to_pylist()[0]["duplicate_count"] == 2
    assert mapping.select(["source_record_id", "relation"]).to_pylist() == [
        {"source_record_id": "source-a", "relation": "PRIMARY"},
        {"source_record_id": "source-b", "relation": "DUPLICATE"},
    ]
    decision_rows = decisions.to_pylist()
    assert all(
        "raw_format" in json.loads(row["detail"])
        for row in decision_rows
    )
    rejected_ids = {
        row["entity_id"] for row in decision_rows if row["outcome"] == "REJECT"
    }
    assert rejected_ids == {
        "source-empty",
        "source-parse",
        "source-tie",
        "source-valence",
    }
    assert {
        row["reason_code"] for row in decision_rows if row["outcome"] == "REJECT"
    } == {"EMPTY_OR_FRAGMENT_ONLY", "PARSE_ERROR", "AMBIGUOUS_FRAGMENTS", "VALENCE_ERROR"}
    assert any(
        row["entity_id"] == "source-b" and row["reason_code"] == "EXACT_DUPLICATE"
        for row in decision_rows
    )
    assert not (output_root / ".rdkit-standardize-index.sqlite3").exists()

    # A downstream port receives the same shared artifact root.  Contract
    # metadata must select parent/v1 without accidentally reading mapping or
    # decision Parquet files beside it.
    shared_parent_input = StageInput(
        ref=_artifact_ref(PARENT_V1.id, "4"),
        root=output_root,
        contract_id=PARENT_V1.id,
    )
    chained_root = tmp_path / "chained-hard-gate"
    chained = RDKitHardGatePlugin().execute(
        StageRequest(
            stage_id="hard_gate",
            inputs={"parents": shared_parent_input},
            config={"policy": {"flag_pains": False}},
        ),
        StageContext(chained_root),
    )
    assert chained.metadata["input_count"] == 1
    assert chained.metadata["pass_count"] == 1


def test_standardizer_accepts_raw_v2_mol2_and_audits_every_decision_format(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw-v2-artifact"
    rows = [
        _raw_v2_row("source-smiles", 0, "SMILES", "CCO"),
        _raw_v2_row("source-mol2", 1, "MOL2", ETHANOL_MOL2),
        _raw_v2_row("source-bad-smiles", 2, "SMILES", "C1"),
    ]
    _write_table(input_root, "raw-v2.parquet", RAW_MOLECULE_V2.schema, rows)
    stage_input = StageInput(
        ref=_artifact_ref(RAW_MOLECULE_V2.id, "c"),
        root=input_root,
        contract_id=RAW_MOLECULE_V2.id,
    )
    output_root = tmp_path / "standardized-v2"

    response = RDKitStandardizePlugin().execute(
        StageRequest(
            stage_id="standardize_v2",
            inputs={"molecules": stage_input},
            config={"batch_size": 1},
        ),
        StageContext(output_root),
    )

    parents = pq.read_table(output_root / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(output_root / response.outputs["decisions"].file_paths[0])
    assert response.metadata["input_contract_id"] == RAW_MOLECULE_V2.id
    assert response.metadata["format_counts"] == {
        "MOL2": 1,
        "SMILES": 2,
    }
    assert response.metadata["input_count"] == 3
    assert response.metadata["reject_count"] == 1
    assert response.metadata["map_count"] == 2
    assert parents.num_rows == 1

    decision_rows = decisions.to_pylist()
    by_entity_reason = {
        (row["entity_id"], row["reason_code"]): json.loads(row["detail"])
        for row in decision_rows
    }
    assert by_entity_reason[("source-smiles", "STANDARDIZED")]["raw_format"] == "SMILES"
    assert by_entity_reason[("source-mol2", "STANDARDIZED")]["raw_format"] == "MOL2"
    assert by_entity_reason[("source-bad-smiles", "PARSE_ERROR")][
        "raw_format"
    ] == "SMILES"
    duplicate_details = [
        detail
        for (entity_id, reason), detail in by_entity_reason.items()
        if entity_id in {"source-smiles", "source-mol2"} and reason == "EXACT_DUPLICATE"
    ]
    assert len(duplicate_details) == 1
    assert duplicate_details[0]["raw_format"] in {"SMILES", "MOL2"}
    assert all("raw_format" in json.loads(row["detail"]) for row in decision_rows)


def test_standardizer_rejects_zero_row_artifact_as_structured_fatal(tmp_path: Path) -> None:
    input_root = tmp_path / "empty-raw"
    _write_table(input_root, "raw.parquet", RAW_MOLECULE_V1.schema, [])
    stage_input = StageInput(
        ref=_artifact_ref(RAW_MOLECULE_V1.id, "2"),
        root=input_root,
        contract_id=RAW_MOLECULE_V1.id,
    )

    with pytest.raises(PluginError) as caught:
        RDKitStandardizePlugin().execute(
            StageRequest(stage_id="standardize", inputs={"raw": stage_input}),
            StageContext(tmp_path / "out"),
        )

    assert caught.value.code == "STANDARDIZE_EMPTY_INPUT"


def test_standardizer_deduplicates_policy_equivalents_without_parent_collisions(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "identity-equivalents"
    rows = [
        _raw_row("enhanced-a", 0, "C[C@H](O)CC |&1:1|"),
        _raw_row("enhanced-b", 1, "C[C@@H](O)CC |&1:1|"),
        _raw_row("tautomer-enol", 2, "CC=CO"),
        _raw_row("tautomer-keto", 3, "CCC=O"),
        _raw_row("metal-bonded", 4, "[Na]OC(=O)c1ccccc1"),
        _raw_row("metal-ionic", 5, "[Na+].[O-]C(=O)c1ccccc1"),
        _raw_row("broad-tautomer-a", 6, "C=C1N=C(N)C(C#N)=CN1"),
        _raw_row("broad-tautomer-b", 7, "C=C1N=CC(C#N)C(=N)N1"),
    ]
    _write_table(input_root, "raw.parquet", RAW_MOLECULE_V1.schema, rows)
    stage_input = StageInput(
        ref=_artifact_ref(RAW_MOLECULE_V1.id, "5"),
        root=input_root,
        contract_id=RAW_MOLECULE_V1.id,
    )
    output_root = tmp_path / "identity-equivalents-output"

    response = RDKitStandardizePlugin().execute(
        StageRequest(stage_id="standardize", inputs={"raw": stage_input}),
        StageContext(output_root),
    )

    parents = pq.read_table(output_root / response.outputs["primary"].file_paths[0])
    mapping = pq.read_table(output_root / response.outputs["mapping"].file_paths[0])
    decisions = pq.read_table(output_root / response.outputs["decisions"].file_paths[0])
    assert response.metadata["input_count"] == 8
    assert response.metadata["reject_count"] == 0
    assert response.metadata["map_count"] == 8
    assert response.metadata["unique_parent_count"] == parents.num_rows
    assert response.metadata["duplicate_occurrence_count"] == 8 - parents.num_rows
    assert sum(parents.column("duplicate_count").to_pylist()) == 8
    assert mapping.num_rows == (
        parents.num_rows + response.metadata["duplicate_occurrence_count"]
    )
    if rdBase.rdkitVersion == "2026.03.1":
        assert response.metadata["unique_parent_count"] == 5
        assert response.metadata["duplicate_occurrence_count"] == 3
        assert response.metadata["canonical_refinement_split_count"] == 1
        assert response.metadata["canonical_refinement_source_count"] == 2
        registration_keys = parents.column("registration_key").to_pylist()
        assert len(registration_keys) > len(set(registration_keys))
        refined_ids = {
            row["entity_id"]
            for row in decisions.to_pylist()
            if row["reason_code"] == "REGISTRATION_EQUIVALENCE_REFINED"
        }
        assert refined_ids == {"broad-tautomer-a", "broad-tautomer-b"}


def test_hard_gate_plugin_conserves_parents_and_keeps_pains_as_warning(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "parent-artifact"
    rows = [
        _parent_row("parent-ethanol", "CCO"),
        _parent_row("parent-sodium", "[Na+]"),
        _parent_row("parent-peroxide", "COO"),
        _parent_row("parent-quinone", "O=C1C=CC(=O)C=C1"),
    ]
    _write_table(input_root, "parents.parquet", PARENT_V1.schema, rows)
    stage_input = StageInput(
        ref=_artifact_ref(PARENT_V1.id, "3"),
        root=input_root,
        contract_id=PARENT_V1.id,
    )
    config = {
        "batch_size": 2,
        "policy": {
            "hard_smarts": [
                {
                    "id": "peroxide",
                    "smarts": "[O]-[O]",
                    "reason_code": "SEVERE_REACTIVITY",
                    "description": "project excludes peroxide bonds",
                }
            ]
        },
    }
    output_root = tmp_path / "hard-gated"
    response = RDKitHardGatePlugin().execute(
        StageRequest(stage_id="hard_gate", inputs={"parents": stage_input}, config=config),
        StageContext(output_root),
    )

    passed = pq.read_table(output_root / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(output_root / response.outputs["decisions"].file_paths[0])
    PARENT_V1.validate(passed)
    DECISION_V1.validate(decisions)
    assert response.metadata["input_count"] == 4
    assert response.metadata["pass_count"] == 2
    assert response.metadata["reject_count"] == 2
    assert response.metadata["hard_gate_policy_id"].startswith(
        "hard-gate-policy:sha256:"
    )
    # The invariant is that the stage records which toolkit decided, not which
    # toolkit that happens to be. Pinning the literal string turned every RDKit
    # patch release into a test failure; the identity goldens next door are the
    # place where an exact version genuinely matters, and they skip instead.
    assert response.metadata["rdkit_version"] == rdBase.rdkitVersion
    assert {row["parent_id"] for row in passed.to_pylist()} == {
        "parent-ethanol",
        "parent-quinone",
    }
    rows_by_reason = {
        (row["entity_id"], row["reason_code"]): row for row in decisions.to_pylist()
    }
    assert rows_by_reason[("parent-sodium", "DISALLOWED_ELEMENT")]["outcome"] == "REJECT"
    assert rows_by_reason[("parent-peroxide", "SEVERE_REACTIVITY")]["outcome"] == "REJECT"
    assert rows_by_reason[("parent-quinone", "PAINS_ALERT")]["outcome"] == "WARN"
    assert not any(
        row["reason_code"] == "PAINS_ALERT" and row["outcome"] == "REJECT"
        for row in decisions.to_pylist()
    )
    assert not (output_root / ".rdkit-hard-gate-index.sqlite3").exists()


def test_hard_gate_reads_only_files_claimed_by_exact_dataset_reference(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "shared-artifact"
    selected_path = "datasets/parents/selected.parquet"
    sibling_path = "datasets/parents/unclaimed-sibling.parquet"
    _write_table(
        input_root,
        selected_path,
        PARENT_V1.schema,
        [_parent_row("selected-parent", "CCO")],
    )
    _write_table(
        input_root,
        sibling_path,
        PARENT_V1.schema,
        [_parent_row("unclaimed-parent", "CCN")],
    )
    stage_input = StageInput(
        ref=_dataset_ref(PARENT_V1.id, "8", selected_path),
        root=input_root,
    )

    output_root = tmp_path / "exact-hard-gate"
    response = RDKitHardGatePlugin().execute(
        StageRequest(
            stage_id="hard_gate",
            inputs={"parents": stage_input},
            config={"policy": {"flag_pains": False}},
        ),
        StageContext(output_root),
    )

    passed = pq.read_table(output_root / response.outputs["primary"].file_paths[0])
    assert response.metadata["input_count"] == 1
    assert passed.column("parent_id").to_pylist() == ["selected-parent"]


def test_exact_dataset_reference_rejects_missing_claimed_file(tmp_path: Path) -> None:
    input_root = tmp_path / "shared-artifact"
    input_root.mkdir()
    missing_path = "datasets/parents/missing.parquet"
    stage_input = StageInput(
        ref=_dataset_ref(PARENT_V1.id, "9", missing_path),
        root=input_root,
    )

    with pytest.raises(PluginError) as caught:
        RDKitHardGatePlugin().execute(
            StageRequest(stage_id="hard_gate", inputs={"parents": stage_input}),
            StageContext(tmp_path / "out"),
        )

    assert caught.value.code == "PLUGIN_INPUT_FILE_NOT_FOUND"


def test_exact_dataset_reference_rejects_symlink_and_missing_contract_metadata(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "shared-artifact"
    outside = tmp_path / "outside.parquet"
    _write_table(
        tmp_path,
        outside.name,
        PARENT_V1.schema,
        [_parent_row("outside", "CCO")],
    )
    input_root.mkdir()
    symlink_path = input_root / "linked.parquet"
    symlink_path.symlink_to(outside)
    symlink_input = StageInput(
        ref=_dataset_ref(PARENT_V1.id, "a", symlink_path.name),
        root=input_root,
    )

    with pytest.raises(PluginError) as symlink_error:
        RDKitHardGatePlugin().execute(
            StageRequest(stage_id="hard_gate", inputs={"parents": symlink_input}),
            StageContext(tmp_path / "symlink-out"),
        )
    assert symlink_error.value.code == "PLUGIN_INPUT_FILE_INVALID"

    no_metadata_path = "no-metadata.parquet"
    _write_table(
        input_root,
        no_metadata_path,
        PARENT_V1.schema.remove_metadata(),
        [_parent_row("metadata-free", "CCO")],
    )
    metadata_input = StageInput(
        ref=_dataset_ref(PARENT_V1.id, "b", no_metadata_path),
        root=input_root,
    )

    with pytest.raises(PluginError) as metadata_error:
        RDKitHardGatePlugin().execute(
            StageRequest(stage_id="hard_gate", inputs={"parents": metadata_input}),
            StageContext(tmp_path / "metadata-out"),
        )
    assert metadata_error.value.code == "PLUGIN_INPUT_CONTRACT_METADATA_MISSING"


def test_builtin_chemistry_plugins_expose_strict_config_models_and_cardinality() -> None:
    assert RDKitStandardizePlugin.config_model.model_config["extra"] == "forbid"
    assert RDKitHardGatePlugin.config_model.model_config["extra"] == "forbid"
    assert RDKitHardGatePlugin.descriptor.cardinality is Cardinality.FILTER
    assert RDKitStandardizePlugin.descriptor.inputs == (
        RAW_MOLECULE_V1.id,
        RAW_MOLECULE_V2.id,
    )


@pytest.mark.parametrize(
    ("plugin", "config"),
    [
        (RDKitStandardizePlugin(), {"unknown": True}),
        (RDKitHardGatePlugin(), {"policy": {"reject_pains": True}}),
    ],
)
def test_chemistry_plugin_configuration_is_strict(
    plugin: object,
    config: dict[str, object],
    tmp_path: Path,
) -> None:
    request = StageRequest(stage_id="chemistry", config=config)

    with pytest.raises(PluginError) as caught:
        plugin.execute(request, StageContext(tmp_path))  # type: ignore[attr-defined]

    assert caught.value.code == "PLUGIN_CONFIG_INVALID"
