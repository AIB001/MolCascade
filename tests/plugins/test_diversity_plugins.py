from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import (
    CLUSTER_ASSIGNMENT_V1,
    PARENT_V1,
    SCAFFOLD_ASSIGNMENT_V1,
    SELECTION_DECISION_V1,
)
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.diversity import (
    NativeHashBudgetSelectorPlugin,
    NativeScaffoldRoundRobinSelectorPlugin,
    NativeStreamingLeaderPlugin,
    RDKitMurckoScaffoldPlugin,
    RDKitScaffoldGroupClusterPlugin,
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
    rows = [
        _parent("benzene", "c1ccccc1"),
        _parent("toluene", "Cc1ccccc1"),
        _parent("cyclohexane", "C1CCCCC1"),
        _parent("ethanol", "CCO"),
        _parent("propanol", "CCCO"),
        _parent("pyridine", "n1ccccc1"),
    ]
    root = tmp_path / "input"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=PARENT_V1.schema), root / "parents.parquet")
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "7" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(plugin: object, stage_input: StageInput, output: Path, config: dict[str, object]):
    return plugin.execute(  # type: ignore[attr-defined]
        StageRequest(stage_id="diversity", inputs={"parents": stage_input}, config=config),
        StageContext(output),
    )


def test_murcko_assigns_ring_and_explicit_acyclic_scaffolds(tmp_path: Path) -> None:
    response = _run(RDKitMurckoScaffoldPlugin(), _input(tmp_path), tmp_path / "out", {})
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    scaffolds = pq.read_table(tmp_path / "out" / response.outputs["scaffolds"].file_paths[0])
    PARENT_V1.validate(parents)
    SCAFFOLD_ASSIGNMENT_V1.validate(scaffolds)
    by_id = {row["parent_id"]: row for row in scaffolds.to_pylist()}
    assert by_id["ethanol"]["acyclic"] is True
    assert by_id["ethanol"]["exact_scaffold_id"] == "ACYCLIC"
    assert by_id["benzene"]["acyclic"] is False
    assert by_id["benzene"]["exact_scaffold_id"] == by_id["toluene"]["exact_scaffold_id"]


def test_streaming_leader_is_total_and_has_no_dense_matrix_output(tmp_path: Path) -> None:
    response = _run(
        NativeStreamingLeaderPlugin(),
        _input(tmp_path),
        tmp_path / "out",
        {"similarity_threshold": 0.4, "fingerprint_bits": 256},
    )
    clusters = pq.read_table(tmp_path / "out" / response.outputs["clusters"].file_paths[0])
    CLUSTER_ASSIGNMENT_V1.validate(clusters)
    assert clusters.num_rows == 6
    assert 1 <= response.metadata["cluster_count"] <= 6
    for row in clusters.to_pylist():
        assert 0.0 <= row["similarity_to_representative"] <= 1.0
        if row["is_representative"]:
            assert row["parent_id"] == row["representative_parent_id"]
            assert row["similarity_to_representative"] == 1.0


def test_scaffold_group_cluster_groups_acyclic_and_shared_murcko(tmp_path: Path) -> None:
    response = _run(
        RDKitScaffoldGroupClusterPlugin(),
        _input(tmp_path),
        tmp_path / "out",
        {"fingerprint_bits": 256},
    )
    clusters = pq.read_table(tmp_path / "out" / response.outputs["clusters"].file_paths[0])
    rows = {row["parent_id"]: row for row in clusters.to_pylist()}
    assert rows["benzene"]["cluster_id"] == rows["toluene"]["cluster_id"]
    assert rows["ethanol"]["cluster_id"] == rows["propanol"]["cluster_id"]
    assert response.metadata["cluster_count"] == 4


def test_hash_budget_is_seeded_exact_and_audits_every_parent(tmp_path: Path) -> None:
    stage_input = _input(tmp_path)
    first = _run(
        NativeHashBudgetSelectorPlugin(),
        stage_input,
        tmp_path / "first",
        {"target_count": 3, "seed": 42},
    )
    second = _run(
        NativeHashBudgetSelectorPlugin(),
        stage_input,
        tmp_path / "second",
        {"target_count": 3, "seed": 42},
    )
    selected = pq.read_table(tmp_path / "first" / first.outputs["primary"].file_paths[0])
    decisions = pq.read_table(tmp_path / "first" / first.outputs["selection"].file_paths[0])
    selected_again = pq.read_table(
        tmp_path / "second" / second.outputs["primary"].file_paths[0]
    )
    PARENT_V1.validate(selected)
    SELECTION_DECISION_V1.validate(decisions)
    assert selected.num_rows == 3
    assert decisions.num_rows == 6
    assert selected.equals(selected_again)
    assert sum(row["selected"] for row in decisions.to_pylist()) == 3


def test_scaffold_round_robin_prioritizes_breadth_and_caps_groups(tmp_path: Path) -> None:
    response = _run(
        NativeScaffoldRoundRobinSelectorPlugin(),
        _input(tmp_path),
        tmp_path / "out",
        {"target_count": 5, "max_per_scaffold": 1, "seed": 9},
    )
    selected = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(tmp_path / "out" / response.outputs["selection"].file_paths[0])
    # Four exact Murcko groups are present; max_per_scaffold=1 is authoritative.
    assert selected.num_rows == 4
    chosen = [row for row in decisions.to_pylist() if row["selected"]]
    assert len(chosen) == 4
    assert {row["basket"] for row in chosen} == {"SCAFFOLD_DIVERSITY"}

