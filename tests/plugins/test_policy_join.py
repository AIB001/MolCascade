from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.policy import NativeDecisionJoinPlugin


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _dataset(
    tmp_path: Path,
    name: str,
    contract: object,
    rows: list[dict[str, object]],
) -> StageInput:
    root = tmp_path / name
    root.mkdir()
    path = root / "part.parquet"
    schema = contract.schema  # type: ignore[attr-defined]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    contract_id = contract.id  # type: ignore[attr-defined]
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + ("a" if name == "parents" else "b") * 64,
            kind=contract_id,
        ),
        root=root,
        contract_id=contract_id,
    )


def _decision(parent_id: str, stage: str, outcome: str) -> dict[str, object]:
    return {
        "entity_id": parent_id,
        "entity_kind": "PARENT",
        "stage_id": stage,
        "outcome": outcome,
        "reason_code": f"{stage.upper()}_{outcome}",
        "rule_id": f"rule:{stage}",
        "detail": "test evidence",
    }


def _inputs(tmp_path: Path) -> dict[str, StageInput]:
    parents = [_parent("a", "CCO"), _parent("b", "c1ccccc1"), _parent("c", "CCN")]
    first = [
        _decision("a", "ro5", "PASS"),
        _decision("b", "ro5", "REJECT"),
        _decision("c", "ro5", "PASS"),
    ]
    second = [
        _decision("a", "alerts", "PASS"),
        _decision("b", "alerts", "PASS"),
        _decision("c", "alerts", "REJECT"),
    ]
    return {
        "parents": _dataset(tmp_path, "parents", PARENT_V1, parents),
        "ro5": _dataset(tmp_path, "ro5", DECISION_V1, first),
        "alerts": _dataset(tmp_path, "alerts", DECISION_V1, second),
    }


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"mode": "all_required"}, {"a"}),
        ({"mode": "any_required"}, {"a", "b", "c"}),
        ({"mode": "min_pass_count", "min_pass_count": 2}, {"a"}),
    ],
)
def test_join_executes_parallel_branch_policy(
    tmp_path: Path,
    config: dict[str, object],
    expected: set[str],
) -> None:
    response = NativeDecisionJoinPlugin().execute(
        StageRequest(stage_id="policy_join", inputs=_inputs(tmp_path), config=config),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(
        tmp_path / "out" / response.outputs["decisions"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert {row["parent_id"] for row in parents.to_pylist()} == expected
    assert decisions.num_rows == 3
    assert response.metadata["branch_count"] == 2
    assert response.metadata["output_count"] == len(expected)
    for row in decisions.to_pylist():
        assert row["stage_id"] == "policy_join"
        assert row["reason_code"] in {"POLICY_JOIN_PASS", "POLICY_JOIN_REJECT"}


def test_join_rejects_incomplete_decision_branch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    short_root = tmp_path / "short"
    short_root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [_decision("a", "short", "PASS")],
            schema=DECISION_V1.schema,
        ),
        short_root / "part.parquet",
    )
    inputs["alerts"] = StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "c" * 64, kind=DECISION_V1.id),
        root=short_root,
        contract_id=DECISION_V1.id,
    )
    with pytest.raises(PluginError) as caught:
        NativeDecisionJoinPlugin().execute(
            StageRequest(stage_id="join", inputs=inputs, config={}),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "POLICY_JOIN_INCOMPLETE_BRANCH"


def test_join_rejects_impossible_minimum(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        NativeDecisionJoinPlugin().execute(
            StageRequest(
                stage_id="join",
                inputs=_inputs(tmp_path),
                config={"mode": "min_pass_count", "min_pass_count": 3},
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "POLICY_JOIN_THRESHOLD_INVALID"


def test_warn_only_branch_is_covered_and_non_rejecting(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    warning_root = tmp_path / "warnings"
    warning_root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                _decision("a", "warnings", "WARN"),
                _decision("b", "warnings", "WARN"),
                _decision("c", "warnings", "WARN"),
            ],
            schema=DECISION_V1.schema,
        ),
        warning_root / "part.parquet",
    )
    inputs["alerts"] = StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "e" * 64, kind=DECISION_V1.id),
        root=warning_root,
        contract_id=DECISION_V1.id,
    )
    response = NativeDecisionJoinPlugin().execute(
        StageRequest(stage_id="join", inputs=inputs, config={}),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    assert {row["parent_id"] for row in parents.to_pylist()} == {"a", "c"}
