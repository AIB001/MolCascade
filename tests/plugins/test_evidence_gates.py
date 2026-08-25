from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1, PREDICTION_V1, SYNTHESIS_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.evidence_gates import (
    NativePredictionEvidenceGatePlugin,
    NativeSynthesisScoreEvidenceGatePlugin,
)


def _parent(parent_id: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": {"p1": "CCO", "p2": "CCN", "p3": "CCC"}.get(
            parent_id,
            "CO",
        ),
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _prediction(
    parent_id: str,
    value: float,
    *,
    endpoint_id: str = "endpoint:test",
    model_id: str = "model:test",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "prediction_mean": value,
        "prediction_std": None,
        "interval_lower": None,
        "interval_upper": None,
        "calibration_id": None,
    }


def _synthesis(
    parent_id: str,
    score: float,
    *,
    method_id: str = "sa:test",
    direction: str = "HIGHER_HARDER",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "method_id": method_id,
        "score": score,
        "direction": direction,
        "domain": "test fixture",
        "warning_codes_json": "[]",
    }


def _dataset(
    tmp_path: Path,
    name: str,
    contract: object,
    rows: list[dict[str, object]],
    *,
    artifact_hex: str,
) -> StageInput:
    root = tmp_path / name
    root.mkdir()
    schema = contract.schema  # type: ignore[attr-defined]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "part.parquet")
    contract_id = contract.id  # type: ignore[attr-defined]
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + artifact_hex * 64,
            kind=contract_id,
        ),
        root=root,
        contract_id=contract_id,
    )


def _parents(tmp_path: Path) -> StageInput:
    return _dataset(
        tmp_path,
        "parents",
        PARENT_V1,
        [_parent("p1"), _parent("p2"), _parent("p3")],
        artifact_hex="a",
    )


def _read(output: Path, response: object, port: str) -> pa.Table:
    relative = response.outputs[port].file_paths[0]  # type: ignore[attr-defined]
    return pq.read_table(output / relative)


def test_prediction_gate_filters_exact_model_and_rejects_missing_evidence(
    tmp_path: Path,
) -> None:
    predictions = _dataset(
        tmp_path,
        "predictions",
        PREDICTION_V1,
        [_prediction("p1", 0.4), _prediction("p2", 0.8)],
        artifact_hex="b",
    )
    output = tmp_path / "out"
    response = NativePredictionEvidenceGatePlugin().execute(
        StageRequest(
            stage_id="admet_window",
            inputs={"parents": _parents(tmp_path), "predictions": predictions},
            config={
                "endpoint_id": "endpoint:test",
                "model_id": None,
                "semantics_label": "Fixture probability; higher is not always better.",
                "minimum": 0.2,
                "maximum": 0.7,
                "batch_size": 1,
            },
        ),
        StageContext(output),
    )

    parents = _read(output, response, "primary")
    decisions = _read(output, response, "decisions")
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    by_parent = {row["entity_id"]: row for row in decisions.to_pylist()}
    assert by_parent["p1"]["outcome"] == "PASS"
    assert by_parent["p2"]["reason_code"] == "PREDICTION_ABOVE_MAXIMUM"
    assert by_parent["p3"]["reason_code"] == "PREDICTION_MISSING"
    assert json.loads(by_parent["p3"]["detail"])["prediction_mean"] is None
    assert response.metadata["model_id"] == "model:test"
    assert response.metadata["output_count"] == 1


def test_synthesis_gate_uses_inclusive_window_and_explicit_direction(
    tmp_path: Path,
) -> None:
    scores = _dataset(
        tmp_path,
        "scores",
        SYNTHESIS_SCORE_V1,
        [_synthesis("p1", 6.0), _synthesis("p2", 6.1)],
        artifact_hex="c",
    )
    output = tmp_path / "out"
    response = NativeSynthesisScoreEvidenceGatePlugin().execute(
        StageRequest(
            stage_id="sa_window",
            inputs={"parents": _parents(tmp_path), "scores": scores},
            config={
                "maximum": 6.0,
                "method_id": None,
                "expected_direction": "HIGHER_HARDER",
                "batch_size": 2,
            },
        ),
        StageContext(output),
    )

    parents = _read(output, response, "primary")
    decisions = _read(output, response, "decisions")
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    by_parent = {row["entity_id"]: row for row in decisions.to_pylist()}
    assert by_parent["p1"]["reason_code"] == "SYNTHESIS_SCORE_PASS"
    assert by_parent["p2"]["reason_code"] == "SYNTHESIS_SCORE_ABOVE_MAXIMUM"
    assert by_parent["p3"]["reason_code"] == "SYNTHESIS_SCORE_MISSING"
    assert response.metadata["method_id"] == "sa:test"
    assert response.metadata["expected_direction"] == "HIGHER_HARDER"


def test_prediction_gate_requires_unambiguous_auto_selected_model(tmp_path: Path) -> None:
    predictions = _dataset(
        tmp_path,
        "predictions",
        PREDICTION_V1,
        [
            _prediction("p1", 0.4, model_id="model:a"),
            _prediction("p2", 0.5, model_id="model:b"),
        ],
        artifact_hex="b",
    )
    with pytest.raises(PluginError) as caught:
        NativePredictionEvidenceGatePlugin().execute(
            StageRequest(
                stage_id="ambiguous",
                inputs={"parents": _parents(tmp_path), "predictions": predictions},
                config={
                    "endpoint_id": "endpoint:test",
                    "semantics_label": "Fixture endpoint.",
                    "minimum": 0.0,
                },
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "EVIDENCE_GATE_IDENTITY_AMBIGUOUS"
    assert not (tmp_path / "out" / ".prediction-evidence-gate.sqlite3").exists()


def test_synthesis_gate_rejects_direction_mismatch(tmp_path: Path) -> None:
    scores = _dataset(
        tmp_path,
        "scores",
        SYNTHESIS_SCORE_V1,
        [_synthesis("p1", 0.7, direction="HIGHER_EASIER")],
        artifact_hex="c",
    )
    with pytest.raises(PluginError) as caught:
        NativeSynthesisScoreEvidenceGatePlugin().execute(
            StageRequest(
                stage_id="direction",
                inputs={"parents": _parents(tmp_path), "scores": scores},
                config={
                    "minimum": 0.5,
                    "expected_direction": "HIGHER_HARDER",
                },
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "EVIDENCE_GATE_DIRECTION_MISMATCH"


def test_prediction_gate_accepts_retained_upstream_evidence_after_serial_filter(
    tmp_path: Path,
) -> None:
    current_parents = _dataset(
        tmp_path,
        "current-parents",
        PARENT_V1,
        [_parent("p1")],
        artifact_hex="d",
    )
    predictions = _dataset(
        tmp_path,
        "predictions",
        PREDICTION_V1,
        [_prediction("p1", 0.4), _prediction("p2", 0.8), _prediction("p3", 0.9)],
        artifact_hex="b",
    )
    output = tmp_path / "out"

    response = NativePredictionEvidenceGatePlugin().execute(
        StageRequest(
            stage_id="serial_second_gate",
            inputs={"survivors": current_parents, "retained_predictions": predictions},
            config={
                "endpoint_id": "endpoint:test",
                "model_id": None,
                "semantics_label": "Fixture endpoint.",
                "minimum": 0.2,
                "maximum": 0.7,
            },
        ),
        StageContext(output),
    )

    assert [
        row["parent_id"] for row in _read(output, response, "primary").to_pylist()
    ] == ["p1"]
    assert _read(output, response, "decisions").num_rows == 1
    assert response.metadata["retained_evidence_ignored_count"] == 2


@pytest.mark.parametrize(
    ("rows", "expected_code"),
    [
        ([_prediction("p1", math.nan)], "EVIDENCE_GATE_NON_FINITE"),
        (
            [_prediction("p1", 0.2), _prediction("p1", 0.3)],
            "EVIDENCE_GATE_DUPLICATE_EVIDENCE",
        ),
    ],
)
def test_prediction_gate_rejects_malformed_evidence(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected_code: str,
) -> None:
    predictions = _dataset(
        tmp_path,
        "predictions",
        PREDICTION_V1,
        rows,
        artifact_hex="b",
    )
    with pytest.raises(PluginError) as caught:
        NativePredictionEvidenceGatePlugin().execute(
            StageRequest(
                stage_id="malformed",
                inputs={"parents": _parents(tmp_path), "predictions": predictions},
                config={
                    "endpoint_id": "endpoint:test",
                    "model_id": "model:test",
                    "semantics_label": "Fixture endpoint.",
                    "minimum": 0.0,
                },
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == expected_code


def test_numeric_gate_configuration_requires_at_least_one_bound(tmp_path: Path) -> None:
    predictions = _dataset(
        tmp_path,
        "predictions",
        PREDICTION_V1,
        [_prediction("p1", 0.2)],
        artifact_hex="b",
    )
    with pytest.raises(PluginError) as caught:
        NativePredictionEvidenceGatePlugin().execute(
            StageRequest(
                stage_id="unbounded",
                inputs={"parents": _parents(tmp_path), "predictions": predictions},
                config={
                    "endpoint_id": "endpoint:test",
                    "model_id": "model:test",
                    "semantics_label": "Fixture endpoint.",
                },
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "PLUGIN_CONFIG_INVALID"
