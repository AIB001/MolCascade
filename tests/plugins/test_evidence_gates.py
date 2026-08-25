from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import (
    DECISION_V1,
    DOCKING_SCORE_V1,
    PARENT_V1,
    PREDICTION_V1,
    SYNTHESIS_SCORE_V1,
)
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.evidence_gates import (
    NativeDockingScoreEvidenceGatePlugin,
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


_RECEPTOR = "sha256:" + "d" * 64
_OTHER_RECEPTOR = "sha256:" + "e" * 64


def _pose(
    parent_id: str,
    score: float,
    *,
    pose_rank: int = 0,
    engine_id: str = "unidock",
    receptor_id: str = _RECEPTOR,
    score_kind: str = "VINA_KCAL_MOL",
    direction: str = "LOWER_STRONGER",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "engine_id": engine_id,
        "receptor_id": receptor_id,
        "pose_rank": pose_rank,
        "score": score,
        "score_kind": score_kind,
        "direction": direction,
        "secondary_score": None,
        "pose_molblock": None,
        "method_id": "docking:test",
    }


def _cnn(parent_id: str, score: float, *, pose_rank: int = 0) -> dict[str, object]:
    """A GNINA-shaped row: the other scale, and the other direction."""

    return _pose(
        parent_id,
        score,
        pose_rank=pose_rank,
        engine_id="gnina",
        score_kind="CNN_AFFINITY",
        direction="HIGHER_STRONGER",
    )


def _dock(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    config: dict[str, object],
    stage_id: str = "docking_window",
    name: str = "poses",
) -> tuple[Path, object]:
    output = tmp_path / f"out-{stage_id}"
    # Named per call rather than shared, so a test can gate twice in one
    # tmp_path -- comparing two receptors needs exactly that.
    parents = _dataset(
        tmp_path,
        f"parents-{stage_id}",
        PARENT_V1,
        [_parent("p1"), _parent("p2"), _parent("p3")],
        artifact_hex="a",
    )
    response = NativeDockingScoreEvidenceGatePlugin().execute(
        StageRequest(
            stage_id=stage_id,
            inputs={
                "parents": parents,
                "scores": _dataset(tmp_path, name, DOCKING_SCORE_V1, rows, artifact_hex="c"),
            },
            config=config,
        ),
        StageContext(output),
    )
    return output, response


def test_docking_gate_scores_the_best_pose_not_the_first_one(tmp_path: Path) -> None:
    """The pose that decides a molecule is the strongest one, whatever its rank.

    ``pose_rank`` is written best-first by every engine here, so on well-formed
    evidence the two agree.  The test feeds evidence where they do not, because
    a gate that reads rank 0 and calls it the answer would invert silently the
    day an engine's ranking and the configured direction disagree -- keeping a
    library's worst poses and reporting them as hits.
    """

    output, response = _dock(
        tmp_path,
        [
            _pose("p1", -7.0, pose_rank=0),
            _pose("p1", -9.5, pose_rank=1),
            _pose("p2", -6.0),
        ],
        config={
            "engine_id": "unidock",
            "receptor_id": None,
            "expected_score_kind": "VINA_KCAL_MOL",
            "expected_direction": "LOWER_STRONGER",
            "maximum": -8.0,
            "batch_size": 2,
        },
    )

    parents = _read(output, response, "primary")
    decisions = _read(output, response, "decisions")
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    by_parent = {row["entity_id"]: row for row in decisions.to_pylist()}
    assert by_parent["p1"]["outcome"] == "PASS"
    assert by_parent["p2"]["reason_code"] == "DOCKING_SCORE_ABOVE_MAXIMUM"
    assert by_parent["p3"]["reason_code"] == "DOCKING_SCORE_MISSING"

    detail = json.loads(by_parent["p1"]["detail"])
    assert detail["score"] == -9.5
    # Which pose decided it, and out of how many -- an audit reader can go back
    # to the evidence artifact and find the geometry this number came from.
    assert detail["pose_rank"] == 1
    assert detail["pose_count"] == 2
    assert detail["not_affinity"] is True
    assert json.loads(by_parent["p3"]["detail"])["pose_count"] == 0
    assert response.metadata["receptor_id"] == _RECEPTOR
    assert response.metadata["evidence_count"] == 3
    assert response.metadata["output_count"] == 1


def test_docking_gate_takes_the_maximum_when_higher_is_stronger(tmp_path: Path) -> None:
    output, response = _dock(
        tmp_path,
        [
            _cnn("p1", 5.1, pose_rank=0),
            _cnn("p1", 6.4, pose_rank=1),
            _cnn("p2", 4.0),
        ],
        config={
            "engine_id": "gnina",
            "expected_score_kind": "CNN_AFFINITY",
            "expected_direction": "HIGHER_STRONGER",
            "minimum": 6.0,
        },
    )

    # 6.4 is p1's best pose and 5.1 is the one ranked first; taking the maximum
    # is the difference between keeping this molecule and dropping it.
    assert [row["parent_id"] for row in _read(output, response, "primary").to_pylist()] == ["p1"]
    by_parent = {
        row["entity_id"]: row for row in _read(output, response, "decisions").to_pylist()
    }
    assert json.loads(by_parent["p1"]["detail"])["score"] == 6.4
    assert by_parent["p2"]["reason_code"] == "DOCKING_SCORE_BELOW_MINIMUM"
    assert response.metadata["score_kind"] == "CNN_AFFINITY"
    assert response.metadata["output_count"] == 1


@pytest.mark.parametrize(
    ("expected_score_kind", "expected_direction"),
    [
        # The right scale read backwards: -9.5 would look like a rejection and
        # -6.0 like a hit, and every number in the report would still be true.
        ("VINA_KCAL_MOL", "HIGHER_STRONGER"),
        # The wrong scale: a threshold of -8 kcal/mol against a 0-1 CNN score
        # rejects the entire library and reads like a strict campaign.
        ("CNN_SCORE", "LOWER_STRONGER"),
    ],
)
def test_docking_gate_refuses_evidence_on_another_scale_or_direction(
    tmp_path: Path,
    expected_score_kind: str,
    expected_direction: str,
) -> None:
    with pytest.raises(PluginError) as caught:
        _dock(
            tmp_path,
            [_pose("p1", -9.5), _pose("p2", -6.0)],
            config={
                "engine_id": "unidock",
                "expected_score_kind": expected_score_kind,
                "expected_direction": expected_direction,
                "maximum": -8.0,
            },
            stage_id="semantics",
        )

    assert caught.value.code == "EVIDENCE_GATE_SCORE_SEMANTICS_MISMATCH"
    assert not (tmp_path / "out-semantics" / ".docking-evidence-gate.sqlite3").exists()


def test_docking_gate_will_not_guess_between_two_engines(tmp_path: Path) -> None:
    """Two engines in one evidence stream is the shape a parallel tier produces."""

    with pytest.raises(PluginError) as caught:
        _dock(
            tmp_path,
            [_pose("p1", -9.5), _cnn("p1", 6.4)],
            config={
                "expected_score_kind": "VINA_KCAL_MOL",
                "expected_direction": "LOWER_STRONGER",
                "maximum": -8.0,
            },
            stage_id="ambiguous-engine",
        )

    assert caught.value.code == "EVIDENCE_GATE_IDENTITY_AMBIGUOUS"
    assert caught.value.context["observed_examples"] == ["gnina", "unidock"]


def test_docking_gate_will_not_compare_across_receptors(tmp_path: Path) -> None:
    """Two structures are two questions, and one threshold cannot answer both."""

    with pytest.raises(PluginError) as caught:
        _dock(
            tmp_path,
            [_pose("p1", -9.5), _pose("p2", -9.9, receptor_id=_OTHER_RECEPTOR)],
            config={
                "engine_id": "unidock",
                "expected_score_kind": "VINA_KCAL_MOL",
                "expected_direction": "LOWER_STRONGER",
                "maximum": -8.0,
            },
            stage_id="ambiguous-receptor",
        )

    assert caught.value.code == "EVIDENCE_GATE_IDENTITY_AMBIGUOUS"
    assert caught.value.context["identity"] == "receptor_id for the configured engine"


def test_docking_gate_naming_the_engine_ignores_the_other_one(tmp_path: Path) -> None:
    output, response = _dock(
        tmp_path,
        [_pose("p1", -9.5), _pose("p2", -6.0), _cnn("p1", 0.1), _cnn("p2", 9.9)],
        config={
            "engine_id": "unidock",
            "receptor_id": _RECEPTOR,
            "expected_score_kind": "VINA_KCAL_MOL",
            "expected_direction": "LOWER_STRONGER",
            "maximum": -8.0,
        },
        stage_id="one-engine",
    )

    assert [row["parent_id"] for row in _read(output, response, "primary").to_pylist()] == ["p1"]
    assert response.metadata["engine_id"] == "unidock"


def test_docking_gate_policy_id_is_bound_to_the_receptor_bytes(tmp_path: Path) -> None:
    """Rescreening against a different structure is a different policy, by identity.

    ``policy_id`` travels into every decision row as ``rule_id``.  If it did not
    move with the receptor digest, two runs against two structures would record
    the same rule for contradictory verdicts, and nothing downstream could tell
    them apart.
    """

    config: dict[str, object] = {
        "engine_id": "unidock",
        "receptor_id": _RECEPTOR,
        "expected_score_kind": "VINA_KCAL_MOL",
        "expected_direction": "LOWER_STRONGER",
        "maximum": -8.0,
    }
    _, first = _dock(
        tmp_path,
        [_pose("p1", -9.5)],
        config=config,
        stage_id="receptor-a",
        name="poses-a",
    )
    _, second = _dock(
        tmp_path,
        [_pose("p1", -9.5, receptor_id=_OTHER_RECEPTOR)],
        config={**config, "receptor_id": _OTHER_RECEPTOR},
        stage_id="receptor-b",
        name="poses-b",
    )

    assert first.metadata["policy_id"] != second.metadata["policy_id"]
    assert str(first.metadata["policy_id"]).startswith("docking-gate:sha256:")


def test_docking_gate_rejects_a_repeated_pose_rank(tmp_path: Path) -> None:
    """Two rows claiming the same pose is a contract violation, not a tie."""

    with pytest.raises(PluginError) as caught:
        _dock(
            tmp_path,
            [_pose("p1", -9.5, pose_rank=0), _pose("p1", -9.4, pose_rank=0)],
            config={
                "engine_id": "unidock",
                "expected_score_kind": "VINA_KCAL_MOL",
                "expected_direction": "LOWER_STRONGER",
                "maximum": -8.0,
            },
            stage_id="duplicate",
        )

    assert caught.value.code == "EVIDENCE_GATE_DUPLICATE_EVIDENCE"


def test_docking_gate_rejects_a_non_finite_score(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _dock(
            tmp_path,
            [_pose("p1", math.nan)],
            config={
                "engine_id": "unidock",
                "expected_score_kind": "VINA_KCAL_MOL",
                "expected_direction": "LOWER_STRONGER",
                "maximum": -8.0,
            },
            stage_id="non-finite",
        )

    assert caught.value.code == "EVIDENCE_GATE_NON_FINITE"
