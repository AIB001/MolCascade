from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1, SYNTHESIS_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.druglikeness import (
    DRUG_LIKENESS_V1,
    RDKitDrugLikenessPlugin,
)
from molcascade.plugins.builtin.structural_alerts import RDKitStructuralAlertPlugin
from molcascade.plugins.builtin.synthesis import RDKitSAScorePlugin
from molcascade.plugins.registry import PluginRegistry


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
            [
                _parent("ethanol", "CCO"),
                _parent("rhodanine", "O=C1NC(=S)SC1"),
                _parent("long_alkane", "C" * 40),
            ],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + "a" * 64,
            kind=PARENT_V1.id,
        ),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(
    plugin: object,
    stage_input: StageInput,
    output: Path,
    config: dict[str, object],
):
    return plugin.execute(  # type: ignore[attr-defined]
        StageRequest(
            stage_id="round4_component",
            inputs={"parents": stage_input},
            config=config,
        ),
        StageContext(output),
    )


def _table(output: Path, response: object, port: str) -> pa.Table:
    path = response.outputs[port].file_paths[0]  # type: ignore[attr-defined]
    return pq.read_table(output / path)


def test_round4_plugins_satisfy_registry_contract_boundaries() -> None:
    plugins = (
        RDKitDrugLikenessPlugin(),
        RDKitStructuralAlertPlugin(),
        RDKitSAScorePlugin(),
    )
    registry = PluginRegistry(plugins)
    for plugin in plugins:
        assert registry.get(plugin.descriptor.key) is plugin


def test_drug_likeness_emits_typed_qed_and_rule_of_five_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence"
    response = _run(
        RDKitDrugLikenessPlugin(),
        _input(tmp_path),
        output,
        {"batch_size": 1},
    )
    parents = _table(output, response, "primary")
    evidence = _table(output, response, "evidence")
    decisions = _table(output, response, "decisions")
    PARENT_V1.validate(parents)
    DRUG_LIKENESS_V1.validate(evidence)
    DECISION_V1.validate(decisions)
    assert parents.num_rows == evidence.num_rows == decisions.num_rows == 3
    rows = {row["parent_id"]: row for row in evidence.to_pylist()}
    assert rows["ethanol"]["rule_of_five_violation_count"] == 0
    assert rows["ethanol"]["rule_of_five_pass"] is True
    assert 0.0 <= rows["ethanol"]["qed_weighted"] <= 1.0
    assert rows["ethanol"]["rule_of_five_method_id"].startswith(
        "rule-of-five:sha256:"
    )
    assert rows["ethanol"]["qed_method_id"].startswith("qed:sha256:")
    assert rows["ethanol"]["toolkit_version"]
    assert (
        rows["ethanol"]["rule_of_five_method_id"]
        != rows["ethanol"]["qed_method_id"]
    )
    assert rows["long_alkane"]["rule_of_five_violation_count"] >= 2
    assert all(row["outcome"] == "PASS" for row in decisions.to_pylist())
    assert {
        row["reason_code"] for row in decisions.to_pylist()
    } == {"DRUG_LIKENESS_EVIDENCE_ANNOTATED"}
    assert response.metadata["policy_enabled"] is False
    assert response.metadata["input_count"] == response.metadata["output_count"] == 3


def test_drug_likeness_combines_qed_and_ro5_policy_without_linear_stages(
    tmp_path: Path,
) -> None:
    stage_input = _input(tmp_path)
    warn_output = tmp_path / "warn"
    warned = _run(
        RDKitDrugLikenessPlugin(),
        stage_input,
        warn_output,
        {
            "batch_size": 2,
            "maximum_rule_of_five_violations": 0,
            "minimum_qed": 0.95,
            "failure_action": "warn",
        },
    )
    assert _table(warn_output, warned, "primary").num_rows == 3
    warned_rows = _table(warn_output, warned, "decisions").to_pylist()
    assert any(row["outcome"] == "WARN" for row in warned_rows)
    assert {
        row["reason_code"] for row in warned_rows if row["entity_id"] == "long_alkane"
    } == {
        "DRUG_LIKENESS_QED_POLICY_FAIL",
        "DRUG_LIKENESS_RO5_POLICY_FAIL",
    }

    reject_output = tmp_path / "reject"
    rejected = _run(
        RDKitDrugLikenessPlugin(),
        stage_input,
        reject_output,
        {
            "maximum_rule_of_five_violations": 0,
            "failure_action": "reject",
        },
    )
    retained_ids = {
        row["parent_id"] for row in _table(reject_output, rejected, "primary").to_pylist()
    }
    assert "long_alkane" not in retained_ids
    assert rejected.metadata["reject_count"] == 1
    assert _table(reject_output, rejected, "evidence").num_rows == 3


def test_structural_alerts_default_to_pains_warning_not_rejection(tmp_path: Path) -> None:
    output = tmp_path / "alerts"
    response = _run(
        RDKitStructuralAlertPlugin(),
        _input(tmp_path),
        output,
        {"batch_size": 1, "decision_buffer_size": 1},
    )
    parents = _table(output, response, "primary")
    decisions = _table(output, response, "decisions")
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert parents.num_rows == 3
    rhodanine = [
        row for row in decisions.to_pylist() if row["entity_id"] == "rhodanine"
    ]
    assert rhodanine
    assert all(row["outcome"] == "WARN" for row in rhodanine)
    detail = json.loads(rhodanine[0]["detail"])
    assert detail["catalog"] == "PAINS"
    assert detail["interpretation"] == "substructure triage flag; not experimental proof"
    assert response.metadata["reject_count"] == 0


def test_structural_alert_catalog_actions_are_independent_and_audited(
    tmp_path: Path,
) -> None:
    output = tmp_path / "alerts-reject"
    response = _run(
        RDKitStructuralAlertPlugin(),
        _input(tmp_path),
        output,
        {
            "pains_action": "ignore",
            "brenk_action": "reject",
            "nih_action": "ignore",
            "zinc_action": "ignore",
        },
    )
    parents = _table(output, response, "primary")
    retained_ids = {row["parent_id"] for row in parents.to_pylist()}
    assert "rhodanine" not in retained_ids
    decisions = _table(output, response, "decisions")
    matches = [
        row for row in decisions.to_pylist() if row["entity_id"] == "rhodanine"
    ]
    assert any(row["outcome"] == "REJECT" for row in matches)
    assert all("STRUCTURAL_ALERT_BRENK_" in row["reason_code"] for row in matches)


def test_sa_score_is_typed_reproducible_proxy_evidence(tmp_path: Path) -> None:
    stage_input = _input(tmp_path)
    first_output = tmp_path / "sa-first"
    second_output = tmp_path / "sa-second"
    first = _run(
        RDKitSAScorePlugin(),
        stage_input,
        first_output,
        {"batch_size": 1},
    )
    second = _run(RDKitSAScorePlugin(), stage_input, second_output, {})
    parents = _table(first_output, first, "primary")
    scores = _table(first_output, first, "synthesis_scores")
    scores_again = _table(second_output, second, "synthesis_scores")
    PARENT_V1.validate(parents)
    SYNTHESIS_SCORE_V1.validate(scores)
    assert parents.num_rows == scores.num_rows == 3
    assert scores.equals(scores_again)
    for row in scores.to_pylist():
        assert row["direction"] == "HIGHER_HARDER"
        assert 1.0 <= row["score"] <= 10.0
        assert "SYNTHESIS_PROXY_NOT_ROUTE" in row["warning_codes_json"]
    assert first.metadata["proxy_not_route"] is True
    assert len(first.metadata["source_sha256"]) == 64
    assert len(first.metadata["weights_sha256"]) == 64


def test_sa_score_reports_missing_contrib_assets_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import molcascade.plugins.builtin.synthesis as synthesis_module

    monkeypatch.setattr(synthesis_module, "_load_sa_backend", lambda: (None, {}))
    with pytest.raises(PluginError) as caught:
        _run(
            RDKitSAScorePlugin(),
            _input(tmp_path),
            tmp_path / "missing",
            {},
        )
    assert caught.value.code == "SYNTHESIS_SASCORE_UNAVAILABLE"
    assert caught.value.hint is not None
    assert not (tmp_path / "missing").exists()
