"""The gate that turns a derived number into a decision.

A derived metric is arithmetic over evidence an earlier tier produced, and the
arithmetic lives in a featurizer so that what was computed stays in the trace
whether or not it decided anything.  These tests pin down the half that decides:
which rows it reads, which it refuses to read, and what it does with a molecule
the metric admits it could not measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, DERIVED_METRIC_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.evidence_gates import (
    DerivedMetricDirection,
    DerivedMetricStatus,
    DerivedMetricUnits,
    NativeDerivedMetricEvidenceGatePlugin,
)

_STRAIN_WINDOW: dict[str, object] = {
    "metric_id": "pose_strain",
    "method_id": None,
    "expected_units": "KCAL_PER_MOL",
    "expected_direction": "LOWER_BETTER",
    "maximum": 8.0,
    "batch_size": 2,
}


def _parent(parent_id: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": {"p1": "CCO", "p2": "CCN", "p3": "CCC"}.get(parent_id, "CO"),
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _metric(
    parent_id: str,
    value: float | None,
    *,
    metric_id: str = "pose_strain",
    method_id: str = "mmff94s:12:seed-7",
    units: str = "KCAL_PER_MOL",
    direction: str = "LOWER_BETTER",
    status: str = "OK",
    status_detail: str | None = None,
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "metric_id": metric_id,
        "method_id": method_id,
        "value": value,
        "units": units,
        "direction": direction,
        "status": status,
        "status_detail": status_detail,
        "source_json": None,
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


def _parents(tmp_path: Path, name: str = "parents") -> StageInput:
    return _dataset(
        tmp_path,
        name,
        PARENT_V1,
        [_parent("p1"), _parent("p2"), _parent("p3")],
        artifact_hex="a",
    )


def _metrics(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    name: str = "metrics",
    artifact_hex: str = "b",
) -> StageInput:
    return _dataset(tmp_path, name, DERIVED_METRIC_V1, rows, artifact_hex=artifact_hex)


def _read(output: Path, response: object, port: str) -> pa.Table:
    relative = response.outputs[port].file_paths[0]  # type: ignore[attr-defined]
    return pq.read_table(output / relative)


def _run(
    tmp_path: Path,
    rows: list[dict[str, object]],
    config: dict[str, object],
    *,
    suffix: str = "",
) -> tuple[Path, object]:
    output = tmp_path / f"out{suffix}"
    response = NativeDerivedMetricEvidenceGatePlugin().execute(
        StageRequest(
            stage_id=f"metric_window{suffix}",
            inputs={
                "parents": _parents(tmp_path, f"parents{suffix}"),
                "evidence": _metrics(tmp_path, rows, name=f"metrics{suffix}"),
            },
            config=config,
        ),
        StageContext(output),
    )
    return output, response


def test_derived_metric_gate_applies_an_inclusive_window_and_rejects_absent_rows(
    tmp_path: Path,
) -> None:
    output, response = _run(
        tmp_path,
        [_metric("p1", 8.0), _metric("p2", 8.01)],
        dict(_STRAIN_WINDOW),
    )

    parents = _read(output, response, "primary")
    decisions = _read(output, response, "decisions")
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    by_parent = {row["entity_id"]: row for row in decisions.to_pylist()}
    assert by_parent["p1"]["reason_code"] == "DERIVED_METRIC_PASS"
    assert by_parent["p2"]["reason_code"] == "DERIVED_METRIC_ABOVE_MAXIMUM"
    assert by_parent["p3"]["reason_code"] == "DERIVED_METRIC_MISSING"
    assert json.loads(by_parent["p3"]["detail"])["value"] is None
    assert response.metadata["method_id"] == "mmff94s:12:seed-7"  # type: ignore[attr-defined]
    assert response.metadata["output_count"] == 1  # type: ignore[attr-defined]
    assert response.metadata["unscorable_policy"] == "reject"  # type: ignore[attr-defined]


def test_derived_metric_gate_rejects_a_number_the_method_disowns(tmp_path: Path) -> None:
    """A negative strain sits inside the window and means nothing.

    The MMFF94s backend reports strain as the relaxed pose energy minus the
    minimum of a conformer ensemble, so a failed reference search comes out
    negative -- about a tenth of real molecules.  ``-19.42`` is comfortably under
    ``maximum=8``, which is why the status has to be consulted before the value:
    reading the number alone would admit exactly the molecules the metric knows
    it could not measure, and they would arrive labelled as the best in the run.
    """

    output, response = _run(
        tmp_path,
        [
            _metric("p1", 1.84),
            _metric(
                "p2",
                -19.42,
                status=DerivedMetricStatus.OUT_OF_DOMAIN.value,
                status_detail="ensemble minimum above the relaxed pose",
            ),
        ],
        dict(_STRAIN_WINDOW),
    )

    parents = _read(output, response, "primary")
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    decisions = {row["entity_id"]: row for row in _read(output, response, "decisions").to_pylist()}
    assert decisions["p2"]["outcome"] == "REJECT"
    assert decisions["p2"]["reason_code"] == "DERIVED_METRIC_OUT_OF_DOMAIN"
    # The measurement is still recorded: it is the evidence that the reference
    # search failed, and a reader of the trace needs to see it to know that.
    assert json.loads(decisions["p2"]["detail"])["value"] == -19.42


def test_derived_metric_gate_can_be_asked_to_admit_what_it_could_not_score(
    tmp_path: Path,
) -> None:
    output, response = _run(
        tmp_path,
        [
            _metric("p1", 1.84),
            _metric("p2", None, status=DerivedMetricStatus.BACKEND_FAILED.value),
        ],
        dict(_STRAIN_WINDOW) | {"on_unscorable": "accept"},
    )

    parents = _read(output, response, "primary")
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1", "p2"]
    decisions = {row["entity_id"]: row for row in _read(output, response, "decisions").to_pylist()}
    assert decisions["p2"]["reason_code"] == "DERIVED_METRIC_UNSCORABLE_ACCEPTED"
    # Absent evidence is a different thing from a metric that declined to score,
    # and opting into the second does not opt into the first.
    assert decisions["p3"]["reason_code"] == "DERIVED_METRIC_MISSING"
    assert response.metadata["unscorable_policy"] == "accept"  # type: ignore[attr-defined]


def test_derived_metric_gate_refuses_a_window_on_another_scale(tmp_path: Path) -> None:
    """``8`` is three quarters of the population in kcal/mol and nonsense in TEU.

    Both are small positive reals called strain, so nothing downstream would
    reveal the mix-up: the run would simply filter at a strength nobody chose.
    """

    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            [_metric("p1", 1.84, units=DerivedMetricUnits.TEU.value)],
            dict(_STRAIN_WINDOW),
        )

    assert caught.value.code == "DERIVED_METRIC_GATE_UNITS_MISMATCH"
    assert caught.value.context["expected_units"] == "KCAL_PER_MOL"
    assert caught.value.context["observed_units"] == ["TEU"]


def test_derived_metric_gate_refuses_a_window_read_backwards(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            [
                _metric(
                    "p1",
                    1.84,
                    direction=DerivedMetricDirection.HIGHER_BETTER.value,
                )
            ],
            dict(_STRAIN_WINDOW),
        )

    assert caught.value.code == "EVIDENCE_GATE_DIRECTION_MISMATCH"
    assert caught.value.context["observed_directions"] == ["HIGHER_BETTER"]


def test_derived_metric_gate_names_the_metrics_it_could_have_gated(
    tmp_path: Path,
) -> None:
    """A typo in metric_id would otherwise reject the whole population silently.

    A LEFT JOIN on a name no row carries is indistinguishable from a featurizer
    that scored nobody, so the gate would run cleanly and produce an empty
    shortlist.  The list of names that *are* present is the entire diagnosis.
    """

    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            [
                _metric("p1", 0.34, metric_id="ligand_efficiency"),
                _metric("p1", 1.84, metric_id="pose_strain"),
            ],
            dict(_STRAIN_WINDOW) | {"metric_id": "pose_stain"},
        )

    assert caught.value.code == "DERIVED_METRIC_GATE_METRIC_UNAVAILABLE"
    assert caught.value.context["available_metric_ids"] == [
        "ligand_efficiency",
        "pose_strain",
    ]


def test_derived_metric_gate_requires_one_method_unless_it_is_pinned(
    tmp_path: Path,
) -> None:
    """Two conformer seeds are two measurements, and averaging them is not a gate."""

    rows = [
        _metric("p1", 1.84, method_id="mmff94s:12:seed-7"),
        _metric("p1", 3.10, method_id="mmff94s:12:seed-11"),
    ]
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, rows, dict(_STRAIN_WINDOW))

    assert caught.value.code == "EVIDENCE_GATE_IDENTITY_AMBIGUOUS"
    assert caught.value.context["observed_count"] == 2

    output, response = _run(
        tmp_path,
        rows,
        dict(_STRAIN_WINDOW) | {"method_id": "mmff94s:12:seed-11"},
        suffix="-pinned",
    )
    decisions = {row["entity_id"]: row for row in _read(output, response, "decisions").to_pylist()}
    assert json.loads(decisions["p1"]["detail"])["value"] == 3.10


def test_derived_metric_gate_reads_only_the_metric_it_was_given(tmp_path: Path) -> None:
    """One featurizer writes several metrics; one gate thresholds one of them.

    The other rows are not merely ignored -- they must not leak into the
    decision, or a molecule would be judged against a window chosen for a
    different quantity on a different scale.
    """

    output, response = _run(
        tmp_path,
        [
            _metric("p1", 1.84),
            _metric(
                "p1",
                0.34,
                metric_id="ligand_efficiency",
                method_id="le:hac",
                units=DerivedMetricUnits.KCAL_PER_MOL_PER_HEAVY_ATOM.value,
                direction=DerivedMetricDirection.HIGHER_BETTER.value,
            ),
            _metric("p2", 11.41),
        ],
        dict(_STRAIN_WINDOW),
    )

    parents = _read(output, response, "primary")
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1"]
    assert response.metadata["evidence_count"] == 3  # type: ignore[attr-defined]
    detail = json.loads(
        next(
            row
            for row in _read(output, response, "decisions").to_pylist()
            if row["entity_id"] == "p1"
        )["detail"]
    )
    assert detail["metric_id"] == "pose_strain"
    assert detail["value"] == 1.84


def test_derived_metric_gate_requires_at_least_one_bound(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            [_metric("p1", 1.84)],
            {
                "metric_id": "pose_strain",
                "expected_units": "KCAL_PER_MOL",
                "expected_direction": "LOWER_BETTER",
            },
        )

    assert caught.value.code == "PLUGIN_CONFIG_INVALID"


def test_derived_metric_gate_refuses_a_row_that_claims_a_value_it_lacks(
    tmp_path: Path,
) -> None:
    """``value`` has to stay nullable, so this invariant can only be checked here.

    A row saying the metric computed successfully and then carrying no number
    means the producer disagrees with itself.  Treating the null as a rejection
    would make it look identical to a molecule that missed the window.
    """

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, [_metric("p1", None)], dict(_STRAIN_WINDOW))

    assert caught.value.code == "DERIVED_METRIC_GATE_STATUS_INCONSISTENT"
    assert caught.value.context["status"] == "OK"


def test_derived_metric_gate_vocabularies_track_the_contract() -> None:
    """The gate hardcodes the contract's enums, so drift has to be a test failure.

    Adding a scale to the contract and forgetting it here would not break a
    build; it would make evidence on the new scale fail closed with a message
    about an invalid identity, pointing at the producer instead of at this file.
    """

    assert {item.value for item in DerivedMetricUnits} == DERIVED_METRIC_V1.enum_values["units"]
    assert {item.value for item in DerivedMetricDirection} == DERIVED_METRIC_V1.enum_values[
        "direction"
    ]
    assert {item.value for item in DerivedMetricStatus} == DERIVED_METRIC_V1.enum_values["status"]


def test_redock_angstrom_gate_is_strict_and_records_its_boundary(tmp_path: Path) -> None:
    output, response = _run(
        tmp_path,
        [
            _metric("p1", 1.999, metric_id="dock_redock_rmsd", units="ANGSTROM"),
            _metric("p2", 2.0, metric_id="dock_redock_rmsd", units="ANGSTROM"),
            _metric("p3", None, metric_id="dock_redock_rmsd", units="ANGSTROM",
                    status="BACKEND_FAILED"),
        ],
        {"metric_id": "dock_redock_rmsd", "expected_units": "ANGSTROM",
         "expected_direction": "LOWER_BETTER", "maximum": 2.0,
         "maximum_exclusive": True, "on_unscorable": "reject"},
    )
    assert _read(output, response, "primary").column("parent_id").to_pylist() == ["p1"]
    decisions = _read(output, response, "decisions").to_pylist()
    detail = json.loads(decisions[1]["detail"])
    assert detail["maximum_exclusive"] is True
    assert detail["bounds_inclusive"] is False
    assert decisions[2]["reason_code"] == "DERIVED_METRIC_BACKEND_FAILED"
