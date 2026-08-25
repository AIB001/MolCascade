from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.mordred_gate import (
    MordredDescriptorGateConfig,
    MordredDescriptorGatePlugin,
)

pytest.importorskip("mordred")


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


def _input(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> StageInput:
    root = tmp_path / "input"
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows
            or [
                _row("benzene", "c1ccccc1"),
                _row("naphthalene", "c1ccc2ccccc2c1"),
                _row("cyclohexane", "C1CCCCC1"),
                _row("norbornane", "C1CC2CCC1C2"),
            ],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "7" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(tmp_path: Path, config: dict[str, object], rows=None):
    return MordredDescriptorGatePlugin().execute(
        StageRequest(
            stage_id="mordred_gate",
            inputs={"parents": _input(tmp_path, rows)},
            config=config,
        ),
        StageContext(tmp_path / "out"),
    )


def _read(tmp_path: Path, response):
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(tmp_path / "out" / response.outputs["decisions"].file_paths[0])
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    return parents.to_pylist(), decisions.to_pylist()


def test_a_ring_topology_window_keeps_only_the_aromatics(tmp_path: Path) -> None:
    """The gate has to filter on the descriptor it was told to, not a lookalike."""

    response = _run(tmp_path, {"descriptor": "naRing", "minimum": 1})
    parents, decisions = _read(tmp_path, response)
    assert {row["parent_id"] for row in parents} == {"benzene", "naphthalene"}
    by_id = {row["entity_id"]: row for row in decisions}
    assert by_id["benzene"]["outcome"] == "PASS"
    assert by_id["cyclohexane"]["reason_code"] == "MORDRED_NARING_BELOW_MIN"
    assert response.metadata["backend"] == "mordred"
    assert response.metadata["descriptors"] == ["naRing"]


def test_the_aromatic_and_aliphatic_ring_counts_are_genuinely_different_filters(
    tmp_path: Path,
) -> None:
    """``naRing`` and ``nARing`` differ only in case and mean opposite things.

    If Mordred ever folded case, or if the gate normalised names on the way in,
    both spellings would return the same survivors and every config that picked
    the wrong one would look like it worked.  This is the test that notices.
    """

    aromatic = _run(tmp_path / "a", {"descriptor": "naRing", "minimum": 1})
    aliphatic = _run(tmp_path / "b", {"descriptor": "nARing", "minimum": 1})
    kept_aromatic = {row["parent_id"] for row in _read(tmp_path / "a", aromatic)[0]}
    kept_aliphatic = {row["parent_id"] for row in _read(tmp_path / "b", aliphatic)[0]}
    assert kept_aromatic == {"benzene", "naphthalene"}
    assert kept_aliphatic == {"cyclohexane", "norbornane"}
    assert not kept_aromatic & kept_aliphatic


def test_a_descriptor_that_cannot_be_computed_rejects_instead_of_passing(
    tmp_path: Path,
) -> None:
    """Mordred returns a Missing object, not an exception and not a NaN.

    ``Vabc`` is undefined across disconnected fragments, so sodium chloride has
    no van der Waals volume to compare.  Treating that as ``0.0`` would put the
    salt inside any window with a lower bound at or below zero -- it would be
    waved through on the strength of a number that does not exist.
    """

    rows = [_row("salt", "[Na+].[Cl-]"), _row("benzene", "c1ccccc1")]
    response = _run(tmp_path, {"descriptor": "Vabc", "minimum": 50.0}, rows=rows)
    parents, decisions = _read(tmp_path, response)
    assert {row["parent_id"] for row in parents} == {"benzene"}
    by_id = {row["entity_id"]: row for row in decisions}
    assert by_id["salt"]["outcome"] == "REJECT"
    assert by_id["salt"]["reason_code"] == "MORDRED_VABC_UNAVAILABLE"
    assert "undefined for this molecule" in by_id["salt"]["detail"]


def test_an_unknown_descriptor_stops_the_run_and_names_the_near_miss(
    tmp_path: Path,
) -> None:
    """Skipping an unrecognised filter would still produce a plausible artifact."""

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, {"descriptor": "nAromRing", "minimum": 1})
    assert caught.value.code == "MORDRED_DESCRIPTOR_UNKNOWN"
    suggestions = caught.value.context["did_you_mean"]["nAromRing"]
    assert "naRing" in suggestions
    assert caught.value.context["registry_size"] > 1_000


def test_a_case_only_typo_is_reported_as_a_case_only_typo(tmp_path: Path) -> None:
    """The worst failure mode is a name that is almost right, so say so."""

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, {"descriptor": "NARING", "minimum": 1})
    assert caught.value.code == "MORDRED_DESCRIPTOR_UNKNOWN"
    assert set(caught.value.context["did_you_mean"]["NARING"]) >= {"nARing", "naRing"}


def test_several_windows_must_all_hold_at_once(tmp_path: Path) -> None:
    response = _run(
        tmp_path,
        {
            "windows": [
                {"descriptor": "naRing", "minimum": 1},
                {"descriptor": "nRing", "maximum": 1},
            ]
        },
    )
    parents, decisions = _read(tmp_path, response)
    assert {row["parent_id"] for row in parents} == {"benzene"}
    reasons = {row["entity_id"]: row["reason_code"] for row in decisions}
    assert reasons["naphthalene"] == "MORDRED_NRING_ABOVE_MAX"
    assert reasons["cyclohexane"] == "MORDRED_NARING_BELOW_MIN"


def test_the_recorded_pass_detail_carries_the_numbers_that_justified_it(
    tmp_path: Path,
) -> None:
    import json

    response = _run(tmp_path, {"descriptor": "naRing", "minimum": 1})
    _, decisions = _read(tmp_path, response)
    passed = next(row for row in decisions if row["outcome"] == "PASS")
    assert json.loads(passed["detail"]) == {"naRing": 2.0} or json.loads(passed["detail"]) == {
        "naRing": 1.0
    }


def test_the_policy_id_moves_when_the_window_moves(tmp_path: Path) -> None:
    """Two runs with different thresholds must not claim the same policy."""

    first = _run(tmp_path / "a", {"descriptor": "naRing", "minimum": 1})
    second = _run(tmp_path / "b", {"descriptor": "naRing", "minimum": 2})
    assert first.metadata["policy_id"] != second.metadata["policy_id"]
    again = _run(tmp_path / "c", {"descriptor": "naRing", "minimum": 1})
    assert again.metadata["policy_id"] == first.metadata["policy_id"]


def test_batch_size_does_not_change_the_verdicts(tmp_path: Path) -> None:
    whole = _run(tmp_path / "a", {"descriptor": "naRing", "minimum": 1, "batch_size": 16_384})
    split = _run(tmp_path / "b", {"descriptor": "naRing", "minimum": 1, "batch_size": 1})
    assert {row["parent_id"] for row in _read(tmp_path / "a", whole)[0]} == {
        row["parent_id"] for row in _read(tmp_path / "b", split)[0]
    }
    assert whole.metadata["policy_id"] == split.metadata["policy_id"]


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"windows": [{"descriptor": "naRing"}]}, "a window with no bounds rejects nothing"),
        ({"descriptor": "naRing"}, "the shorthand needs a bound too"),
        ({"minimum": 1}, "the shorthand needs a descriptor"),
        (
            {
                "windows": [{"descriptor": "naRing", "minimum": 1}],
                "descriptor": "nRing",
                "maximum": 2,
            },
            "both spellings at once is ambiguous",
        ),
        (
            {"windows": [{"descriptor": "naRing", "minimum": 4, "maximum": 1}]},
            "an inverted window accepts nothing",
        ),
        (
            {
                "windows": [
                    {"descriptor": "naRing", "minimum": 1},
                    {"descriptor": "naRing", "maximum": 3},
                ]
            },
            "two windows on one descriptor",
        ),
        ({"windows": [{"descriptor": "na Ring", "minimum": 1}]}, "not a descriptor name"),
        ({"windows": []}, "no windows at all"),
    ],
)
def test_unusable_configurations_are_refused_before_any_molecule_is_read(
    tmp_path: Path, config: dict[str, object], reason: str
) -> None:
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, config)
    assert caught.value.code == "PLUGIN_CONFIG_INVALID", reason


def test_the_curated_catalogue_descriptors_all_exist_in_the_installed_release() -> None:
    """The builder offers a fixed list; Mordred is free to rename things.

    A name that has drifted out of the registry only fails at run time, after the
    user has assembled a flow and started a screen.  Catching it here costs one
    import.
    """

    from mordred import Calculator
    from mordred import descriptors as mordred_descriptors

    from molcascade.cascade.catalog import CRITERIA_BY_ID

    available = {str(item) for item in Calculator(mordred_descriptors, ignore_3D=True).descriptors}
    option = next(
        item
        for item in CRITERIA_BY_ID["physchem_window"].options
        if item.id == "mordred_descriptor_window"
    )
    offered = {
        value
        for field in option.thresholds
        if field.name == "descriptor"
        for value, _ in field.choices
    }
    assert offered, "the descriptor field must offer a curated choice list"
    assert offered <= available, sorted(offered - available)
    assert MordredDescriptorGateConfig.model_validate(dict(option.defaults))
