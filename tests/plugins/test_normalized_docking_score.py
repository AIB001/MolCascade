"""Reading a docking score against the size of the molecule that earned it.

A Vina-family score grows with heavy atoms, so the strongest score in a library
is often just the largest molecule in it.  This module computes three ways of
correcting for that and writes all three, deciding nothing.  These tests pin
down the arithmetic, which pose it is computed from, which scales it refuses,
and how it reports a molecule that has no score at all -- because a plausible
number produced from the wrong pose or the wrong engine is the failure that
would survive review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DERIVED_METRIC_V1, DOCKING_SCORE_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.normalized_docking_score import (
    NormalizedDockingScoreConfig,
    NormalizedDockingScorePlugin,
    NormalizedScoreKind,
)

#: Three molecules with three different heavy-atom counts, so a metric that
#: forgot to divide is visibly wrong rather than coincidentally right.
_SMILES = {
    "p1": "c1ccccc1",  # benzene, 6 heavy atoms
    "p2": "CC(=O)Oc1ccccc1C(=O)O",  # aspirin, 13 heavy atoms
    "p3": "CCO",  # ethanol, 3 heavy atoms
}
_HEAVY_ATOMS = {"p1": 6, "p2": 13, "p3": 3}


def _parent(parent_id: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": _SMILES[parent_id],
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _pose(
    parent_id: str,
    score: float,
    *,
    pose_rank: int = 0,
    engine_id: str = "unidock",
    receptor_id: str = "receptor:sha256:" + "c" * 64,
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
        "method_id": f"{engine_id}:test",
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


def _read(output: Path, response: object, port: str) -> pa.Table:
    relative = response.outputs[port].file_paths[0]  # type: ignore[attr-defined]
    return pq.read_table(output / relative)


def _run(
    tmp_path: Path,
    poses: list[dict[str, object]],
    config: dict[str, object] | None = None,
    *,
    parent_ids: tuple[str, ...] = ("p1", "p2", "p3"),
    suffix: str = "",
) -> tuple[Path, object]:
    output = tmp_path / f"out{suffix}"
    response = NormalizedDockingScorePlugin().execute(
        StageRequest(
            stage_id=f"normalized_score{suffix}",
            inputs={
                "parents": _dataset(
                    tmp_path,
                    f"parents{suffix}",
                    PARENT_V1,
                    [_parent(parent_id) for parent_id in parent_ids],
                    artifact_hex="a",
                ),
                "evidence": _dataset(
                    tmp_path,
                    f"poses{suffix}",
                    DOCKING_SCORE_V1,
                    poses,
                    artifact_hex="b",
                ),
            },
            config=dict(config or {}),
        ),
        StageContext(output),
    )
    return output, response


def _by_metric(table: pa.Table) -> dict[tuple[str, str], dict[str, object]]:
    return {(row["parent_id"], row["metric_id"]): row for row in table.to_pylist()}


def test_normalized_score_writes_all_three_readings_of_one_score(tmp_path: Path) -> None:
    output, response = _run(tmp_path, [_pose("p1", -6.0), _pose("p2", -9.1), _pose("p3", -4.5)])

    metrics = _read(output, response, "derived_metrics")
    DERIVED_METRIC_V1.validate(metrics)
    rows = _by_metric(metrics)
    assert metrics.num_rows == 9
    assert {metric_id for _, metric_id in rows} == {
        "ligand_efficiency",
        "score_per_hac_pow",
        "score_baseline_residual",
    }

    # Benzene at -6.0 kcal/mol over 6 heavy atoms is exactly 1.0 kcal/mol per
    # heavy atom, which is the whole point of the metric being reproducible.
    assert rows[("p1", "ligand_efficiency")]["value"] == pytest.approx(1.0)
    assert rows[("p2", "ligand_efficiency")]["value"] == pytest.approx(9.1 / 13)
    assert rows[("p3", "ligand_efficiency")]["value"] == pytest.approx(4.5 / 3)
    assert rows[("p1", "score_per_hac_pow")]["value"] == pytest.approx(6.0 / 36)
    assert rows[("p2", "score_per_hac_pow")]["value"] == pytest.approx(9.1 / 169)
    assert rows[("p1", "score_baseline_residual")]["value"] == pytest.approx(
        (-7.66 + -0.0427 * 6) - -6.0
    )

    assert {row["units"] for row in metrics.to_pylist()} == {
        "KCAL_PER_MOL_PER_HEAVY_ATOM",
        "KCAL_PER_MOL_PER_HEAVY_ATOM_POW",
        "KCAL_PER_MOL",
    }
    assert {row["direction"] for row in metrics.to_pylist()} == {"HIGHER_BETTER"}
    assert {row["status"] for row in metrics.to_pylist()} == {"OK"}

    parents = _read(output, response, "primary")
    PARENT_V1.validate(parents)
    assert [row["parent_id"] for row in parents.to_pylist()] == ["p1", "p2", "p3"]
    assert response.metadata["scored_count"] == 3  # type: ignore[attr-defined]
    assert response.metadata["not_applicable_count"] == 0  # type: ignore[attr-defined]
    assert response.metadata["engine_id"] == "unidock"  # type: ignore[attr-defined]
    assert response.metadata["pose_policy"] == "BEST_POSE"  # type: ignore[attr-defined]


def test_normalized_score_counts_heavy_atoms_the_way_rdkit_does(tmp_path: Path) -> None:
    """The divisor is computed here, not read from a properties tier.

    ``properties/v1`` is produced by a finalize step that runs after every tier,
    so a post-docking tier cannot bind to it.  Computing the count in-plugin is
    what makes this tier independent of whether the operator kept the property
    tier at all -- but it also means the count has to be checked against the
    library everyone else in the repo uses.
    """

    Chem = pytest.importorskip("rdkit.Chem")

    output, response = _run(tmp_path, [_pose("p1", -6.0), _pose("p2", -9.1), _pose("p3", -4.5)])
    rows = _by_metric(_read(output, response, "derived_metrics"))

    for parent_id, smiles in _SMILES.items():
        expected = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
        assert expected == _HEAVY_ATOMS[parent_id]
        recorded = json.loads(str(rows[(parent_id, "ligand_efficiency")]["source_json"]))
        assert recorded["heavy_atom_count"] == expected


def test_normalized_score_reads_the_strongest_pose_not_the_first_one(tmp_path: Path) -> None:
    """The pose that decides a molecule is the strongest one.

    The docking gate accepts a molecule on ``MIN(score)`` grouped by parent, not
    on ``pose_rank == 0``.  A normalized score taken from a different pose would
    describe a conformer the run never accepted, so the aggregation has to match
    even when the ranks disagree with the scores.
    """

    output, response = _run(
        tmp_path,
        [
            _pose("p1", -5.0, pose_rank=0),
            _pose("p1", -7.5, pose_rank=1),
            _pose("p2", -9.1),
            _pose("p3", -4.5),
        ],
    )
    rows = _by_metric(_read(output, response, "derived_metrics"))

    assert rows[("p1", "ligand_efficiency")]["value"] == pytest.approx(7.5 / 6)
    assert json.loads(str(rows[("p1", "ligand_efficiency")]["source_json"]))["score"] == -7.5


def test_normalized_score_reports_a_missing_score_as_not_applicable(tmp_path: Path) -> None:
    """A parent with no pose gets a null, never a zero.

    Zero is a real ligand efficiency, and a strong one on a small molecule.
    Writing it for a molecule that was never scored would put the failures at
    the top of the ranking, which is the same trap the derived-metric gate's
    status check exists to close.
    """

    output, response = _run(tmp_path, [_pose("p1", -6.0), _pose("p2", -9.1)])

    metrics = _read(output, response, "derived_metrics")
    DERIVED_METRIC_V1.validate(metrics)
    rows = _by_metric(metrics)
    for metric_id in ("ligand_efficiency", "score_per_hac_pow", "score_baseline_residual"):
        row = rows[("p3", metric_id)]
        assert row["value"] is None
        assert row["status"] == "NOT_APPLICABLE"
        assert "unidock" in str(row["status_detail"])
        assert row["source_json"] is None
    assert rows[("p1", "ligand_efficiency")]["status"] == "OK"
    assert response.metadata["scored_count"] == 2  # type: ignore[attr-defined]
    assert response.metadata["not_applicable_count"] == 1  # type: ignore[attr-defined]
    # The population is unchanged: this tier is evidence, not a filter.
    assert response.metadata["output_count"] == 3  # type: ignore[attr-defined]


def test_normalized_score_refuses_a_scale_that_is_not_an_interaction_energy(
    tmp_path: Path,
) -> None:
    """Dividing KarmaDock's MDN score by heavy atoms is not ligand efficiency.

    It is a real number with no units, no published threshold, and no reason to
    be comparable across molecules of different sizes.  The hint names
    ``evidence_from`` because the usual cause is a criterion wired to the wrong
    docking stage rather than a mis-set option.
    """

    with pytest.raises(PluginError) as excinfo:
        _run(
            tmp_path,
            [
                _pose(
                    "p1",
                    8.4,
                    engine_id="karmadock",
                    score_kind="KARMADOCK_MDN",
                    direction="HIGHER_STRONGER",
                )
            ],
        )

    error = excinfo.value
    assert error.code == "NORMALIZED_SCORE_SCALE_UNSUPPORTED"
    assert error.context["observed_score_kinds"] == ["KARMADOCK_MDN"]
    assert error.context["engine_id"] == "karmadock"
    assert "evidence_from" in str(error.hint)


def test_normalized_score_refuses_a_different_kcal_scale_than_the_configured_one(
    tmp_path: Path,
) -> None:
    with pytest.raises(PluginError) as excinfo:
        _run(tmp_path, [_pose("p1", -6.0, score_kind="VINARDO_KCAL_MOL")])

    error = excinfo.value
    assert error.code == "NORMALIZED_SCORE_SEMANTICS_MISMATCH"
    assert error.context["expected_score_kind"] == "VINA_KCAL_MOL"
    assert error.context["observed_score_kinds"] == ["VINARDO_KCAL_MOL"]


def test_normalized_score_refuses_a_kcal_scale_read_backwards(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as excinfo:
        _run(tmp_path, [_pose("p1", -6.0, direction="HIGHER_STRONGER")])

    error = excinfo.value
    assert error.code == "NORMALIZED_SCORE_DIRECTION_UNSUPPORTED"
    assert error.context["observed_directions"] == ["HIGHER_STRONGER"]


def test_normalized_score_requires_one_engine_unless_it_is_pinned(tmp_path: Path) -> None:
    """Two docking engines in one tier is the default cascade, not an edge case.

    Each docking stage writes only its own rows, so binding this tier to the
    wrong one produces no overlap at all rather than slightly different numbers.
    Refusing to guess when two engines are visible is what turns that into a
    message about the binding.
    """

    poses = [
        _pose("p1", -6.0),
        _pose("p1", 8.4, engine_id="karmadock", score_kind="KARMADOCK_MDN"),
    ]

    with pytest.raises(PluginError) as excinfo:
        _run(tmp_path, poses, parent_ids=("p1",))

    error = excinfo.value
    assert error.code == "NORMALIZED_SCORE_IDENTITY_AMBIGUOUS"
    assert error.context["identity"] == "engine_id"
    assert error.context["observed_count"] == 2

    output, response = _run(
        tmp_path,
        poses,
        {"engine_id": "unidock"},
        parent_ids=("p1",),
        suffix="-pinned",
    )
    rows = _by_metric(_read(output, response, "derived_metrics"))
    assert rows[("p1", "ligand_efficiency")]["value"] == pytest.approx(1.0)


def test_normalized_score_says_so_when_the_pinned_engine_scored_nobody(tmp_path: Path) -> None:
    """The one failure that a per-parent status cannot express.

    A cascade wired to a docking stage that scored a different population comes
    out as a full table of nulls, which reads exactly like a docking stage that
    rejected everything.  Zero matches is a wiring error, so it stops the stage.
    """

    with pytest.raises(PluginError) as excinfo:
        _run(
            tmp_path,
            [_pose("p1", -6.0, engine_id="unidock-other")],
            {"engine_id": "unidock"},
            parent_ids=("p1",),
        )

    error = excinfo.value
    assert error.code == "NORMALIZED_SCORE_NO_MATCHING_EVIDENCE"
    assert error.context["engine_id"] == "unidock"
    assert error.context["parent_count"] == 1
    assert "evidence_from" in str(error.hint)


def test_normalized_score_gives_each_metric_its_own_parameter_digest(tmp_path: Path) -> None:
    """Changing one coefficient must rename exactly one metric.

    The three metrics share an engine, a receptor and a pose policy, but not
    their arithmetic.  Digesting the whole configuration into every
    ``method_id`` would make a change to ``hac_exponent`` look like a change to
    ligand efficiency, and a trace comparing two runs could not say which
    number actually moved.
    """

    poses = [_pose("p1", -6.0)]
    _, baseline = _run(tmp_path, poses, parent_ids=("p1",), suffix="-a")
    _, exponent = _run(
        tmp_path,
        poses,
        {"hac_exponent": 3.0},
        parent_ids=("p1",),
        suffix="-b",
    )
    _, slope = _run(
        tmp_path,
        poses,
        {"baseline_slope": -0.0569},
        parent_ids=("p1",),
        suffix="-c",
    )
    _, intercept = _run(
        tmp_path,
        poses,
        {"baseline_intercept": -8.0},
        parent_ids=("p1",),
        suffix="-d",
    )

    def methods(response: object) -> dict[str, str]:
        return dict(response.metadata["method_ids"])  # type: ignore[attr-defined]

    first = methods(baseline)
    assert set(first) == {
        "ligand_efficiency",
        "score_per_hac_pow",
        "score_baseline_residual",
    }
    assert all(method.startswith("normalized-score:sha256:") for method in first.values())
    assert len(set(first.values())) == 3

    for changed, moved in (
        (exponent, "score_per_hac_pow"),
        (slope, "score_baseline_residual"),
        (intercept, "score_baseline_residual"),
    ):
        after = methods(changed)
        assert after[moved] != first[moved]
        assert {name for name, method in after.items() if method != first[name]} == {moved}


def test_normalized_score_records_the_inputs_of_the_number_it_wrote(tmp_path: Path) -> None:
    output, response = _run(tmp_path, [_pose("p2", -9.1)], parent_ids=("p2",))
    rows = _by_metric(_read(output, response, "derived_metrics"))

    assert json.loads(str(rows[("p2", "ligand_efficiency")]["source_json"])) == {
        "heavy_atom_count": 13,
        "score": -9.1,
    }
    assert json.loads(str(rows[("p2", "score_per_hac_pow")]["source_json"])) == {
        "hac_exponent": 2.0,
        "heavy_atom_count": 13,
        "score": -9.1,
    }
    assert json.loads(str(rows[("p2", "score_baseline_residual")]["source_json"])) == {
        "baseline_intercept": -7.66,
        "baseline_slope": -0.0427,
        "heavy_atom_count": 13,
        "score": -9.1,
    }


def test_normalized_score_rejects_an_unparseable_parent(tmp_path: Path) -> None:
    output = tmp_path / "out"
    parent = _parent("p1")
    parent["parent_smiles"] = "not-a-molecule"
    with pytest.raises(PluginError) as excinfo:
        NormalizedDockingScorePlugin().execute(
            StageRequest(
                stage_id="normalized_score",
                inputs={
                    "parents": _dataset(tmp_path, "parents", PARENT_V1, [parent], artifact_hex="a"),
                    "evidence": _dataset(
                        tmp_path,
                        "poses",
                        DOCKING_SCORE_V1,
                        [_pose("p1", -6.0)],
                        artifact_hex="b",
                    ),
                },
                config={},
            ),
            StageContext(output),
        )

    assert excinfo.value.code == "NORMALIZED_SCORE_PARENT_INVALID"
    assert excinfo.value.context["parent_id"] == "p1"
    assert not (output / "datasets" / "derived_metrics" / "part-00000.parquet").exists()


def test_normalized_score_refuses_a_scale_it_has_no_definition_for_at_config_time(
    tmp_path: Path,
) -> None:
    with pytest.raises(PluginError) as excinfo:
        _run(tmp_path, [_pose("p1", -6.0)], {"expected_score_kind": "CNN_AFFINITY"})

    assert excinfo.value.code == "PLUGIN_CONFIG_INVALID"


def test_normalized_score_accepts_the_scale_spelled_the_way_a_builder_writes_it(
    tmp_path: Path,
) -> None:
    """Every exported cascade names the scale as a string, so the string must work.

    Strict validation does not coerce, which for an enum field means a bare
    ``str`` is refused before any validator of ours is consulted -- unless that
    validator returns the member.  A validator that only checks membership passes
    the string straight through and every JSON-shaped config is then rejected
    with the message written for an unsupported scale, pointing at the scale
    instead of at the type.  The dump round-trip is the same path a second time,
    which is what a sharded plugin does to its own settings inside each worker.
    """

    pinned = NormalizedDockingScoreConfig(expected_score_kind="VINARDO_KCAL_MOL")
    assert pinned.expected_score_kind is NormalizedScoreKind.VINARDO_KCAL_MOL
    assert NormalizedDockingScoreConfig.model_validate(pinned.model_dump(mode="json")) == pinned

    output, response = _run(
        tmp_path,
        [_pose("p1", -6.0, score_kind="VINARDO_KCAL_MOL")],
        {"expected_score_kind": "VINARDO_KCAL_MOL"},
        parent_ids=("p1",),
    )
    rows = _by_metric(_read(output, response, "derived_metrics"))
    assert rows[("p1", "ligand_efficiency")]["status"] == "OK"


def test_normalized_score_never_declares_the_field_that_injects_a_target() -> None:
    """The one field name that would make this tier score the wrong receptor.

    Lowering injects the campaign's target into any criterion whose config model
    declares a top-level ``receptor_path``, regardless of whether it is
    required.  This tier reads a receptor identity out of evidence instead, so
    the field must never appear -- and requiredness is not the guard, the name
    is.
    """

    assert "receptor_path" not in NormalizedDockingScoreConfig.model_fields


def test_normalized_score_vocabularies_track_the_contracts() -> None:
    """Both ends of this plugin spell their vocabulary as local literals.

    That is the house pattern -- no module imports another's enum -- which means
    a renamed contract member would leave this plugin comparing against a string
    nothing writes any more.  On the docking side that fails open: an unknown
    ``score_kind`` would simply never match, and every scale would be refused.
    """

    accepted = tuple(kind.value for kind in NormalizedScoreKind)
    assert set(accepted) <= set(DOCKING_SCORE_V1.enum_values["score_kind"])
    assert set(DOCKING_SCORE_V1.enum_values["score_kind"]) - set(accepted) == {
        "CNN_AFFINITY",
        "CNN_SCORE",
        "KARMADOCK_MDN",
    }
    assert "LOWER_STRONGER" in DOCKING_SCORE_V1.enum_values["direction"]

    units = {
        "KCAL_PER_MOL",
        "KCAL_PER_MOL_PER_HEAVY_ATOM",
        "KCAL_PER_MOL_PER_HEAVY_ATOM_POW",
    }
    assert units <= set(DERIVED_METRIC_V1.enum_values["units"])
    assert "HIGHER_BETTER" in DERIVED_METRIC_V1.enum_values["direction"]
    assert "NOT_APPLICABLE" in DERIVED_METRIC_V1.enum_values["status"]
