"""The tier that reads the docking score back, and what it refuses to do.

``t10_docking_metrics`` is the first tier in the shipped cascade that consumes
another tier's evidence across a tier boundary, and the contract it consumes --
``docking_score/v1`` -- has two producers by default.  Three things therefore
have to hold at once, and none of them is visible in a run that succeeds:

*   the tier exists only when a kcal/mol engine is above it, because neither
    metric is defined on KarmaDock's MDN scale;
*   it sits *below* the docking tier, because criteria inside one tier are
    parallel branches that cannot see each other; and
*   it names Uni-Dock explicitly, because with two producers lowering refuses
    to guess -- and guessing would bind the metrics to KarmaDock, whose poses
    have already been relaxed through a force field.

A wrong binding here does not raise.  Each docking stage writes only its own
engine's rows, so a metric bound to the wrong one matches nothing, every
molecule comes back unscorable, and a gate that fails closed rejects the lot.
That reads as "the strain threshold is too strict", which is why these tests
assert the binding itself rather than the shape of the output.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.defaults import _docking_metrics_tier, _docking_tier, default_cascade
from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import CriterionConfig, TierConfig
from molcascade.errors import ConfigError
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

_METRIC_CRITERIA = ("pose_strain", "size_normalized_docking_score")
_SCORE = "docking_score/v1"


@pytest.fixture(scope="module")
def registry() -> PluginRegistry:
    return create_builtin_registry()


@pytest.fixture(scope="module")
def docking(registry: PluginRegistry) -> TierConfig:
    tier = _docking_tier(registry)
    if tier is None:
        pytest.skip("no docking engine is installed")
    return tier


@pytest.fixture(scope="module")
def metrics(registry: PluginRegistry, docking: TierConfig) -> TierConfig:
    tier = _docking_metrics_tier(registry, docking)
    if tier is None:
        pytest.skip("no post-docking metric is installed")
    return tier


def test_the_metrics_tier_does_not_exist_without_a_docking_tier_to_read(
    registry: PluginRegistry,
) -> None:
    """A user who deletes docking has not made an error, so this cannot raise.

    Lowering a metric with nobody producing ``docking_score/v1`` stops with
    CASCADE_CRITERION_EVIDENCE_UNAVAILABLE.  Building the tier anyway would turn
    "I am screening without a target" into a compile error about a tier the user
    never asked for.
    """

    assert _docking_metrics_tier(registry, None) is None


def test_the_metrics_tier_needs_a_score_in_kcal_per_mol_above_it(
    registry: PluginRegistry, docking: TierConfig
) -> None:
    """KarmaDock alone is not enough, and the refusal happens while building.

    Ligand efficiency is a score divided by heavy atoms; dividing an MDN score
    by heavy atoms gives a ratio with no units and no literature.  The
    normalized-score plugin refuses that scale outright at run time, so a tier
    built on a KarmaDock-only installation would be a cascade that compiles and
    then fails -- on a machine whose owner did nothing wrong.
    """

    karmadock_only = docking.model_copy(
        update={
            "criteria": tuple(
                criterion
                for criterion in docking.criteria
                if not criterion.backend.startswith("docking.unidock@")
            )
        }
    )
    assert karmadock_only.criteria, "fixture assumes the default tier has a second engine"
    assert _docking_metrics_tier(registry, karmadock_only) is None


def test_both_metrics_name_uni_dock_rather_than_inheriting_a_tier_order(
    metrics: TierConfig,
) -> None:
    """The choice of engine belongs in the exported plan, not in an ordering.

    ``evidence_from`` survives into the JSON a builder writes out, so a methods
    section can quote which engine's poses were measured.  Tier order would
    decide the same thing invisibly and would change under a drag.
    """

    assert metrics.id == "t10_docking_metrics"
    assert metrics.mode.value == "all"
    assert tuple(criterion.criterion for criterion in metrics.criteria) == _METRIC_CRITERIA
    for criterion in metrics.criteria:
        assert criterion.evidence_from == {_SCORE: "docking_score"}


def test_the_shipped_cascade_puts_the_metrics_below_docking(registry: PluginRegistry) -> None:
    order = [tier.id for tier in default_cascade(registry=registry).tiers]
    if "t10_docking_metrics" not in order:
        pytest.skip("no post-docking metric is installed")
    assert order.index("t10_docking_metrics") == order.index("t9_docking") + 1


def test_lowering_binds_both_metrics_to_uni_docks_scores_and_not_karma_docks(
    registry: PluginRegistry,
) -> None:
    """The one assertion that a wrong binding cannot survive.

    Both docking stages emit the same contract, so the binding is the only
    place the engine is named after the tier is lowered; the stage output
    itself carries no marker a test could read.
    """

    cascade = default_cascade(registry=registry)
    if not any(tier.id == "t10_docking_metrics" for tier in cascade.tiers):
        pytest.skip("no post-docking metric is installed")
    plan = lower_cascade(
        cascade, registry=registry, allow_missing_library=True, allow_missing_target=True
    )
    stages = {stage.id: stage for stage in plan.pipeline.stages}
    for stage_id in _METRIC_CRITERIA:
        bindings = {binding.request_port: binding for binding in stages[stage_id].inputs}
        score = bindings["docking_score"]
        assert (score.stage, score.port) == ("docking_score", "scores")


def test_dropping_the_pin_stops_the_cascade_rather_than_choosing_an_engine(
    registry: PluginRegistry,
) -> None:
    """Without ``evidence_from`` this is ambiguous, and ambiguity must be loud.

    Two producers and no pin used to mean "the last one wins", which is how a
    metric would end up on KarmaDock's relaxed poses without anyone choosing
    that.  The error names both candidates so the fix is a copy of a stage id.
    """

    cascade = default_cascade(registry=registry)
    if not any(tier.id == "t10_docking_metrics" for tier in cascade.tiers):
        pytest.skip("no post-docking metric is installed")
    unpinned = cascade.model_copy(
        update={
            "tiers": tuple(
                tier
                if tier.id != "t10_docking_metrics"
                else tier.model_copy(
                    update={
                        "criteria": tuple(
                            criterion.model_copy(update={"evidence_from": {}})
                            for criterion in tier.criteria
                        )
                    }
                )
                for tier in cascade.tiers
            )
        }
    )
    with pytest.raises(ConfigError) as excinfo:
        lower_cascade(
            unpinned, registry=registry, allow_missing_library=True, allow_missing_target=True
        )
    assert excinfo.value.code == "CASCADE_CRITERION_EVIDENCE_AMBIGUOUS"
    assert excinfo.value.context["candidate_stages"] == ["docking_score", "docking_score_2"]
    assert "evidence_from" in (excinfo.value.hint or "")


def test_a_metric_moved_up_beside_the_engine_it_reads_cannot_be_lowered(
    registry: PluginRegistry,
) -> None:
    """Why the tier has to be its own tier, expressed as the failure it avoids.

    A parallel tier freezes the producer map before its criteria are lowered so
    that no branch can filter the population another branch sees.  Side
    evidence follows the same rule, so a metric sitting beside Uni-Dock is not
    downstream of it -- and the diagnosis is a message about missing evidence,
    not about tier layout.
    """

    cascade = default_cascade(registry=registry)
    tiers = {tier.id: tier for tier in cascade.tiers}
    if "t10_docking_metrics" not in tiers:
        pytest.skip("no post-docking metric is installed")
    moved: CriterionConfig = tiers["t10_docking_metrics"].criteria[0]
    merged = tuple(
        tier.model_copy(update={"criteria": (*tier.criteria, moved)})
        if tier.id == "t9_docking"
        else tier
        for tier in cascade.tiers
        if tier.id != "t10_docking_metrics"
    )
    with pytest.raises(ConfigError) as excinfo:
        lower_cascade(
            cascade.model_copy(update={"tiers": merged}),
            registry=registry,
            allow_missing_library=True,
            allow_missing_target=True,
        )
    assert excinfo.value.code == "CASCADE_CRITERION_EVIDENCE_UNAVAILABLE"
