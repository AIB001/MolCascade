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

from typing import Any

import pytest

from molcascade.cascade import defaults
from molcascade.cascade.defaults import (
    _affinity_tier,
    _criterion_if_available,
    _docking_metrics_tier,
    _docking_tier,
    default_cascade,
)
from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import CriterionConfig, TierConfig, TierMode
from molcascade.errors import ConfigError
from molcascade.plugins.builtin.boltz2 import Boltz2Config
from molcascade.plugins.builtin.specificity_panel import SpecificityPanelConfig
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

_METRIC_CRITERIA = ("pose_strain", "size_normalized_docking_score")
_SCORE = "docking_score/v1"


@pytest.fixture(scope="module")
def registry() -> PluginRegistry:
    return create_builtin_registry()


def _panel_criterion(registry: PluginRegistry) -> CriterionConfig:
    """One decoy pocket, given the way the builder gives it: flat keys."""

    criterion = _criterion_if_available(
        "specificity_panel",
        registry,
        settings={
            "structure_path": "/tmp/decoy.pdb",
            "reference_ligand_path": "/tmp/decoy-ligand.sdf",
        },
        evidence_from={_SCORE: "docking_score"},
    )
    if criterion is None:
        pytest.skip("the specificity panel backend is not installed")
    return criterion


def _with_panel(registry: PluginRegistry, criterion: CriterionConfig) -> Any:
    """Append the panel as its own tier below whatever the defaults produced."""

    cascade = default_cascade(registry=registry)
    tier = TierConfig(
        id="t11_specificity",
        title="Cross-pocket specificity",
        mode=TierMode.ALL,
        criteria=(criterion,),
    )
    return cascade.model_copy(update={"tiers": (*cascade.tiers, tier)})


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


# ------------------------------------------------- the panel, and why it is not here


def test_the_shipped_cascade_does_not_build_a_panel_nobody_supplied_receptors_for(
    registry: PluginRegistry,
) -> None:
    """There is no ``_specificity_tier``, and its absence is the design.

    A panel is a list of decoy pockets the operator chose; ``default_cascade``
    has no source for one and no defensible default -- picking pockets for
    someone would decide what "specific" means on their behalf.  A builder that
    could only ever return ``None`` would be dead code, so the placement rule
    lives in the catalogue entry instead, where the builder reads it.
    """

    assert not hasattr(defaults, "_specificity_tier")
    order = [tier.id for tier in default_cascade(registry=registry).tiers]
    assert not any(tier.startswith("t11") for tier in order)
    assert all("specificity" not in tier for tier in order)


def test_the_panel_cannot_be_handed_the_campaign_target(registry: PluginRegistry) -> None:
    """One field name is the whole switch, so the test names the field.

    ``_target_settings`` injects the campaign's receptor into any stage whose
    config model declares ``receptor_path``, and ``docking_requirements`` skips
    a stage on the same test.  A panel that declared it would silently gain the
    target as a decoy -- scoring it against itself, delta zero, every molecule
    perfectly unspecific.  The shorthand keys go through a ``before``
    validator, so they never become declared fields either.
    """

    fields = set(SpecificityPanelConfig.model_fields)
    assert "receptor_path" not in fields
    assert "structure_path" not in fields
    assert "reference_ligand_path" not in fields
    assert "panel" in fields


def test_a_panel_below_docking_binds_to_the_engine_it_names(registry: PluginRegistry) -> None:
    """Hand-built, because the shipped cascade cannot build this tier.

    The panel consumes the same contract with the same two producers as the
    metrics above it, so it needs the same pin -- and this is the assertion
    that the pin works for a stage that arrives from the builder rather than
    from ``defaults.py``.
    """

    criterion = _panel_criterion(registry)
    cascade = _with_panel(registry, criterion)
    plan = lower_cascade(
        cascade, registry=registry, allow_missing_library=True, allow_missing_target=True
    )
    stages = {stage.id: stage for stage in plan.pipeline.stages}
    bindings = {binding.request_port: binding for binding in stages["specificity_panel"].inputs}
    score = bindings["docking_score"]
    assert (score.stage, score.port) == ("docking_score", "scores")


def test_a_panel_beside_the_engine_it_reads_cannot_be_lowered(registry: PluginRegistry) -> None:
    """The panel is a third tier for the same reason the metrics are a second.

    It re-docks the pose Uni-Dock stored, so it is downstream of Uni-Dock; a
    parallel tier freezes its producer map before lowering its criteria, and
    "below" is the only way to express that.
    """

    criterion = _panel_criterion(registry)
    cascade = default_cascade(registry=registry)
    merged = tuple(
        tier.model_copy(update={"criteria": (*tier.criteria, criterion)})
        if tier.id == "t9_docking"
        else tier
        for tier in cascade.tiers
    )
    with pytest.raises(ConfigError) as excinfo:
        lower_cascade(
            cascade.model_copy(update={"tiers": merged}),
            registry=registry,
            allow_missing_library=True,
            allow_missing_target=True,
        )
    assert excinfo.value.code == "CASCADE_CRITERION_EVIDENCE_UNAVAILABLE"


# ------------------------------------------------- the last tier, which reads no pose


_FASTA = "/tmp/target.fasta"


def test_the_shipped_cascade_does_not_fold_against_a_sequence_nobody_gave_it(
    registry: PluginRegistry,
) -> None:
    """Unlike the panel this tier has a builder, because a path is enough.

    A decoy panel is a judgement call and so has no builder at all; a target
    sequence is one file, and the operator either has it or does not.  So
    ``_affinity_tier`` exists and returns ``None`` until asked, and the starter
    cascade is what it was before the tier was written.
    """

    assert _affinity_tier(registry, boltz2_settings=None) is None
    assert _affinity_tier(registry, boltz2_settings={}) is None
    order = [tier.id for tier in default_cascade(registry=registry).tiers]
    assert not any(tier.startswith("t12") for tier in order)


def test_a_blank_sequence_path_is_no_sequence_at_all(registry: PluginRegistry) -> None:
    """The catalogue default for the FASTA is ``""``, which is the trap.

    A builder that emitted its defaults unedited would hand over a blank path,
    and a blank that survived this far would fail inside the stage as an
    unreadable file named "" rather than as a target nobody chose.  Whitespace
    counts as blank for the same reason.
    """

    for blank in ("", "   ", "\t"):
        assert _affinity_tier(registry, boltz2_settings={"target_fasta_path": blank}) is None


def test_the_sequence_tier_goes_last_and_changes_nothing_above_it(
    registry: PluginRegistry,
) -> None:
    """Appended, not woven in: every other tier must come back identical."""

    plain = default_cascade(registry=registry)
    folded = default_cascade(registry=registry, boltz2_settings={"target_fasta_path": _FASTA})
    if len(folded.tiers) == len(plain.tiers):
        pytest.skip("the Boltz-2 backend is not installed")
    assert folded.tiers[-1].id == "t12_affinity"
    assert folded.tiers[:-1] == plain.tiers
    settings = folded.tiers[-1].criteria[0].settings
    assert settings["target_fasta_path"] == _FASTA
    assert settings["max_molecules"] > 0  # a budget, always, never unbounded


def test_the_sequence_tier_cannot_be_handed_the_campaign_receptor(
    registry: PluginRegistry,
) -> None:
    """Same one-field switch as the panel, and here it matters more.

    ``_target_settings`` injects the campaign receptor into any stage whose
    config declares ``receptor_path``.  This stage wants residues, not atoms,
    and a sequence read off ATOM records closes unresolved gaps in silence --
    so the field it must not declare is exactly the field that would let that
    happen without anyone typing a path.
    """

    fields = set(Boltz2Config.model_fields)
    assert "receptor_path" not in fields
    assert "target_fasta_path" in fields


def test_the_sequence_tier_does_not_need_a_docking_tier_above_it(
    registry: PluginRegistry,
) -> None:
    """The one post-docking tier that is not one.

    ``t10`` returns ``None`` without an engine above it because it re-expresses
    a docking score.  This tier folds the complex itself, reads ``parent/v1``
    and nothing else, and so lowers on a cascade with no docking at all.  It is
    last because of what it costs, not because of what it reads -- and a test
    that only ever saw it below Uni-Dock would not distinguish the two.
    """

    cascade = default_cascade(registry=registry, boltz2_settings={"target_fasta_path": _FASTA})
    if not any(tier.id == "t12_affinity" for tier in cascade.tiers):
        pytest.skip("the Boltz-2 backend is not installed")
    kept = tuple(tier for tier in cascade.tiers if tier.id in {"t1_chemistry", "t12_affinity"})
    lowered = lower_cascade(
        cascade.model_copy(update={"tiers": kept}),
        registry=registry,
        allow_missing_library=True,
        allow_missing_target=True,
    )
    assert any("boltz2" in stage.plugin for stage in lowered.pipeline.stages)
