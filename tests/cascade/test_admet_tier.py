"""The predicted-ADMET tier, and the two conditions under which it exists.

``t6_admet`` is the first default tier whose arms depend on what is installed on
the machine rather than on what is in the config, and that makes it the first
one that can get a fresh install wrong in two opposite directions.

Too eager, and the starter cascade names a model the host cannot run: lowering
succeeds, the preflight then refuses the run, and a user who asked for the
default gets an installation error instead of a screen.  That is the failure the
old comment in ``defaults.py`` was avoiding by shipping no ADMET tier at all, so
these tests pin the arms' absence at least as carefully as their presence.

Too shy, and the tier quietly becomes something stricter than it claims.  With
one arm ``any`` and ``all`` are the same rule, but the note the operator reads
is not the same sentence, and a two-arm tier that reported ``all`` would delete
a molecule either model disliked -- the opposite of the union this is meant to
be, on two models neither of which is strong enough to delete on alone.

The endpoint ids get their own test because nothing else would catch them.  Both
arms answer "is this a hERG liability", but one answers with a blocking
probability and the other with a pIC50; collapsing them into one endpoint would
put a 0-1 probability and a ~4-8 potency in the same column, where a threshold
in either unit is wrong for half the rows.  ``PREDICTION_V1``'s
``endpoint_separate`` invariant exists to forbid exactly that, and this is its
executable form at the point where the two are chosen.
"""

from __future__ import annotations

import pytest

from molcascade.cascade import defaults
from molcascade.cascade.catalog import STAGE_GROUPS, find_option
from molcascade.cascade.defaults import _admet_tier, default_cascade
from molcascade.cascade.models import CascadeConfig, TierConfig
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

_TIER_ID = "t6_admet"
_ABOVE = "t5_synthesis"
_BELOW = "t7_structure_qc"


@pytest.fixture(scope="module")
def registry() -> PluginRegistry:
    return create_builtin_registry()


def _pretend_admet_ai_is(present: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the version probe without installing or uninstalling anything.

    ``_admet_ai_arm`` looks the helper up as a module global, so patching it
    here reaches the call.  Patching this rather than ``importlib.metadata``
    keeps the test about the arm's condition instead of about how a version is
    read.
    """

    from molcascade.plugins.builtin.admet import PINNED_ADMET_AI_VERSION

    version = PINNED_ADMET_AI_VERSION if present else None
    monkeypatch.setattr(defaults, "_installed_version", lambda _dist: version)


def _pretend_openadmet_is(present: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(defaults, "_openadmet_paths_resolve", lambda _registry: present)


def _tier(cascade: CascadeConfig) -> TierConfig | None:
    return next((tier for tier in cascade.tiers if tier.id == _TIER_ID), None)


def test_with_neither_model_there_is_no_tier(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default that cannot run is worse than a default that is not offered."""

    _pretend_admet_ai_is(False, monkeypatch)
    _pretend_openadmet_is(False, monkeypatch)

    assert _admet_tier(registry) is None
    assert _tier(default_cascade(registry=registry)) is None


def test_a_bare_install_gets_exactly_the_cascade_it_got_before(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tier is a bonus, so its absence has to be indistinguishable from never.

    Asserting the whole id sequence rather than just ``t6_admet not in`` also
    catches the mistake of inserting the tier by renumbering the ones below it,
    which would silently invalidate every saved cascade and every run record
    that names a tier by id.
    """

    _pretend_admet_ai_is(False, monkeypatch)
    _pretend_openadmet_is(False, monkeypatch)

    order = [tier.id for tier in default_cascade(registry=registry).tiers]

    assert _TIER_ID not in order
    assert order.index(_ABOVE) + 1 == order.index(_BELOW)


@pytest.mark.parametrize("arm", ["admet_ai", "openadmet"])
def test_one_model_is_one_arm_and_reads_as_all(
    arm: str, registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a single arm the join is a formality, and ``all`` is the honest word."""

    _pretend_admet_ai_is(arm == "admet_ai", monkeypatch)
    _pretend_openadmet_is(arm == "openadmet", monkeypatch)

    tier = _admet_tier(registry)

    assert tier is not None
    assert len(tier.criteria) == 1
    assert tier.mode == "all"


def test_one_arm_says_how_to_get_the_second_opinion(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier deleting on one weak model should say what would make it two."""

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(False, monkeypatch)

    tier = _admet_tier(registry)

    assert tier is not None
    assert "bash envs/bootstrap.sh openadmet" in tier.note


def test_two_models_make_it_a_union(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete only where both object -- the same rule as the tier above it."""

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)

    assert tier is not None
    assert len(tier.criteria) == 2
    assert tier.mode == "any"


def test_the_tier_sits_where_docking_has_not_been_paid_for_yet(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above conformers and docking, which is the whole reason for the position.

    On the 103,515-molecule reference run, conformers plus two docking engines
    were 3,020 s of the 4,607 s spent reaching the affinity tier.  A molecule
    deleted here does not pay any of that; the same molecule deleted one tier
    lower pays all of it.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    order = [tier.id for tier in default_cascade(registry=registry).tiers]

    assert order.index(_ABOVE) < order.index(_TIER_ID) < order.index(_BELOW)
    assert order.index(_TIER_ID) < order.index("t9_docking")


def test_the_admet_ai_arm_disappears_off_the_pinned_release(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digests are a claim about one release, so the arm is too.

    The trust flag this arm passes is only defensible because the parent
    rehashes the installed package and model trees against digests measured on
    admet-ai 2.0.1.  On any other release those digests describe bytes that are
    not there, so the correct default is no arm rather than a flag whose
    justification has evaporated.
    """

    monkeypatch.setattr(defaults, "_installed_version", lambda _dist: "1.0.0")
    _pretend_openadmet_is(False, monkeypatch)

    assert _admet_tier(registry) is None


def test_the_trust_flag_is_earned_here_and_still_asked_for_in_the_builder(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default may assert what it verified; the catalogue may not assume it.

    These two halves have to be tested together, because each is only correct
    given the other: the tier passes the flag because it just checked the
    version the digests belong to, and the catalogue default stays ``False`` so
    that a hand-built cascade still makes the user say it themselves.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(False, monkeypatch)

    tier = _admet_tier(registry)
    assert tier is not None
    (arm,) = tier.criteria
    assert dict(arm.settings)["allow_unsafe_model_deserialization"] is True

    catalogue = find_option("admet_endpoint", "admet_ai_v2")
    assert catalogue is not None
    assert dict(catalogue.defaults).get("allow_unsafe_model_deserialization", False) is False


def test_the_openadmet_arm_needs_no_trust_flag(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its checkpoint is deserialized in its own interpreter, not in this one."""

    _pretend_admet_ai_is(False, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)
    assert tier is not None
    (arm,) = tier.criteria
    assert "allow_unsafe_model_deserialization" not in dict(arm.settings)


def test_the_two_arms_are_recorded_as_two_endpoints(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probability and a potency are not two readings of one number.

    ``PREDICTION_V1``'s primary key is ``(parent_id, endpoint_id, model_id)``,
    so equal endpoint ids would not collide -- they would sit in one column and
    be compared against one threshold, which is wrong in whichever unit the
    threshold was written.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)
    assert tier is not None
    endpoints = set()
    for criterion in tier.criteria:
        assert criterion.gate is not None, "an evidence producer only filters once gated"
        endpoints.add(dict(criterion.gate.settings)["endpoint_id"])

    assert endpoints == {"herg_blocking", "herg_pic50"}


def test_both_arms_cut_from_above(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rising probability and rising pIC50 both mean more hERG, so both are maxima.

    The units differ but the direction does not, and a tier that mixed
    directions would keep the molecules it meant to delete.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)
    assert tier is not None
    for criterion in tier.criteria:
        assert criterion.gate is not None
        gate = dict(criterion.gate.settings)
        assert "maximum" in gate and "minimum" not in gate


def test_the_joined_arm_cuts_lower_than_the_standalone_gate(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same model, two thresholds, and the difference is load-bearing.

    The catalogue's standalone gate for ``admet_ai_v2`` cuts at 0.95 because a
    lone filter at 0.90 deletes 13 of the 77 approved oral drugs -- one in six.
    This tier runs the same model at 0.90, because ``any`` deletes only what
    both arms object to and the other arm objects to three drugs in the whole
    panel, so 3/77 is the ceiling however far this arm loosens.  What the looser
    cut buys is dofetilide, at 0.919 the one withdrawn QT drug a 0.95 cut lets
    through, and whose mechanism of action *is* hERG blockade.

    Both numbers are measured, and the two are easy to mistake for a
    disagreement worth tidying into one.  This is the test that says they are
    not: unify them in either direction and it fails, in the direction that
    names which cost was paid.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)
    assert tier is not None
    arm = next(c for c in tier.criteria if c.id == "admet_herg_ai")
    assert arm.gate is not None
    assert dict(arm.gate.settings)["maximum"] == pytest.approx(0.90)

    option = find_option("admet_endpoint", "admet_ai_v2")
    assert option is not None
    standalone = dict(option.gate_defaults)["maximum"]
    assert standalone == pytest.approx(0.95), (
        "the standalone gate is a different decision on a different cost curve"
    )
    assert dict(arm.gate.settings)["maximum"] < standalone


def test_the_note_refuses_to_call_this_a_safety_assessment(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approved drugs bind hERG; the margin that makes them safe is not visible here.

    Verapamil, diltiazem and imatinib are all hERG binders and all approved, on
    an affinity-to-exposure ratio this stage has no way to compute.  The note is
    where that limit is stated, so it is the note that is pinned.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)

    assert tier is not None
    assert "not a cardiac safety assessment" in tier.note


def test_the_builder_group_joins_the_same_way_the_default_does(
    registry: PluginRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a cascade drawn in the builder is stricter than the shipped one.

    The group's mode is what the builder pre-selects when a user adds this
    group's options by hand, and "all" there would mean either model's objection
    deletes -- a different and much blunter rule than the one measured.
    """

    _pretend_admet_ai_is(True, monkeypatch)
    _pretend_openadmet_is(True, monkeypatch)

    tier = _admet_tier(registry)
    group = next(group for group in STAGE_GROUPS if group.id == "admet")

    assert tier is not None
    assert group.default_mode == tier.mode == "any"
