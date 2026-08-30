"""The docking cutoffs a generated cascade starts with, pinned to their evidence.

Unlike ``keep_poses``, these numbers are written down once: the gate plugin's
config model defaults both bounds to ``None`` and only insists that one of them
be given, so ``gate_defaults`` in the builder catalogue is the sole source of a
starting cutoff.  That makes them easy to change and impossible to notice --
nothing downstream fails when a threshold moves, the run simply keeps more or
fewer molecules, and a screen that quietly stopped filtering looks exactly like
a screen that had nothing to filter.

So the values live here with the measurement that chose them.  On a real
2,000-molecule library (``stk17b-subset-09``, 1,134 molecules reaching the
docking tier) the shipped -7.0 kept 88% of what Uni-Dock scored: six molecules
removed by a tier costing GPU hours, with KarmaDock's 50 doing the rest of the
work alone.  Moving to -9.5 / 45 brought the two engines to 151 / 107 survivors
and 41 after the consensus, and changed *which* molecules those were -- 16 held,
6 dropped, 25 new -- which is the difference between a two-engine tier and an
expensive way to ask one engine.

GNINA is deliberately not part of that pairing: CNNaffinity is a predicted
log-affinity where 6.0 is roughly one micromolar, a third scale unrelated to
either of the others, and no measurement here speaks to it.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.catalog import CRITERIA, BackendOption

#: Every catalogue option whose gate filters on a docking score, keyed by the
#: engine the gate names.  Built from the catalogue rather than listed, so a new
#: engine arriving without a pinned cutoff fails the first test below.
_GATES: dict[str, BackendOption] = {
    option.gate_defaults["engine_id"]: option
    for criterion in CRITERIA
    for option in criterion.options
    if option.gate_defaults.get("expected_score_kind")
    in {"VINA_KCAL_MOL", "VINARDO_KCAL_MOL", "AD4_KCAL_MOL", "KARMADOCK_MDN", "CNN_AFFINITY"}
}

#: Bound name and value, per engine.  Which bound matters is a property of the
#: scale's direction: a kcal/mol score is capped from above, a score where
#: higher is stronger is floored from below.
_EXPECTED = {
    "unidock": ("maximum", -9.5),
    "karmadock": ("minimum", 45.0),
    "gnina": ("minimum", 6.0),
}


def test_every_docking_gate_has_a_pinned_cutoff() -> None:
    """A new engine would otherwise ship an unreviewed threshold in silence."""

    assert set(_GATES) == set(_EXPECTED)


@pytest.mark.parametrize("engine_id", sorted(_EXPECTED))
def test_the_builder_writes_the_reviewed_cutoff(engine_id: str) -> None:
    bound, value = _EXPECTED[engine_id]

    assert _GATES[engine_id].gate_defaults[bound] == value


@pytest.mark.parametrize("engine_id", sorted(_EXPECTED))
def test_the_cutoff_is_reachable_from_the_builders_own_control(engine_id: str) -> None:
    """A default the slider cannot express is a default no operator can restore.

    The builder renders these as numeric fields with their own range and step.
    A cutoff outside that range, or off its step, is one an operator who nudged
    the control could never get back to.
    """

    bound, value = _EXPECTED[engine_id]
    field = next(
        candidate
        for candidate in _GATES[engine_id].thresholds
        if candidate.name == bound and candidate.target == "gate"
    )

    assert field.minimum is not None and field.maximum is not None
    assert field.minimum <= value <= field.maximum
    assert field.step is not None
    # Exact on the step grid, in the field's own units, without trusting binary
    # floats to divide cleanly.
    assert round((value - field.minimum) / field.step) == pytest.approx(
        (value - field.minimum) / field.step
    )
