"""Every name the builder offers must be a name its backend recognises.

A dropdown in the builder is a promise: the user picks "Oprea lead-like", the
run starts, and four hours later a shortlist appears.  If the string behind that
label is not a name medchem knows, one of two things happens -- the adapter
raises four hours in, or worse, some future adapter skips it and the run
completes having filtered on nothing at all.  Both failures are invisible at the
moment the choice is made, which is the only moment the user is looking.

The catalogue holds roughly sixty such strings across four backends and they are
typed by hand from documentation.  Nothing else in the suite compares them
against the installed package, so this file does: it walks the shipped
catalogue, pulls out every ``choice`` threshold, and resolves each value through
the same registry the adapter uses at run time.  Adding a preset without adding
its backend name is then a test failure rather than a support ticket.

Each backend is skipped where it is not installed, since these are optional
dependencies.  Skipping is safe: the names cannot be validated, but neither can
the option be selected.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.catalog import CRITERIA, BackendOption, ThresholdField


def _option(option_id: str) -> BackendOption:
    for criterion in CRITERIA:
        for option in criterion.options:
            if option.id == option_id:
                return option
    raise AssertionError(f"no catalogue option with id {option_id!r}")


def _threshold(option_id: str, field_name: str) -> ThresholdField:
    option = _option(option_id)
    for threshold in option.thresholds:
        if threshold.name == field_name:
            return threshold
    raise AssertionError(
        f"option {option_id!r} has no threshold {field_name!r}; "
        f"it has {[item.name for item in option.thresholds]}"
    )


def _choice_values(option_id: str, field_name: str) -> tuple[str, ...]:
    threshold = _threshold(option_id, field_name)
    assert threshold.kind == "choice", f"{option_id}.{field_name} is not a choice field"
    assert threshold.choices, f"{option_id}.{field_name} offers no choices"
    return tuple(value for value, _label in threshold.choices)


def _names(values: tuple[str, ...]) -> list[str]:
    """Split the comma-separated presets back into individual backend names."""

    return sorted({item.strip() for value in values for item in value.split(",") if item.strip()})


def _require(package: str) -> None:
    pytest.importorskip(package, reason=f"{package} is an optional backend")


# --------------------------------------------------------------------------
# medchem


@pytest.mark.parametrize(
    ("option_id", "field_name"),
    [
        ("medchem_rules", "rules"),
        ("medchem_admet_rules", "rules"),
    ],
)
def test_every_offered_medchem_rule_set_exists(option_id: str, field_name: str) -> None:
    _require("medchem")
    from medchem.rules import RuleFilters

    available = {str(name).casefold() for name in RuleFilters.list_available_rules_names()}
    offered = _names(_choice_values(option_id, field_name))
    missing = [name for name in offered if name.casefold() not in available]
    assert not missing, f"{option_id} offers rule sets medchem does not have: {missing}"


def test_every_offered_medchem_alert_collection_exists() -> None:
    _require("medchem")
    from medchem import structural

    table = structural.CommonAlertsFilters.list_default_available_alerts()
    available = {str(name).casefold() for name in table["rule_set_name"].tolist()}
    offered = _names(_choice_values("medchem_common_alerts", "alert_sets"))
    missing = [name for name in offered if name.casefold() not in available]
    assert not missing, f"medchem does not have these alert collections: {missing}"


def test_the_offered_medchem_rules_are_actually_constructible() -> None:
    """Resolving a name is not the same as the rule running.

    ``list_available_rules_names`` reports what the package advertises.  A rule
    whose implementation is missing, or whose bundled data failed to install,
    still appears in that list and only fails when it is asked to decide.
    """

    _require("medchem")
    import datamol as dm
    from medchem.rules import RuleFilters

    molecule = dm.to_mol("CC(=O)Oc1ccccc1C(=O)O")
    for name in _names(_choice_values("medchem_rules", "rules")):
        result = RuleFilters(rule_list=[name])(mols=[molecule], n_jobs=1)
        assert len(result) == 1, f"{name} returned no verdict"


# --------------------------------------------------------------------------
# rd_filters


def test_every_offered_rd_filters_set_is_one_upstream_ships() -> None:
    """These are declared twice -- once as a widget, once as adapter validation."""

    from molcascade.plugins.builtin.rd_filters_alerts import RULE_SETS

    option = _option("rd_filters")
    offered = {
        threshold.name.removesuffix("_action")
        for threshold in option.thresholds
        if threshold.name.endswith("_action")
    }
    assert offered, "the rd_filters option exposes no per-set control"
    expected = {name.lower() for name in RULE_SETS}
    assert offered == expected, (
        "the builder and the adapter disagree about which sets exist: "
        f"only in the builder {sorted(offered - expected)}, "
        f"only in the adapter {sorted(expected - offered)}"
    )


# --------------------------------------------------------------------------
# Mordred


def test_every_offered_mordred_descriptor_is_in_the_live_registry() -> None:
    """Mordred distinguishes naRing from nARing by one letter's case.

    A typo here does not raise at configuration time and does not raise at run
    time either if the misspelling happens to name a different real descriptor:
    it silently applies a window on aliphatic rings where the label says
    aromatic.  Comparing against the live registry is the only check that
    catches it.
    """

    _require("mordred")
    from mordred import Calculator
    from mordred import descriptors as mordred_descriptors

    available = {str(item) for item in Calculator(mordred_descriptors, ignore_3D=True).descriptors}
    offered = _choice_values("mordred_descriptor_window", "descriptor")
    missing = [name for name in offered if name not in available]
    assert not missing, f"Mordred has no such descriptors: {missing}"


def test_the_shipped_mordred_default_is_one_of_the_offered_choices() -> None:
    """A default outside the list leaves the widget showing someone else's value."""

    option = _option("mordred_descriptor_window")
    assert option.defaults["descriptor"] in _choice_values(
        "mordred_descriptor_window", "descriptor"
    )


# --------------------------------------------------------------------------
# Shape of the catalogue itself


def test_no_choice_field_offers_a_duplicate_or_empty_value() -> None:
    """Two entries with one value make the second unreachable in a select."""

    for criterion in CRITERIA:
        for option in criterion.options:
            for threshold in option.thresholds:
                if threshold.kind != "choice":
                    continue
                where = f"{criterion.id}/{option.id}/{threshold.name}"
                values = [value for value, _ in threshold.choices]
                labels = [label for _, label in threshold.choices]
                assert all(values), f"{where} has an empty value"
                assert all(labels), f"{where} has an unlabelled choice"
                assert len(set(values)) == len(values), f"{where} repeats a value"
                assert len(set(labels)) == len(labels), f"{where} repeats a label"


def test_every_criterion_has_a_runnable_backend() -> None:
    """A criterion with no executable option is a tier the user cannot fill.

    There is no longer one.  Route search was the first to be filled, through
    ``aizynthcli``; predicted affinity was the last, through the ``boltz`` CLI.
    Both live in environments of the user's own making, which is why "runnable"
    here means "an adapter is registered for it" and not "it will work on this
    machine without further setup".

    An id appearing in this list again would mean an adapter regressed, or that
    a catalogue entry was written as a signpost -- a description of a tool with
    nothing behind it.  Neither should reach the builder, where every block that
    can be dragged in is a block that can be run.
    """

    empty = sorted(criterion.id for criterion in CRITERIA if not criterion.executable_options)
    assert empty == []


def test_every_criterion_resolves_to_a_default_backend() -> None:
    """More than one option may be recommended -- a tier can run several tools.

    ``default_option`` is what the builder preselects when a block is dragged
    in, and it must exist wherever anything is runnable at all.
    """

    for criterion in CRITERIA:
        if not criterion.executable_options:
            continue
        default = criterion.default_option
        assert default is not None, f"{criterion.id} has no default"
        assert default.executable, f"{criterion.id} defaults to an option that cannot run"


def test_every_recorded_rate_declares_the_width_it_was_measured_at() -> None:
    """A rate is only comparable against another rate at the same width.

    The catalogue's rates are what decide where a block is placed, and a reader
    comparing ADMET-AI's 7/s against the ChEMBL checker's 100/s concludes the
    model is fourteen times slower. It is not: 7/s is a whole-machine figure
    across sixteen cores, so per lane it is nearer 0.44/s and the model is two
    hundred times slower. The field used to claim every figure was a one-core
    number while that one was not, and the prose had already tripped over it --
    the OpenADMET note warns its own figure is "not comparable with the 7/s
    recorded for ADMET-AI two options up".

    So the width is a field. This pins the one exception, because an exception
    nobody can see is the failure mode being fixed.
    """

    from molcascade.cascade.catalog import CRITERIA

    multi_lane = {
        (spec.id, option.id): option.throughput_lanes
        for spec in CRITERIA
        for option in spec.options
        if option.throughput_lanes != 1
    }
    assert multi_lane == {("admet_endpoint", "admet_ai_v2"): 16}

    for spec in CRITERIA:
        for option in spec.options:
            assert option.throughput_lanes >= 1, (spec.id, option.id)
            if option.throughput_per_second is None:
                # Unknown must stay unknown: a width without a rate would read
                # as a measurement nobody took.
                assert option.throughput_lanes == 1, (spec.id, option.id)


def test_the_width_travels_with_the_rate_into_the_builder_payload() -> None:
    """Whoever renders the rate must be able to see what it was measured on."""

    from molcascade.cascade.catalog import CRITERIA

    for spec in CRITERIA:
        for option, payload in zip(spec.options, spec.as_json()["options"], strict=True):
            assert payload["throughput_lanes"] == option.throughput_lanes
            assert payload["throughput_per_second"] == option.throughput_per_second


def test_every_runnable_option_can_actually_be_built_into_a_criterion() -> None:
    """The catalogue validates. Building a criterion from it is a different question.

    An option's ``defaults`` and ``gate_defaults`` are serialised into the cascade file,
    so every value in them has to be JSON. Nothing checked that. A tuple in
    ``gate_defaults`` passes every catalogue test in this directory, renders in the
    builder, appears in the bibliography -- and raises the first time a criterion is
    constructed from it, which is the first time anyone tries to use the entry. That is
    a long way from the typo.

    So this walks the whole catalogue and builds one criterion per runnable option,
    round-tripping it through JSON exactly as writing a cascade file would.
    """

    import json

    from molcascade.cascade.defaults import build_criterion
    from molcascade.plugins import create_builtin_registry

    registry = create_builtin_registry()
    built = 0
    for spec in CRITERIA:
        for option in spec.executable_options:
            if option.plugin_ref is None:
                continue
            try:
                registry.entry(option.plugin_ref)
            except Exception:  # not installed here; the option cannot be selected either
                continue
            criterion = build_criterion(spec, option, registry=registry)
            json.dumps(criterion.model_dump(mode="json"))
            built += 1
    assert built, "no option could be built, so this test proved nothing"
