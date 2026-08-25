"""What the builder is allowed to advertise as runnable.

The bug this guards against is quiet and expensive: an adapter registers, so
the builder offers it, so the user selects it and starts a screen -- and the
package or data file it needs is not on the machine, which is discovered on
the first batch of the tier, after everything cheaper has already been paid
for.  Every assertion here is about refusing to make that promise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.assets import AssetState
from molcascade.backends.catalog import distribution_for_module
from molcascade.cascade import availability as module
from molcascade.cascade.availability import (
    install_command,
    is_importable,
    missing_assets,
    missing_requirements,
    option_availability,
)
from molcascade.cascade.catalog import CRITERIA, BackendOption
from molcascade.plugins.registry import create_builtin_registry

#: A registered adapter with no dependencies of its own, used as the "the
#: registry is not the problem" control in the tests below.
_REAL_PLUGIN = "chemistry.rdkit_property_range_gate@0.1.0"
_ABSENT = "molcascade_definitely_not_installed"


@pytest.fixture(autouse=True)
def _clear_probe_caches() -> None:
    """Both probes are cached for the life of the process, deliberately.

    That is right in production -- a screen must not change its mind halfway
    through about what is installed -- and wrong across tests, which
    monkeypatch the thing being cached.
    """

    module.is_importable.cache_clear()
    module._asset_state.cache_clear()


def _option(**overrides: object) -> BackendOption:
    fields: dict[str, object] = {
        "id": "probe",
        "label": "Probe",
        "engine": "test",
        "summary": "A synthetic option used to exercise the availability probe.",
        "plugin_ref": _REAL_PLUGIN,
    }
    fields.update(overrides)
    return BackendOption(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------- the probes


def test_is_importable_answers_both_ways_without_importing() -> None:
    assert is_importable("json") is True
    assert is_importable(_ABSENT) is False


def test_a_broken_parent_package_reads_as_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``find_spec`` raises for a namespace parent that cannot be imported.

    A caller asking "will this tool work" needs the same answer either way, so
    the exception must not escape into the builder.
    """

    def explode(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib.util, "find_spec", explode)
    module.is_importable.cache_clear()
    assert is_importable("anything") is False


def test_missing_requirements_preserves_declaration_order() -> None:
    option = _option(requires=("json", _ABSENT, "os", _ABSENT + "_2"))
    assert missing_requirements(option) == (_ABSENT, _ABSENT + "_2")


# ------------------------------------------------------------ the decision


def test_an_option_with_everything_present_is_runnable() -> None:
    availability = option_availability(_option(requires=("json",)), create_builtin_registry())
    assert availability.runnable is True
    assert availability.missing_packages == ()
    assert availability.missing_assets == ()
    assert availability.reason is None


def test_a_research_candidate_is_never_runnable() -> None:
    availability = option_availability(_option(plugin_ref=None), create_builtin_registry())
    assert availability.runnable is False
    assert "no adapter" in (availability.reason or "")


def test_an_unregistered_adapter_is_not_runnable() -> None:
    option = _option(plugin_ref="nothing.at.all@9.9.9")
    assert option_availability(option, create_builtin_registry()).runnable is False


def test_a_registered_adapter_with_a_missing_package_is_not_runnable() -> None:
    """The defect that motivated this module, stated directly.

    ``executable`` used to mean "has a plugin_ref", which is true here.
    """

    option = _option(requires=(_ABSENT,))
    assert option.executable is True  # registered...
    availability = option_availability(option, create_builtin_registry())
    assert availability.runnable is False  # ...but not runnable
    assert availability.missing_packages == (_ABSENT,)
    assert _ABSENT in (availability.reason or "")


def test_the_reason_names_every_missing_package_readably() -> None:
    option = _option(requires=(_ABSENT, _ABSENT + "_2"))
    reason = option_availability(option, create_builtin_registry()).reason or ""
    assert f"{_ABSENT} and {_ABSENT}_2 are not installed" in reason


def test_a_missing_threshold_gate_is_not_runnable() -> None:
    option = _option(gate_plugin="nothing.at.all@9.9.9")
    assert option_availability(option, create_builtin_registry()).runnable is False


# ---------------------------------------------------------------- assets


def test_an_unfetched_asset_makes_an_installed_adapter_unrunnable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An asset is a dependency as real as a package, and fails just as late."""

    monkeypatch.setenv("MOLCASCADE_ASSET_ROOT", str(tmp_path))
    module._asset_state.cache_clear()
    option = _option(requires_assets=("rd_filters",))
    availability = option_availability(option, create_builtin_registry())
    assert availability.runnable is False
    assert availability.missing_assets == ("rd_filters",)
    assert "molcascade assets fetch rd_filters" in (availability.reason or "")
    # MolCascade must not offer to fix this by downloading mid-run.
    assert "never downloads" in (availability.reason or "")


def test_an_unknown_asset_id_reads_as_missing_rather_than_raising() -> None:
    """A stale declaration must degrade, not crash the builder."""

    assert missing_assets(_option(requires_assets=("no_such_asset",))) == ("no_such_asset",)


@pytest.mark.parametrize(
    "state", [AssetState.MISSING, AssetState.INCOMPLETE, AssetState.CORRUPT]
)
def test_an_unverified_asset_is_treated_as_absent(
    state: AssetState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest-verified or unusable; there is no third state worth offering."""

    monkeypatch.setattr(module, "_asset_state", lambda _id: state)
    assert missing_assets(_option(requires_assets=("rd_filters",))) == ("rd_filters",)


def test_a_verified_asset_clears_the_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_asset_state", lambda _id: AssetState.READY)
    assert missing_assets(_option(requires_assets=("rd_filters",))) == ()


def test_packages_are_reported_before_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """One actionable instruction at a time, and the cheaper one first.

    Telling someone to download 6 MB of weights for a tool they cannot import
    is wasted work; the package is what they hit first.
    """

    monkeypatch.setattr(module, "_asset_state", lambda _id: AssetState.MISSING)
    option = _option(requires=(_ABSENT,), requires_assets=("rd_filters",))
    availability = option_availability(option, create_builtin_registry())
    assert availability.missing_packages == (_ABSENT,)
    assert availability.missing_assets == ()


# ------------------------------------------------------------- the catalogue


def test_every_catalogue_option_declares_dependencies_it_actually_has() -> None:
    """A shipped option that is runnable here must really run here.

    This is the assertion that would have caught the original bug: on this
    machine ``admet_ai`` is absent, so an option claiming to be runnable while
    requiring it is a lie the builder would repeat to the user.
    """

    registry = create_builtin_registry()
    for spec in CRITERIA:
        for option in spec.options:
            availability = option_availability(option, registry)
            if not availability.runnable:
                continue
            assert option.plugin_ref in registry, option.id
            assert missing_requirements(option) == (), option.id
            assert missing_assets(option) == (), option.id


def test_an_unrunnable_option_always_explains_itself() -> None:
    """A greyed-out tool with no reason is a bug report waiting to happen."""

    registry = create_builtin_registry()
    for spec in CRITERIA:
        for option in spec.options:
            availability = option_availability(option, registry)
            if availability.runnable:
                continue
            assert availability.reason, option.id
            assert availability.reason.strip() == availability.reason, option.id


def test_options_needing_assets_point_at_a_real_asset_id() -> None:
    """A typo here would silently make an option permanently unselectable."""

    from molcascade.assets import iter_assets

    known = {spec.id for spec in iter_assets()}
    declared = {
        asset_id
        for spec in CRITERIA
        for option in spec.options
        for asset_id in option.requires_assets
    }
    assert declared <= known, sorted(declared - known)


def test_a_missing_package_is_reported_as_the_command_that_installs_it() -> None:
    """Naming the import is not enough when the two names disagree.

    ``import mordred`` is satisfied by the ``mordredcommunity`` distribution.
    ``pip install mordred`` also succeeds -- and installs the abandoned 2018
    original, which cannot compute a descriptor against a current RDKit.  A
    message that names only the import name therefore leads a user directly
    into a broken install that looks like the right package, which is worse
    than saying nothing.
    """

    assert install_command(("mordred",)) == "pip install mordredcommunity"


def test_one_wheel_supplying_several_imports_is_named_once() -> None:
    """ADMET-AI brings chemprop, lightning and torch. One install, not four."""

    command = install_command(("admet_ai", "chemprop", "lightning", "torch"))
    assert command.startswith("pip install admet-ai")
    assert command.count("admet-ai") == 1


def test_an_unmapped_import_falls_back_to_its_own_name() -> None:
    """The fallback has to be the guess the user would have made unaided."""

    assert install_command((_ABSENT,)) == f"pip install {_ABSENT}"


def test_the_reason_for_a_missing_package_carries_a_runnable_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    option = BackendOption(
        id="needs_mordred",
        label="Needs Mordred",
        engine="Mordred",
        summary="Control option for the install-hint path.",
        plugin_ref=_REAL_PLUGIN,
        requires=("mordred",),
    )
    monkeypatch.setattr(module, "is_importable", lambda name: False)
    availability = option_availability(option, create_builtin_registry())
    assert not availability.runnable
    assert availability.reason is not None
    assert "pip install mordredcommunity" in availability.reason


def test_every_package_a_runnable_option_needs_has_a_recorded_distribution() -> None:
    """The map is only useful if it covers what the builder actually offers.

    A new adapter whose package is absent from the backend catalogue would fall
    back to its import name silently, and the fallback is exactly the guess that
    was wrong for Mordred.  Torch and Lightning are excluded because nothing
    installs them on their own here -- they arrive inside another wheel.
    """

    arrives_with_another_wheel = {"lightning", "torch", "chemprop"}
    unmapped = sorted(
        {
            name
            for spec in CRITERIA
            for option in spec.options
            if option.executable
            for name in option.requires
            if name not in arrives_with_another_wheel
            and distribution_for_module(name) is None
        }
    )
    assert unmapped == [], unmapped
