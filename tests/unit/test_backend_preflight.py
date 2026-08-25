"""The package half of the preflight: a missing backend costs nothing.

The behaviour under test is a timing property, not a message.  Every adapter
that needs an optional package already fails closed with an install hint, but it
did so from inside ``execute()`` -- so the failure arrived after the tiers above
it had processed the whole library, and after a run directory and a cache had
been written.  These tests pin the answer to "before any of that", which is the
same guarantee ``preflight_assets`` already gave for weights.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from molcascade.backends.catalog import BUILTIN_BACKEND_SPECS
from molcascade.backends.models import (
    BackendInterface,
    BackendSpec,
    BackendTier,
    Capability,
    LicenseClass,
)
from molcascade.backends.preflight import (
    _index_by_plugin_ref,
    preflight_backends,
    required_backends,
)
from molcascade.errors import PluginError
from molcascade.pipeline.compiler import CompiledStage
from molcascade.plugins import create_builtin_registry

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _normalize(requirement: str) -> str:
    """The PEP 503 form of the distribution name at the head of ``requirement``."""

    name = re.split(r"[<>=!~\[;\s]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _stage(stage_id: str, plugin_key: str) -> CompiledStage:
    """A compiled stage stripped to the two fields the preflight reads.

    Building one through the compiler would need a whole valid pipeline for
    each case; the preflight only ever looks at ``stage_id`` and ``plugin_key``,
    and a descriptor it never touches is not worth constructing.
    """

    descriptor = create_builtin_registry().entries()[0].descriptor
    return CompiledStage(
        stage_id=stage_id,
        slot="gate",
        plugin_key=plugin_key,
        descriptor=descriptor,
        config={},
        input_contract=None,
        output_contract="molcascade.decision.v1",
        input_port=None,
    )


def test_a_stage_whose_package_is_absent_is_named_before_the_run_starts() -> None:
    absent = BackendSpec(
        id="nonexistent.tool",
        capability=Capability.GATE,
        display_name="A tool that is not installed",
        tier=BackendTier.OPTIONAL,
        interface=BackendInterface.PYTHON,
        license_spdx="MIT",
        license_class=LicenseClass.PERMISSIVE,
        module="molcascade_no_such_module",
        distribution="molcascade-no-such-distribution",
        plugin_ref="chemistry.imaginary@0.1.0",
    )
    index = _index_by_plugin_ref((*BUILTIN_BACKEND_SPECS, absent))
    assert index["chemistry.imaginary@0.1.0"] is absent


def test_probe_reports_every_missing_backend_rather_than_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One command has to fix the whole cascade.

    Reporting one package at a time turns provisioning into an
    install-restart-install loop, which is the delay this check exists to
    remove -- and it is the loop the old per-adapter failure produced, since
    each restart got a tier further before hitting the next missing package.
    """

    missing = {
        "chemistry.a@0.1.0": _spec("a", "molcascade_absent_a", extra="alerts"),
        "chemistry.b@0.1.0": _spec("b", "molcascade_absent_b", extra="admet"),
    }
    monkeypatch.setattr(
        "molcascade.backends.preflight._BY_PLUGIN_REF", missing, raising=True
    )
    stages = (_stage("tier3_a", "chemistry.a@0.1.0"), _stage("tier5_b", "chemistry.b@0.1.0"))

    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)

    error = raised.value
    assert error.code == "BACKEND_PREFLIGHT_FAILED"
    assert "tier3_a" in str(error)
    assert "tier5_b" in str(error)
    # Both extras, one command, and quoted -- a bare bracket is a shell glob.
    assert error.hint is not None
    assert 'pip install "molcascade[admet,alerts]"' in error.hint


def _spec(suffix: str, module: str, *, extra: str | None = None) -> BackendSpec:
    return BackendSpec(
        id=f"absent.{suffix}",
        capability=Capability.GATE,
        display_name=f"Absent backend {suffix}",
        tier=BackendTier.OPTIONAL,
        interface=BackendInterface.PYTHON,
        license_spdx="MIT",
        license_class=LicenseClass.PERMISSIVE,
        module=module,
        distribution=f"absent-{suffix}",
        plugin_ref=f"chemistry.{suffix}@0.1.0",
        extra=extra,
    )


def test_an_uncatalogued_plugin_does_not_block_the_run() -> None:
    """A third-party plugin has dependencies this module cannot know.

    Failing closed on the unknown would mean a working custom cascade refuses to
    start, so the unknown is skipped and its adapter keeps whatever fail-closed
    behaviour it wrote for itself.
    """

    stages = (_stage("custom", "someones.own.plugin@9.9.9"),)
    assert required_backends(stages) == ()
    assert preflight_backends(stages) == ()


def test_installed_backends_are_reported_not_merely_tolerated() -> None:
    """RDKit is a core dependency, so this is the confirmed-ready path."""

    stages = (_stage("t1_validity", "chemistry.rdkit_hard_gate@0.1.0"),)
    assert preflight_backends(stages) == ("rdkit.filtercatalog",)


def test_one_plugin_ref_served_by_backends_needing_different_packages_is_skipped() -> None:
    """Which one runs would depend on settings this module does not interpret."""

    contested = (
        _spec("x", "module_one"),
        BackendSpec(
            id="absent.y",
            capability=Capability.GATE,
            display_name="Absent backend y",
            tier=BackendTier.OPTIONAL,
            interface=BackendInterface.PYTHON,
            license_spdx="MIT",
            license_class=LicenseClass.PERMISSIVE,
            module="module_two",
            distribution="absent-y",
            plugin_ref="chemistry.x@0.1.0",
        ),
    )
    assert _index_by_plugin_ref(contested) == {}


def test_every_declared_extra_exists_in_pyproject_and_installs_its_distribution() -> None:
    """The reason ``extra`` is declared on the spec instead of read at run time.

    ``importlib.metadata`` reports the extras of the installed dist-info, which
    in an editable checkout lags pyproject by however long since the last
    reinstall -- during this change it still advertised two extras that had been
    renamed away.  Declaring the pairing and testing it keeps the hint honest
    without depending on when the user last ran ``pip install -e .``.
    """

    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras: dict[str, list[str]] = project["optional-dependencies"]

    for spec in BUILTIN_BACKEND_SPECS:
        if spec.extra is None:
            continue
        assert spec.extra in extras, (
            f"{spec.id} names extra '{spec.extra}', which pyproject does not define"
        )
        assert spec.distribution is not None
        # PEP 503: '-', '_' and '.' are the same character in a distribution
        # name, so 'chembl-structure-pipeline' and 'chembl_structure_pipeline'
        # are one package spelled two ways, and both spellings are in the tree.
        installed = {_normalize(requirement) for requirement in extras[spec.extra]}
        assert _normalize(spec.distribution) in installed, (
            f"extra '{spec.extra}' does not install {spec.distribution}, "
            f"which {spec.id} imports as {spec.module}"
        )


def test_every_selectable_optional_backend_names_the_extra_that_installs_it() -> None:
    """An optional backend a cascade can select but nothing documents is a trap.

    ``plugin_ref`` set means the builder can put it in a cascade; no ``extra``
    then means the preflight can only offer a bare ``pip install <dist>``, which
    drops the version floor the adapter was written against.  Core dependencies
    are exempt because they cannot be missing, and copyleft backends because
    deployment policy blocks them before installability matters.
    """

    core = {"rdkit", "pyarrow", "openpyxl", "numpy", "pydantic", "PyYAML"}
    for spec in BUILTIN_BACKEND_SPECS:
        if spec.plugin_ref is None or spec.interface is not BackendInterface.PYTHON:
            continue
        if spec.distribution in core or spec.license_class is LicenseClass.COPYLEFT:
            continue
        assert spec.extra is not None, (
            f"{spec.id} is selectable in a cascade but names no extra; "
            f"add one to pyproject and declare it on the spec"
        )
