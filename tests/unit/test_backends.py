from __future__ import annotations

from molcascade.backends import (
    Availability,
    BackendInterface,
    BackendSpec,
    BackendTier,
    Capability,
    LicenseClass,
    ProbePolicy,
    create_backend_registry,
)
from molcascade.backends.registry import BackendRegistry, probe_backend


def test_catalog_has_multiple_local_options_for_every_screening_capability() -> None:
    registry = create_backend_registry()
    required = {
        Capability.SOURCE,
        Capability.STANDARDIZE,
        Capability.GATE,
        Capability.DESCRIPTORS,
        Capability.PREDICT,
        Capability.APPLICABILITY,
        Capability.SCAFFOLD,
        Capability.CLUSTER,
        Capability.SELECT,
        Capability.EXPORT,
    }
    for capability in required:
        assert len(registry.specs(capability)) >= 2, capability


def test_probe_detects_available_core_without_importing_backend() -> None:
    registry = create_backend_registry()
    status = probe_backend(registry.get("rdkit.standardize"))
    assert status.availability is Availability.AVAILABLE
    assert status.version


def test_license_policy_blocks_copyleft_even_when_module_is_installed() -> None:
    registry = create_backend_registry()
    status = probe_backend(registry.get("openbabel.rescue"))
    assert status.availability is Availability.BLOCKED
    allowed = probe_backend(
        registry.get("openbabel.rescue"),
        ProbePolicy(allow_copyleft=True),
    )
    assert allowed.availability in {Availability.AVAILABLE, Availability.DEGRADED}


def test_probe_never_installs_missing_module() -> None:
    spec = BackendSpec(
        id="test.missing",
        capability=Capability.PREDICT,
        display_name="Missing test backend",
        tier=BackendTier.OPTIONAL,
        interface=BackendInterface.PYTHON,
        license_spdx="MIT",
        license_class=LicenseClass.PERMISSIVE,
        module="molcascade_module_that_cannot_exist",
    )
    assert probe_backend(spec).availability is Availability.UNAVAILABLE


def test_registry_rejects_duplicate_ids() -> None:
    spec = create_backend_registry().get("python.delimited")
    try:
        BackendRegistry((spec, spec))
    except ValueError as error:
        assert "already registered" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate backend was accepted")
