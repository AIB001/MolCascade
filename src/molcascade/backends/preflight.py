"""Check every Python package a run will import before the run reads a molecule.

This is the package half of :mod:`molcascade.assets.preflight`, and it exists
because the two halves behaved differently in a way no user could have
predicted.  A cascade whose synthesis tier wants SCScore weights fails
immediately, before any work happens, naming the ``molcascade assets fetch``
that fixes it.  The same cascade with a medchem tier and no medchem installed
ran chemistry, physicochemistry and drug-likeness over the whole library first
and only then raised ``MEDCHEM_BACKEND_UNAVAILABLE`` -- because
``import_medchem()`` is called inside ``execute()``, which is to say when
execution reaches the stage.  Measured on a three-molecule library that already
left a run directory and six cache entries behind; on the library sizes this
tool is built for it is hours of compute and a funnel that has to be discarded.

Nothing here imports the backend.  ``importlib.util.find_spec`` answers "would
this import succeed" without running module-level code, which matters because
some of these packages pull torch in on import and take seconds to do it.

The mapping needed no new bookkeeping.  A compiled stage names its plugin as
``id@version``; :attr:`BackendSpec.plugin_ref` is that same string; and the spec
already carries the ``module`` to probe and the ``distribution`` to install,
because :command:`molcascade doctor` has always used them.  So the chain is
stage -> plugin_ref -> spec -> module, and a cascade the project has never seen
is covered as long as its plugin is in the catalogue.  One that is not is left
alone: an unknown third-party plugin has unknown dependencies, and guessing at
them would turn a preflight into a reason a working run refuses to start.

Nothing here installs anything either.  MolCascade does not run pip; it reports
what is missing and the exact command a human can decide to run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from molcascade.backends.catalog import BUILTIN_BACKEND_SPECS
from molcascade.backends.models import Availability, BackendSpec, ProbePolicy
from molcascade.backends.registry import probe_backend
from molcascade.errors import PluginError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from molcascade.pipeline.compiler import CompiledStage


def _index_by_plugin_ref(
    specs: Sequence[BackendSpec],
) -> dict[str, BackendSpec]:
    """Map ``plugin_ref`` to the one spec that decides what to install for it.

    Several backends may share a plugin_ref -- ``rdkit.filtercatalog`` and
    ``rdkit.smarts`` are both ``chemistry.rdkit_hard_gate`` -- which is fine as
    long as they agree on the module, since then the requirement is the same
    whichever one the stage's settings end up selecting.  Where they would
    disagree the answer depends on settings this module deliberately does not
    interpret, so the ref is dropped and the adapter keeps its existing
    fail-closed behaviour.  No shipped ref disagrees; the check is here so that
    adding one that does degrades quietly instead of reporting the wrong
    package.
    """

    index: dict[str, BackendSpec] = {}
    ambiguous: set[str] = set()
    for spec in specs:
        ref = spec.plugin_ref
        if ref is None:
            continue
        seen = index.get(ref)
        if seen is None:
            index[ref] = spec
        elif seen.module != spec.module:
            ambiguous.add(ref)
    for ref in ambiguous:
        del index[ref]
    return index


_BY_PLUGIN_REF: dict[str, BackendSpec] = _index_by_plugin_ref(BUILTIN_BACKEND_SPECS)


def required_backends(
    stages: Sequence[CompiledStage],
) -> tuple[tuple[str, BackendSpec], ...]:
    """Return ``(stage_id, spec)`` for every stage backed by a catalogued tool.

    The stage id travels with the spec for the same reason it does in the asset
    preflight: "the run needs medchem" is a fact about the cascade, but "the
    t4_alerts_medchem stage needs medchem" tells the user which block to drop if
    they would rather not install it.

    Order follows execution order and duplicates are kept, so a caller can see
    that two tiers share one package.
    """

    found: list[tuple[str, BackendSpec]] = []
    for stage in stages:
        spec = _BY_PLUGIN_REF.get(stage.plugin_key)
        if spec is not None:
            found.append((stage.stage_id, spec))
    return tuple(found)


def _install_command(missing: Sequence[BackendSpec]) -> str | None:
    """The single pip command that would satisfy every missing package."""

    extras = sorted({spec.extra for spec in missing if spec.extra})
    loose = sorted(
        {
            spec.distribution
            for spec in missing
            if spec.distribution and not spec.extra
        }
    )
    parts: list[str] = []
    if extras:
        # Quoted because a bare bracket is a glob in most shells.  One command
        # with every extra, not one per package: provisioning in a
        # install-restart-install loop is the same delay this check removes.
        parts.append(f'"molcascade[{",".join(extras)}]"')
    parts.extend(loose)
    if not parts:
        return None
    return f"pip install {' '.join(parts)}"


def preflight_backends(
    stages: Sequence[CompiledStage],
    *,
    policy: ProbePolicy | None = None,
) -> tuple[str, ...]:
    """Verify every catalogued package the compiled ``stages`` will import.

    Returns the distinct backend ids that were checked and are usable, so a
    caller can report what it confirmed rather than only what it rejected.

    Raises :class:`~molcascade.errors.PluginError` naming *all* unusable
    backends at once.  ``DEGRADED`` is not a failure: it means the module
    imports but its distribution metadata could not be read, which is a
    reporting gap, not a missing package, and refusing to run over it would
    block conda-installed and vendored builds for no benefit.
    """

    # Probing a version command means running a subprocess, which a preflight
    # has no business doing on the way into a run.  Availability of the shipped
    # backends never depends on it: every registered adapter is a Python import.
    selected_policy = policy or ProbePolicy(run_version_commands=False)

    ready: list[str] = []
    problems: list[str] = []
    missing: list[BackendSpec] = []
    for stage_id, spec in required_backends(stages):
        status = probe_backend(spec, selected_policy)
        if status.availability in (Availability.AVAILABLE, Availability.DEGRADED):
            if spec.id not in ready:
                ready.append(spec.id)
            continue
        problems.append(f"  {stage_id} ({spec.display_name}): {status.reason}")
        if all(known.id != spec.id for known in missing):
            missing.append(spec)

    if problems:
        detail = "\n".join(problems)
        hint = (
            "MolCascade never installs anything during a screening run, so "
            "install these first, then start the run again. Removing the block "
            "from the cascade is the other way out."
        )
        command = _install_command(missing)
        if command:
            hint = f"{command}; {hint}"
        raise PluginError(
            f"{len(problems)} stage(s) need a backend that is not usable on "
            f"this machine:\n{detail}",
            code="BACKEND_PREFLIGHT_FAILED",
            hint=hint,
            context={"backends": [spec.id for spec in missing]},
        )
    return tuple(ready)


__all__ = ["preflight_backends", "required_backends"]
