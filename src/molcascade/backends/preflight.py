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

import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from molcascade.backends.catalog import BUILTIN_BACKEND_SPECS
from molcascade.backends.models import (
    Availability,
    BackendSpec,
    BackendStatus,
    BackendTier,
    LicenseClass,
    ProbePolicy,
)
from molcascade.backends.registry import probe_backend
from molcascade.errors import ConfigError, PluginError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from molcascade.config.models import StageConfig
    from molcascade.pipeline.compiler import CompiledStage
    from molcascade.plugins.registry import PluginRegistry


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


def copyleft_backends(
    stages: Sequence[CompiledStage],
) -> tuple[BackendSpec, ...]:
    """The copyleft tools this pipeline will execute, deduplicated, in order.

    Answering "which ones, exactly" rather than "are there any" is the whole
    point.  A run permitted to link copyleft code has to be able to name what it
    linked -- in the audit log while it happens, and in a methods section
    afterwards -- and "the operator passed --allow-copyleft" is not that.
    """

    found: list[BackendSpec] = []
    for _stage_id, spec in required_backends(stages):
        if spec.license_class is not LicenseClass.COPYLEFT:
            continue
        if all(known.id != spec.id for known in found):
            found.append(spec)
    return tuple(found)


def _isolated_status(
    spec: BackendSpec,
    config: Mapping[str, object],
) -> BackendStatus:
    """Check an isolated backend where it actually lives: the stage's own config.

    ``shutil.which`` is the right probe for every other CLI backend and the
    wrong one here.  An isolated backend is isolated *because* its dependency
    set conflicts with this environment's, so its ``bin`` directory is not on
    this process's ``PATH`` -- and if it were, a PATH lookup would be finding
    the very installation that must not be imported.  What decides whether the
    run can proceed is the absolute path the stage was configured with, so that
    is what gets stat'ed, and ``executable_config_key`` says which key holds it
    without this module having to know one backend's vocabulary.
    """

    key = spec.executable_config_key
    assert key is not None  # guarded by the caller
    evidence: dict[str, JsonValue] = {
        "config_key": key,
        "tier": str(spec.tier),
    }
    configured = config.get(key)
    if not isinstance(configured, str) or not configured.strip():
        return BackendStatus(
            spec=spec,
            availability=Availability.UNAVAILABLE,
            reason=f"the stage does not say where to find {spec.command[0]} ('{key}' is unset)",
            evidence=evidence,
        )
    path = Path(configured).expanduser()
    evidence["executable"] = str(path)
    if not path.is_absolute():
        return BackendStatus(
            spec=spec,
            availability=Availability.UNAVAILABLE,
            reason=f"'{key}' must be an absolute path, not a command name to resolve on PATH",
            evidence=evidence,
        )
    try:
        info = path.stat()
    except OSError:
        return BackendStatus(
            spec=spec,
            availability=Availability.UNAVAILABLE,
            reason=f"nothing exists at the configured '{key}': {path}",
            evidence=evidence,
        )
    if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        return BackendStatus(
            spec=spec,
            availability=Availability.UNAVAILABLE,
            reason=f"the configured '{key}' is not an executable file: {path}",
            evidence=evidence,
        )
    return BackendStatus(
        spec=spec,
        availability=Availability.AVAILABLE,
        reason="isolated executable found at the configured path",
        evidence=evidence,
    )


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
    blocked: list[BackendSpec] = []
    configs = {stage.stage_id: stage.config for stage in stages}
    for stage_id, spec in required_backends(stages):
        status = probe_backend(spec, selected_policy)
        if (
            spec.tier is BackendTier.ISOLATED
            and spec.executable_config_key is not None
            and status.availability is not Availability.BLOCKED
        ):
            # A licence or platform refusal stands whatever the stage says; only
            # the "is it installed" half of the verdict moves off PATH.
            status = _isolated_status(spec, configs.get(stage_id, {}))
        if status.availability in (Availability.AVAILABLE, Availability.DEGRADED):
            if spec.id not in ready:
                ready.append(spec.id)
            continue
        problems.append(f"  {stage_id} ({spec.display_name}): {status.reason}")
        if all(known.id != spec.id for known in missing):
            missing.append(spec)
            if status.availability is Availability.BLOCKED:
                blocked.append(spec)

    if problems:
        detail = "\n".join(problems)
        hint = (
            "MolCascade never installs anything during a screening run, so "
            "install these first, then start the run again. Removing the block "
            "from the cascade is the other way out."
        )
        # A licence refusal is not a missing package, and telling someone to pip
        # install their way out of it sends them to fix something that is not
        # broken.  These are separated out of both instructions below.
        refused = {spec.id for spec in blocked}
        installable = [spec for spec in missing if spec.id not in refused]
        command = _install_command(installable)
        if command:
            hint = f"{command}; {hint}"
        if not installable:
            hint = "Removing the block from the cascade is the other way out."
        if blocked:
            names = ", ".join(
                f"{spec.display_name} ({spec.license_spdx})" for spec in blocked
            )
            hint = (
                "Refused by this installation's licence policy, not missing from "
                f"it: {names}. Pass '--allow-copyleft' if your project accepts "
                "those terms -- the run records that it was given, so a methods "
                f"section can say so. {hint}"
            )
        isolated = [spec for spec in installable if spec.tier is BackendTier.ISOLATED]
        if isolated:
            # These are not pip installs into this environment -- installing one
            # here is the thing that would break the run -- so the instruction is
            # different in kind, not just in wording.
            names = ", ".join(spec.display_name for spec in isolated)
            hint = (
                f"{names} runs in an environment of its own because its dependencies "
                "conflict with this one's. Create that environment separately, then "
                "give the stage the absolute path to the executable inside it. "
                f"{hint}"
            )
        raise PluginError(
            f"{len(problems)} stage(s) need a backend that is not usable on "
            f"this machine:\n{detail}",
            code="BACKEND_PREFLIGHT_FAILED",
            hint=hint,
            context={"backends": [spec.id for spec in missing]},
        )
    return tuple(ready)


#: Every field a docking engine can learn from the cascade's target block, and
#: the run-time flag that supplies it.  A stage missing one of these has to be
#: told which flag fixes it; "receptor_pdbqt_path is required" is a sentence
#: about a Pydantic model, not about anything the operator can type.
TARGET_SETTING_FLAGS: Mapping[str, str] = MappingProxyType(
    {
        "receptor_path": "--receptor",
        "receptor_pdbqt_path": "--receptor-pdbqt",
        "reference_ligand_path": "--reference-ligand",
        "pocket_pdb_path": "--pocket",
        "center_x": "--box",
        "center_y": "--box",
        "center_z": "--box",
        "size_x": "--box",
        "size_y": "--box",
        "size_z": "--box",
    }
)

_SITE_FORMS = (
    "--reference-ligand LIGAND.sdf (a ligand bound in the site; its extent "
    "becomes the box), --pocket POCKET.pdb (the residues lining it), or "
    "--box cx,cy,cz,sx,sy,sz (the six numbers directly)"
)


def _docking_config_model(plugin: object) -> type[Any] | None:
    """The config model of a plugin that docks, or ``None`` for everything else.

    A plugin that declares somewhere to find a receptor is a plugin that docks
    against one.  That is the same marker lowering uses to decide where to write
    the target, so the two cannot drift into disagreeing about which stages the
    requirement applies to.
    """

    model = getattr(plugin, "config_model", None)
    fields = getattr(model, "model_fields", None)
    if not isinstance(fields, Mapping) or "receptor_path" not in fields:
        return None
    return cast("type[Any]", model)


def preflight_docking_target(
    stages: Sequence[StageConfig],
    *,
    registry: PluginRegistry,
) -> tuple[str, ...]:
    """Refuse to start a docking run that has not been told what to dock into.

    This runs on the *lowered* stage list rather than the compiled one, and
    deliberately so.  Compilation validates each stage against its plugin's
    config model, so a docking stage with no receptor already fails -- with
    ``Field required [type=missing, input_value=...]``, three tiers of Pydantic
    context, and no mention of the flag that would fix it.  Getting in front of
    that is the entire point.

    A flat pipeline that spells the receptor into its stage settings by hand
    passes untouched: the check is that the stage *has* what it needs, not that
    a cascade-level target supplied it.

    Returns the ids of the docking stages it checked, so a caller can say what
    it confirmed rather than only what it rejected.
    """

    # Imported here rather than at module scope so that checking packages stays
    # a cheap thing to do.  This module is loaded by 'doctor' and by the
    # builder's availability report, neither of which should pay for the
    # chemistry stack; the docking helpers are only needed once a cascade
    # actually contains an engine, by which point the caller holds a registry
    # and the plugin tree is loaded anyway.
    from molcascade.plugins.builtin.docking.common import (
        require_pdb_receptor,
        structure_digest,
    )

    checked: list[str] = []
    for stage in stages:
        try:
            plugin = registry.entry(stage.plugin).plugin
        except Exception:
            # An unregistered plugin is the compiler's problem to report, and
            # guessing at its requirements here would turn a preflight into a
            # reason a working run refuses to start.
            continue
        model = _docking_config_model(plugin)
        if model is None:
            continue
        checked.append(stage.id)
        missing = [
            name
            for name, field_info in model.model_fields.items()
            if name in TARGET_SETTING_FLAGS
            and field_info.is_required()
            and name not in stage.config
        ]
        if "receptor_path" in missing:
            raise ConfigError(
                f"stage {stage.id!r} docks against a protein and this run was not given one",
                code="DOCKING_TARGET_RECEPTOR_REQUIRED",
                hint=(
                    "Pass '--receptor RECEPTOR.pdb' together with one binding-site "
                    f"definition: {_SITE_FORMS}."
                ),
                context={"stage": stage.id, "plugin": stage.plugin},
            )
        if missing:
            flags = sorted({TARGET_SETTING_FLAGS[name] for name in missing})
            raise ConfigError(
                f"stage {stage.id!r} has a receptor but no binding site it can use",
                code="DOCKING_TARGET_SITE_REQUIRED",
                hint=(
                    f"This engine needs {', '.join(flags)}. A binding site is named "
                    f"exactly once, as one of: {_SITE_FORMS}."
                ),
                context={
                    "stage": stage.id,
                    "plugin": stage.plugin,
                    "missing_settings": missing,
                    "missing_flags": flags,
                },
            )

        receptor = str(stage.config["receptor_path"])
        require_pdb_receptor(
            receptor,
            engine=stage.plugin,
            hint=(
                "Convert the structure to PDB before screening. MolCascade reads "
                "and hashes the receptor but never edits it, so the conversion is "
                "yours to check."
            ),
        )
        _, digest = structure_digest(
            receptor,
            code="DOCKING_RECEPTOR_UNREADABLE",
            hint=(
                "This is the receptor for the docking tier, read once here so a "
                "run cannot discover it is unreadable three tiers deep."
            ),
        )
        pinned = stage.config.get("receptor_sha256")
        if pinned is not None and pinned != digest:
            raise ConfigError(
                f"stage {stage.id!r} pins a receptor digest that these bytes do not match",
                code="DOCKING_RECEPTOR_DIGEST_MISMATCH",
                hint=(
                    "Every docking score records the receptor it was computed "
                    "against. Either restore the structure this cascade was written "
                    "for, or update 'target.receptor_sha256' and accept that the new "
                    "scores are not comparable with the old ones."
                ),
                context={
                    "stage": stage.id,
                    "receptor_path": receptor,
                    "expected_sha256": pinned,
                    "actual_sha256": digest,
                },
            )
    return tuple(checked)


__all__ = [
    "TARGET_SETTING_FLAGS",
    "copyleft_backends",
    "preflight_backends",
    "preflight_docking_target",
    "required_backends",
]
