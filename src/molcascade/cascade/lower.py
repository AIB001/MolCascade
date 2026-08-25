"""Lower a tier-first cascade into the executable flat pipeline.

The cascade is what a user edits; the pipeline is what the compiler, cache,
artifact store, and runner already understand.  Lowering is deliberately total
and deterministic: the same cascade and the same registry always produce the
same stage list, in the same order, with the same generated stage identifiers,
so the pipeline revision identity stays stable across regenerations.

All wiring that the old node canvas asked users to draw by hand is derived here
from three facts: the tier order, the tier mode, and each plugin's declared
contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from molcascade.cascade.introspect import with_schema_version
from molcascade.cascade.library import resolve_library
from molcascade.cascade.models import (
    CascadeConfig,
    CriterionConfig,
    LibraryFormat,
    StepConfig,
    TierConfig,
    TierMode,
)
from molcascade.config.models import PipelineConfig
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import ConfigError
from molcascade.plugins.manifest import PluginDescriptor, PluginKind
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

DECISION_JOIN_PLUGIN = "policy.native_decision_join@0.1.0"

# The compiler checks that a stage's slot matches its plugin's kind.  Users of a
# cascade never type a slot, so the canonical name is derived here.
_KIND_SLOTS: Mapping[PluginKind, str] = MappingProxyType(
    {
        PluginKind.SOURCE: "source",
        PluginKind.STANDARDIZER: "standardize",
        PluginKind.GATE: "gate",
        PluginKind.FEATURIZER: "featurize",
        PluginKind.TRAINER: "train",
        PluginKind.PREDICTOR: "predict",
        PluginKind.APPLICABILITY: "applicability",
        PluginKind.SYNTHESIS: "synthesis",
        PluginKind.SCAFFOLDER: "scaffold",
        PluginKind.CLUSTERER: "cluster",
        PluginKind.SELECTOR: "select",
        PluginKind.ENUMERATOR: "enumerate",
        PluginKind.EXPORTER: "export",
    }
)


@dataclass(frozen=True, slots=True)
class StageOrigin:
    """Where a generated stage came from, for messages and the UI."""

    stage_id: str
    role: str
    tier_id: str | None = None
    criterion_id: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class LoweredCascade:
    """A cascade rendered as an executable pipeline plus its provenance map."""

    pipeline: PipelineConfig
    library_path: str
    origins: tuple[StageOrigin, ...] = field(default_factory=tuple)
    library_is_placeholder: bool = False

    def origin_of(self, stage_id: str) -> StageOrigin | None:
        return next((origin for origin in self.origins if origin.stage_id == stage_id), None)


def _entry(registry: PluginRegistry, backend: str, *, where: str) -> Any:
    try:
        return registry.entry(backend)
    except Exception as error:  # registry raises its own typed error
        raise ConfigError(
            f"{where} uses backend {backend!r} which is not a registered executable plugin",
            code="CASCADE_BACKEND_UNKNOWN",
            context={"backend": backend, "where": where},
        ) from error


def _descriptor(registry: PluginRegistry, backend: str, *, where: str) -> PluginDescriptor:
    descriptor = _entry(registry, backend, where=where).descriptor
    return cast(PluginDescriptor, descriptor)


def _slot_for(descriptor: PluginDescriptor) -> str:
    slot = _KIND_SLOTS.get(descriptor.kind)
    if slot is None:  # pragma: no cover - PluginKind is exhaustive above
        raise ConfigError(
            f"plugin {descriptor.key!r} has no cascade slot for kind {descriptor.kind.value!r}",
            code="CASCADE_SLOT_UNKNOWN",
            context={"backend": descriptor.key, "kind": descriptor.kind.value},
        )
    return slot


def _port_for_contract(descriptor: PluginDescriptor, contract_id: str) -> str | None:
    return next(
        (port for port, contract in descriptor.output_ports.items() if contract == contract_id),
        None,
    )


def _binding(request_port: str, stage: str, port: str = "primary") -> dict[str, str]:
    return {"request_port": request_port, "stage": stage, "port": port}


class _StageAccumulator:
    """Collect stages in execution order while tracking the current population."""

    def __init__(self) -> None:
        self.stages: list[dict[str, Any]] = []
        self.origins: list[StageOrigin] = []

    def add(
        self,
        *,
        stage_id: str,
        descriptor: PluginDescriptor,
        settings: Mapping[str, Any],
        inputs: list[dict[str, str]],
        origin: StageOrigin,
    ) -> None:
        self.stages.append(
            {
                "id": stage_id,
                "slot": _slot_for(descriptor),
                "plugin": descriptor.key,
                "inputs": inputs,
                "config": dict(settings),
                "enabled": True,
            }
        )
        self.origins.append(origin)


def _lower_criterion(
    criterion: CriterionConfig,
    *,
    tier: TierConfig,
    registry: PluginRegistry,
    accumulator: _StageAccumulator,
    parent_stage: str,
    parent_port: str,
) -> tuple[str, str, tuple[str, str] | None]:
    """Emit a criterion's producer and optional gate.

    Returns the stage/port carrying the surviving population after this
    criterion plus the stage/port carrying its ``decision/v1`` stream, if any.
    """

    where = f"criterion {criterion.id!r} in tier {tier.id!r}"
    producer = _descriptor(registry, criterion.backend, where=where)
    if PARENT_V1.id not in producer.inputs:
        raise ConfigError(
            f"{where} uses {criterion.backend!r}, which does not accept a parent population",
            code="CASCADE_CRITERION_NOT_APPLICABLE",
            context={"backend": criterion.backend, "where": where},
        )
    accumulator.add(
        stage_id=criterion.id,
        descriptor=producer,
        settings=criterion.settings,
        inputs=[_binding("primary", parent_stage, parent_port)],
        origin=StageOrigin(
            stage_id=criterion.id,
            role="criterion",
            tier_id=tier.id,
            criterion_id=criterion.id,
            label=criterion.label or producer.display_name,
        ),
    )
    survivor_port = _port_for_contract(producer, PARENT_V1.id)
    if survivor_port is None:
        raise ConfigError(
            f"{where} uses {criterion.backend!r}, which does not emit a parent population",
            code="CASCADE_CRITERION_NO_POPULATION",
            context={"backend": criterion.backend, "where": where},
        )

    if criterion.gate is None:
        decision_port = _port_for_contract(producer, DECISION_V1.id)
        decision = (criterion.id, decision_port) if decision_port else None
        return criterion.id, survivor_port, decision

    gate_where = f"threshold of {where}"
    gate = _descriptor(registry, criterion.gate.backend, where=gate_where)
    evidence_contracts = sorted(
        set(gate.inputs) & set(producer.outputs) - {PARENT_V1.id, DECISION_V1.id}
    )
    if len(evidence_contracts) != 1:
        raise ConfigError(
            f"{gate_where} cannot bind: {criterion.backend!r} and {criterion.gate.backend!r} "
            f"share {len(evidence_contracts)} evidence contracts, expected exactly one",
            code="CASCADE_GATE_EVIDENCE_AMBIGUOUS",
            context={
                "producer": criterion.backend,
                "gate": criterion.gate.backend,
                "shared_contracts": evidence_contracts,
            },
        )
    evidence_port = _port_for_contract(producer, evidence_contracts[0])
    gate_decision_port = _port_for_contract(gate, DECISION_V1.id)
    if gate_decision_port is None:
        raise ConfigError(
            f"{gate_where} uses {criterion.gate.backend!r}, which emits no decisions",
            code="CASCADE_GATE_NO_DECISIONS",
            context={"gate": criterion.gate.backend},
        )
    gate_survivor_port = _port_for_contract(gate, PARENT_V1.id)
    if gate_survivor_port is None:
        raise ConfigError(
            f"{gate_where} uses {criterion.gate.backend!r}, which emits no parent population",
            code="CASCADE_GATE_NO_POPULATION",
            context={"gate": criterion.gate.backend},
        )
    assert evidence_port is not None  # guaranteed by the shared-contract check above
    accumulator.add(
        stage_id=criterion.gate_stage_id,
        descriptor=gate,
        settings=criterion.gate.settings,
        inputs=[
            _binding("parents", criterion.id, survivor_port),
            _binding("evidence", criterion.id, evidence_port),
        ],
        origin=StageOrigin(
            stage_id=criterion.gate_stage_id,
            role="threshold",
            tier_id=tier.id,
            criterion_id=criterion.id,
            label=f"{criterion.label or producer.display_name} threshold",
        ),
    )
    return criterion.gate_stage_id, gate_survivor_port, (criterion.gate_stage_id, gate_decision_port)


def _lower_tier(
    tier: TierConfig,
    *,
    registry: PluginRegistry,
    accumulator: _StageAccumulator,
    anchor: tuple[str, str],
) -> tuple[str, str]:
    criteria = tier.active_criteria
    if not criteria:
        return anchor

    if tier.mode is TierMode.SERIAL:
        stage, port = anchor
        for criterion in criteria:
            stage, port, _ = _lower_criterion(
                criterion,
                tier=tier,
                registry=registry,
                accumulator=accumulator,
                parent_stage=stage,
                parent_port=port,
            )
        return stage, port

    # Parallel modes: every criterion reads the population entering the tier so
    # that no criterion can hide evidence from another by filtering first.
    decisions: list[tuple[str, str, str]] = []
    survivors: list[tuple[str, str]] = []
    for criterion in criteria:
        stage, port, decision = _lower_criterion(
            criterion,
            tier=tier,
            registry=registry,
            accumulator=accumulator,
            parent_stage=anchor[0],
            parent_port=anchor[1],
        )
        survivors.append((stage, port))
        if decision is None:
            raise ConfigError(
                f"criterion {criterion.id!r} in parallel tier {tier.id!r} produces no decision; "
                "give it a threshold or move it to a serial tier",
                code="CASCADE_PARALLEL_CRITERION_WITHOUT_DECISION",
                context={"tier": tier.id, "criterion": criterion.id},
            )
        decisions.append((criterion.id, decision[0], decision[1]))

    if len(decisions) == 1:
        # One criterion combined with ALL/ANY is that criterion.  Emitting a
        # single-input join would only add a stage without changing survivors.
        return survivors[0]

    join = _descriptor(registry, DECISION_JOIN_PLUGIN, where=f"tier {tier.id!r}")
    settings: dict[str, Any] = {"mode": tier.mode.join_mode}
    if tier.mode is TierMode.AT_LEAST:
        settings["min_pass_count"] = tier.minimum_passes
    settings = with_schema_version(DECISION_JOIN_PLUGIN, settings, registry=registry)
    inputs = [_binding("parents", anchor[0], anchor[1])]
    inputs.extend(
        _binding(f"decision_{criterion_id}", stage_id, port)
        for criterion_id, stage_id, port in decisions
    )
    accumulator.add(
        stage_id=tier.join_stage_id,
        descriptor=join,
        settings=settings,
        inputs=inputs,
        origin=StageOrigin(
            stage_id=tier.join_stage_id,
            role="policy",
            tier_id=tier.id,
            label=f"{tier.title} policy",
        ),
    )
    join_port = _port_for_contract(join, PARENT_V1.id) or "primary"
    return tier.join_stage_id, join_port


def _lower_step(
    step: StepConfig,
    *,
    role: str,
    registry: PluginRegistry,
    accumulator: _StageAccumulator,
    anchor: tuple[str, str] | None,
    settings: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    descriptor = _descriptor(registry, step.backend, where=f"{role} step {step.id!r}")
    inputs = [] if anchor is None else [_binding("primary", anchor[0], anchor[1])]
    accumulator.add(
        stage_id=step.id,
        descriptor=descriptor,
        settings=step.settings if settings is None else settings,
        inputs=inputs,
        origin=StageOrigin(stage_id=step.id, role=role, label=descriptor.display_name),
    )
    return step.id, _port_for_contract(descriptor, PARENT_V1.id) or "primary"


def lower_cascade(
    cascade: CascadeConfig,
    *,
    registry: PluginRegistry | None = None,
    library_path: str | None = None,
    library_format: LibraryFormat | None = None,
    allow_missing_library: bool = False,
) -> LoweredCascade:
    """Render a cascade as an ordered, compilable :class:`PipelineConfig`.

    Set ``allow_missing_library`` to compile a cascade that names no library --
    the shape of the funnel is checkable long before a file is chosen.  The
    result is marked as carrying a placeholder path and must not be executed.
    """

    active_registry = registry or create_builtin_registry()
    accumulator = _StageAccumulator()

    source_descriptor = _descriptor(
        active_registry, cascade.ingest.backend, where="the ingest step"
    )
    resolved = resolve_library(
        cascade.library,
        override_path=library_path,
        override_format=library_format,
        base_settings=cascade.ingest.settings,
        base_plugin=source_descriptor.key,
        allow_placeholder=allow_missing_library,
    )
    if resolved.plugin != source_descriptor.key:
        source_descriptor = _descriptor(active_registry, resolved.plugin, where="the ingest step")
    accumulator.add(
        stage_id=cascade.ingest.id,
        descriptor=source_descriptor,
        settings=resolved.settings,
        inputs=[],
        origin=StageOrigin(stage_id=cascade.ingest.id, role="ingest", label="Molecule library"),
    )
    anchor: tuple[str, str] = (cascade.ingest.id, "primary")

    if cascade.standardize is not None and cascade.standardize.enabled:
        anchor = _lower_step(
            cascade.standardize,
            role="standardize",
            registry=active_registry,
            accumulator=accumulator,
            anchor=anchor,
        )

    for tier in cascade.tiers:
        if not tier.enabled:
            continue
        anchor = _lower_tier(
            tier,
            registry=active_registry,
            accumulator=accumulator,
            anchor=anchor,
        )

    for step in cascade.finalize.steps:
        if not step.enabled:
            continue
        settings = _finalize_settings(step, cascade, registry=active_registry)
        anchor = _lower_step(
            step,
            role="finalize",
            registry=active_registry,
            accumulator=accumulator,
            anchor=anchor,
            settings=settings,
        )

    pipeline = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": cascade.name,
            "description": cascade.description,
            "stages": accumulator.stages,
            "metadata": _pipeline_metadata(cascade, resolved.path),
        }
    )
    return LoweredCascade(
        pipeline=pipeline,
        library_path=resolved.path,
        origins=tuple(accumulator.origins),
        library_is_placeholder=resolved.is_placeholder,
    )


def _finalize_settings(
    step: StepConfig,
    cascade: CascadeConfig,
    *,
    registry: PluginRegistry,
) -> dict[str, Any]:
    """Push the single authoritative shortlist size and seed into selectors.

    Only fields the selected plugin actually declares are written, so a
    third-party selector without a ``seed`` option is not handed one.
    """

    entry = _entry(registry, step.backend, where=f"finalize step {step.id!r}")
    settings = dict(step.settings)
    if entry.descriptor.kind is not PluginKind.SELECTOR:
        return settings
    config_model = getattr(entry.plugin, "config_model", None)
    fields = set(getattr(config_model, "model_fields", {}))
    if "target_count" in fields:
        settings["target_count"] = cascade.finalize.target_count
    if "seed" in fields:
        settings["seed"] = cascade.finalize.seed
    return settings


def _pipeline_metadata(cascade: CascadeConfig, library_path: str) -> dict[str, Any]:
    metadata = dict(cascade.metadata)
    metadata.update(
        {
            "cascade_name": cascade.name,
            "cascade_schema_version": cascade.schema_version,
            "final_parent_target": cascade.finalize.target_count,
            "random_seed": cascade.finalize.seed,
            "library_path": library_path,
            "docking_included": False,
            "tiers": [
                {
                    "id": tier.id,
                    "title": tier.title,
                    "mode": tier.mode.value,
                    "minimum_passes": tier.minimum_passes,
                    "criteria": [criterion.id for criterion in tier.active_criteria],
                }
                for tier in cascade.tiers
                if tier.enabled and tier.active_criteria
            ],
        }
    )
    return metadata


__all__ = [
    "DECISION_JOIN_PLUGIN",
    "LoweredCascade",
    "StageOrigin",
    "lower_cascade",
]
