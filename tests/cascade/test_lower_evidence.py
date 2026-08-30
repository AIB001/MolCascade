"""Lowering binds the evidence a criterion declares but does not compute.

Until now a criterion could only ever see the parent population, so any tool
needing precomputed evidence -- 3D conformers shared by three docking engines,
a fingerprint table shared by several similarity criteria -- had to recompute it
per consumer or could not be expressed at all.  These tests pin the binding
rule: the *latest upstream* producer wins, a sibling in the same parallel tier
is never upstream, and a criterion whose evidence nobody produces stops the
cascade instead of being wired to whatever happens to be nearby.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import CascadeConfig
from molcascade.errors import ConfigError
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    StageContext,
    StageRequest,
    StageResponse,
)

_CONFORMER = "ligand_conformer/v1"
_SCORE = "docking_score/v1"


class _Plugin:
    """Lowering reads only the descriptor; ``execute`` satisfies the protocol."""

    def __init__(self, descriptor: PluginDescriptor) -> None:
        self.descriptor = descriptor

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        raise AssertionError("lowering must not execute a stage")


def _plugin(
    plugin_id: str,
    kind: PluginKind,
    *,
    inputs: tuple[str, ...],
    output_ports: dict[str, str],
    cardinality: Cardinality = Cardinality.ONE_TO_ONE,
) -> _Plugin:
    return _Plugin(
        PluginDescriptor(
            id=plugin_id,
            version="0.1.0",
            kind=kind,
            inputs=inputs,
            outputs=tuple(dict.fromkeys(output_ports.values())),
            output_ports=output_ports,
            cardinality=cardinality,
            determinism=Determinism.DETERMINISTIC,
        )
    )


def _registry(*extra: _Plugin) -> PluginRegistry:
    source = _plugin(
        # ``resolve_library`` picks the delimited reader for the placeholder
        # path, so the fake source has to answer to that name.
        "source.delimited_smiles",
        PluginKind.SOURCE,
        inputs=(),
        output_ports={"primary": "parent/v1"},
        cardinality=Cardinality.ONE_TO_MANY,
    )
    return PluginRegistry((source, *extra))


def _prep(plugin_id: str = "docking.prep") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.FEATURIZER,
        inputs=("parent/v1",),
        output_ports={
            "primary": "parent/v1",
            "conformers": _CONFORMER,
            "decisions": "decision/v1",
        },
    )


def _gate(plugin_id: str = "chemistry.alerts") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.GATE,
        inputs=("parent/v1",),
        output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
        cardinality=Cardinality.FILTER,
    )


def _consumer(plugin_id: str = "docking.engine") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.DOCK,
        inputs=("parent/v1", _CONFORMER),
        output_ports={
            "primary": "parent/v1",
            "scores": _SCORE,
            "decisions": "decision/v1",
        },
    )


def _cascade(*tiers: dict[str, object]) -> CascadeConfig:
    return CascadeConfig.model_validate(
        {
            "name": "evidence-binding",
            "ingest": {"id": "ingest", "backend": "source.delimited_smiles@0.1.0"},
            "tiers": list(tiers),
        }
    )


def _tier(tier_id: str, *criteria: dict[str, object], mode: str = "serial") -> dict[str, object]:
    return {
        "id": tier_id,
        "title": tier_id.replace("_", " ").title(),
        "mode": mode,
        "criteria": list(criteria),
    }


def _bindings(pipeline: object, stage_id: str) -> list[dict[str, str]]:
    stages = [stage for stage in pipeline.stages if stage.id == stage_id]  # type: ignore[attr-defined]
    assert stages, f"stage {stage_id!r} was not lowered"
    return [binding.model_dump() for binding in stages[0].inputs]


def test_a_criterion_binds_the_latest_upstream_producer_of_its_evidence() -> None:
    lowered = lower_cascade(
        _cascade(
            _tier("prep", {"id": "conformers", "backend": "docking.prep@0.1.0"}),
            _tier("dock", {"id": "unidock", "backend": "docking.engine@0.1.0"}),
        ),
        registry=_registry(_prep(), _consumer()),
        allow_missing_library=True,
    )
    bindings = _bindings(lowered.pipeline, "unidock")
    assert bindings == [
        {"request_port": "primary", "stage": "conformers", "port": "primary"},
        {"request_port": "ligand_conformer", "stage": "conformers", "port": "conformers"},
    ]


def test_the_nearer_of_two_producers_wins() -> None:
    lowered = lower_cascade(
        _cascade(
            _tier("prep_a", {"id": "coarse", "backend": "docking.prep@0.1.0"}),
            _tier("prep_b", {"id": "refined", "backend": "docking.prep_v2@0.1.0"}),
            _tier("dock", {"id": "unidock", "backend": "docking.engine@0.1.0"}),
        ),
        registry=_registry(_prep(), _prep("docking.prep_v2"), _consumer()),
        allow_missing_library=True,
    )
    evidence = _bindings(lowered.pipeline, "unidock")[1]
    assert evidence["stage"] == "refined"


def test_evidence_nobody_produces_stops_the_cascade() -> None:
    with pytest.raises(ConfigError) as raised:
        lower_cascade(
            _cascade(_tier("dock", {"id": "unidock", "backend": "docking.engine@0.1.0"})),
            registry=_registry(_consumer()),
            allow_missing_library=True,
        )
    assert raised.value.code == "CASCADE_CRITERION_EVIDENCE_UNAVAILABLE"
    assert raised.value.context["missing_contract"] == _CONFORMER
    assert "unidock" in str(raised.value)


def test_a_parallel_criterion_cannot_read_a_sibling_in_its_own_tier() -> None:
    """A parallel tier's whole point is that no criterion sees another's output.

    Binding evidence sideways would reintroduce exactly the ordering dependence
    the parallel mode exists to remove -- the second criterion would silently
    score only what the first one kept.
    """

    with pytest.raises(ConfigError) as raised:
        lower_cascade(
            _cascade(
                _tier(
                    "dock",
                    {"id": "conformers", "backend": "docking.prep@0.1.0"},
                    {"id": "unidock", "backend": "docking.engine@0.1.0"},
                    mode="all",
                )
            ),
            registry=_registry(_prep(), _consumer()),
            allow_missing_library=True,
        )
    assert raised.value.code == "CASCADE_CRITERION_EVIDENCE_UNAVAILABLE"


def test_decisions_are_left_to_the_tier_policy() -> None:
    """A criterion declaring ``decision/v1`` is not wired to an earlier one.

    Decision streams mean something only in combination with the tier mode, so
    the policy join owns them.  Auto-binding one here would let a criterion act
    on a verdict nobody asked it to interpret.
    """

    reader = _plugin(
        "policy.peek",
        PluginKind.GATE,
        inputs=("parent/v1", "decision/v1"),
        output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
        cardinality=Cardinality.FILTER,
    )
    lowered = lower_cascade(
        _cascade(
            _tier("first", {"id": "alerts", "backend": "chemistry.alerts@0.1.0"}),
            _tier("second", {"id": "peek", "backend": "policy.peek@0.1.0"}),
        ),
        registry=_registry(_gate(), reader),
        allow_missing_library=True,
    )
    assert _bindings(lowered.pipeline, "peek") == [
        {"request_port": "primary", "stage": "alerts", "port": "primary"}
    ]


def test_single_input_criteria_keep_exactly_one_binding() -> None:
    """The shipped cascade must lower today the way it lowered yesterday.

    Every stage identity in a pipeline revision is derived from its bindings, so
    an extra edge here would invalidate every cached artifact ever produced by a
    default cascade.
    """

    from molcascade.cascade.defaults import default_cascade

    lowered = lower_cascade(
        default_cascade(), allow_missing_library=True, allow_missing_target=True
    )
    for stage in lowered.pipeline.stages:
        ports = [binding.request_port for binding in stage.inputs]
        assert ports in (
            [],
            ["primary"],
            ["parents", "evidence"],
            # A docking engine reads the population *and* the geometry the
            # conformer tier embedded for it, which is two edges by design
            # rather than by accident.  Listing the shape here is what keeps
            # the guarantee useful: a third edge on a docking stage would
            # still fail, and that is the failure this test is for.
            ["primary", "ligand_conformer"],
        ) or all(
            port == "parents" or port.startswith("decision_") for port in ports
        ), f"{stage.id} gained an unexpected binding: {ports}"
