"""Tier-first cascade authoring models.

A :class:`~molcascade.config.models.PipelineConfig` is a flat, topologically
ordered stage list.  That shape is correct for execution and terrible for
authoring: a screening cascade is naturally a funnel of *tiers*, each tier
holding several *criteria* which are combined either in series or in parallel,
and each criterion carrying its own threshold.

This module makes the tier a first-class configuration object.  The cascade is
the file a scientist edits (in the browser or by hand); the pipeline is a
derived artifact produced by :mod:`molcascade.cascade.lower`.  Nothing in the
runtime, artifact, cache, or provenance layer changes, because lowering emits
exactly the same ordered stage list those layers already validate.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from molcascade.config.models import StrictFrozenModel, _freeze_json_mapping

CASCADE_SCHEMA_VERSION = 2

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_PLUGIN_REF_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*@\d+\.\d+\.\d+[0-9A-Za-z.+-]*$")


def _validate_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            "must start with a letter and contain only letters, digits, '.', '_' or '-'"
        )
    return value


class TierMode(StrEnum):
    """How the criteria inside one tier are combined.

    ``SERIAL`` and the three parallel modes compile to materially different
    graphs, not to different labels on the same graph.  In ``SERIAL`` a
    criterion only ever sees the survivors of its predecessor; in the parallel
    modes every criterion sees the same population entering the tier and their
    complete decision streams meet at one explicit policy join.
    """

    SERIAL = "serial"
    ALL = "all"
    ANY = "any"
    AT_LEAST = "at_least"

    @property
    def is_parallel(self) -> bool:
        return self is not TierMode.SERIAL

    @property
    def join_mode(self) -> str:
        """Return the ``policy.native_decision_join`` mode for this tier mode."""

        if self is TierMode.ALL:
            return "all_required"
        if self is TierMode.ANY:
            return "any_required"
        if self is TierMode.AT_LEAST:
            return "min_pass_count"
        raise ValueError("serial tiers do not use a decision join")


class LibraryFormat(StrEnum):
    """Accepted shapes of the molecule library handed to a cascade."""

    AUTO = "auto"
    DELIMITED = "delimited"
    XLSX = "xlsx"
    SDF = "sdf"
    PARQUET = "parquet"
    MOL2_DIRECTORY = "mol2_directory"


class LibraryConfig(StrictFrozenModel):
    """Where the molecules come from and how to read them.

    ``path`` is optional on purpose.  The intended workflow is that the browser
    builder produces a reusable *screening policy* and the command line supplies
    the library for each run, so the same configuration can screen many
    generated batches without being edited.
    """

    path: str | None = Field(default=None, min_length=1, max_length=4096)
    format: LibraryFormat = LibraryFormat.AUTO
    smiles_column: str | None = Field(default=None, min_length=1, max_length=256)
    id_column: str | None = Field(default=None, min_length=1, max_length=256)
    delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    has_header: bool | None = None
    sheet_name: str | None = Field(default=None, min_length=1, max_length=256)
    #: Leading rows to discard before the header, for exports that open with a
    #: title or a merged section banner.  Applies to delimited files and
    #: worksheets; the readers that have no rows to skip ignore it.
    skip_rows: int | None = Field(default=None, ge=0, le=1_048_575)
    batch_size: int = Field(default=65_536, ge=1, le=1_000_000)

    @field_validator("format", mode="before")
    @classmethod
    def _parse_format(cls, value: Any) -> Any:
        return LibraryFormat(value) if isinstance(value, str) else value

    @field_validator("path")
    @classmethod
    def _reject_control_characters(cls, value: str | None) -> str | None:
        if value is not None and any(character in value for character in "\r\n\x00"):
            raise ValueError("library path must not contain control characters")
        return value


class ThresholdDirection(StrEnum):
    """Which side of a numeric window keeps a molecule."""

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    WINDOW = "window"


class GateConfig(StrictFrozenModel):
    """Turn one criterion's numeric evidence into an explicit decision.

    Evidence producers (ADMET models, SA score, custom ML models) deliberately
    do not filter.  A criterion becomes a filter only when it carries a gate,
    which is where the tier's threshold lives.
    """

    backend: str = Field(min_length=1, max_length=512)
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if not _PLUGIN_REF_RE.fullmatch(value):
            raise ValueError("backend must be an exact 'plugin.id@major.minor.patch' reference")
        return value

    @field_validator("settings")
    @classmethod
    def _freeze_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)


class CriterionConfig(StrictFrozenModel):
    """One screening criterion inside a tier.

    ``criterion`` names the *scientific question* (for example ``pains_alerts``)
    and ``backend`` names the *exact tool* answering it.  Keeping the two
    separate is what lets a user swap RDKit for ``medchem``, ``rd_filters``, or
    their own model without rebuilding the cascade.
    """

    id: str = Field(min_length=1, max_length=128)
    criterion: str | None = Field(default=None, min_length=1, max_length=128)
    backend: str = Field(min_length=1, max_length=512)
    label: str | None = Field(default=None, min_length=1, max_length=256)
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    gate: GateConfig | None = None
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("criterion")
    @classmethod
    def _validate_criterion(cls, value: str | None) -> str | None:
        return None if value is None else _validate_identifier(value)

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if not _PLUGIN_REF_RE.fullmatch(value):
            raise ValueError("backend must be an exact 'plugin.id@major.minor.patch' reference")
        return value

    @field_validator("settings")
    @classmethod
    def _freeze_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

    @property
    def gate_stage_id(self) -> str:
        return f"{self.id}__gate"


class TierConfig(StrictFrozenModel):
    """One level of the funnel."""

    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=160)
    mode: TierMode = TierMode.SERIAL
    minimum_passes: int | None = Field(default=None, ge=1, le=1_000)
    criteria: tuple[CriterionConfig, ...] = ()
    note: str | None = Field(default=None, max_length=2048)
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("mode", mode="before")
    @classmethod
    def _parse_mode(cls, value: Any) -> Any:
        return TierMode(value) if isinstance(value, str) else value

    @field_validator("criteria", mode="before")
    @classmethod
    def _accept_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_tier(self) -> Self:
        seen: set[str] = set()
        for criterion in self.criteria:
            if criterion.id in seen:
                raise ValueError(f"duplicate criterion id {criterion.id!r} in tier {self.id!r}")
            seen.add(criterion.id)
        if self.mode is TierMode.AT_LEAST:
            if self.minimum_passes is None:
                raise ValueError(
                    f"tier {self.id!r} uses 'at_least' and must set minimum_passes"
                )
            active = sum(1 for criterion in self.criteria if criterion.enabled)
            if active and self.minimum_passes > active:
                raise ValueError(
                    f"tier {self.id!r} requires {self.minimum_passes} passing criteria "
                    f"but only {active} are enabled"
                )
        elif self.minimum_passes is not None:
            raise ValueError(
                f"tier {self.id!r} sets minimum_passes but its mode is not 'at_least'"
            )
        return self

    @property
    def active_criteria(self) -> tuple[CriterionConfig, ...]:
        return tuple(criterion for criterion in self.criteria if criterion.enabled)

    @property
    def join_stage_id(self) -> str:
        return f"{self.id}__policy"


class StepConfig(StrictFrozenModel):
    """A fixed, single-tool stage such as ingest, standardization or export."""

    id: str = Field(min_length=1, max_length=128)
    backend: str = Field(min_length=1, max_length=512)
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if not _PLUGIN_REF_RE.fullmatch(value):
            raise ValueError("backend must be an exact 'plugin.id@major.minor.patch' reference")
        return value

    @field_validator("settings")
    @classmethod
    def _freeze_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)


class FinalizeConfig(StrictFrozenModel):
    """The tail of the funnel: annotate, budget-select, and export a shortlist.

    ``target_count`` is authoritative.  Lowering writes it into whichever step
    declares a ``target_count`` setting so that the shortlist size is edited in
    exactly one place.
    """

    target_count: int = Field(default=20_000, ge=1, le=100_000_000)
    seed: int = Field(default=20_260_823, ge=0, le=2**63 - 1)
    steps: tuple[StepConfig, ...] = ()

    @field_validator("steps", mode="before")
    @classmethod
    def _accept_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate finalize step id {step.id!r}")
            seen.add(step.id)
        return self


class CascadeConfig(StrictFrozenModel):
    """Version 2 tier-first screening cascade.

    ``kind`` discriminates this file from a legacy flat pipeline configuration
    so a single loader can accept either without guessing.
    """

    schema_version: Literal[2] = CASCADE_SCHEMA_VERSION
    kind: Literal["cascade"] = "cascade"
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=4096)
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    ingest: StepConfig
    standardize: StepConfig | None = None
    tiers: tuple[TierConfig, ...] = ()
    finalize: FinalizeConfig = Field(default_factory=FinalizeConfig)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    @field_validator("tiers", mode="before")
    @classmethod
    def _accept_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("metadata")
    @classmethod
    def _freeze_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

    @model_validator(mode="after")
    def _validate_identity_space(self) -> Self:
        reserved = {self.ingest.id}
        if self.standardize is not None:
            if self.standardize.id in reserved:
                raise ValueError("standardize step id must differ from the ingest step id")
            reserved.add(self.standardize.id)
        tier_ids: set[str] = set()
        for tier in self.tiers:
            if tier.id in tier_ids:
                raise ValueError(f"duplicate tier id {tier.id!r}")
            tier_ids.add(tier.id)
            for stage_id in (tier.join_stage_id,):
                if stage_id in reserved:
                    raise ValueError(f"tier {tier.id!r} collides with an existing stage id")
                reserved.add(stage_id)
            for criterion in tier.criteria:
                for stage_id in (criterion.id, criterion.gate_stage_id):
                    if stage_id in reserved:
                        raise ValueError(
                            f"criterion {criterion.id!r} in tier {tier.id!r} collides with "
                            "another stage id"
                        )
                    reserved.add(stage_id)
        for step in self.finalize.steps:
            if step.id in reserved:
                raise ValueError(f"finalize step {step.id!r} collides with another stage id")
            reserved.add(step.id)
        if not any(tier.active_criteria for tier in self.tiers if tier.enabled) and not any(
            step.enabled for step in self.finalize.steps
        ):
            raise ValueError(
                "a cascade must contain at least one enabled criterion or finalize step"
            )
        return self

    @property
    def active_tiers(self) -> tuple[TierConfig, ...]:
        return tuple(tier for tier in self.tiers if tier.enabled and tier.active_criteria)


__all__ = [
    "CASCADE_SCHEMA_VERSION",
    "CascadeConfig",
    "CriterionConfig",
    "FinalizeConfig",
    "GateConfig",
    "LibraryConfig",
    "LibraryFormat",
    "StepConfig",
    "ThresholdDirection",
    "TierConfig",
    "TierMode",
]
