"""Configurable RDKit physicochemical range gate.

This gate is intentionally separate from the structural hard-rule catalogue:
teams can keep invalid/reactive chemistry terminal while treating drug-like
property windows as a replaceable project policy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, model_validator

from molcascade.chemistry.datasets import iter_contract_batches, require_single_input
from molcascade.config.canonical import canonical_sha256
from molcascade.config.models import StrictFrozenModel
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import PendingOutput, StageContext, StageRequest, StageResponse
from molcascade.plugins.manifest import (
    Cardinality,
    Determinism,
    PluginDescriptor,
    PluginKind,
)

_PARENT_PATH = Path("datasets/passed_parents/part-00000.parquet")
_DECISION_PATH = Path("datasets/decisions/part-00000.parquet")


class PropertyRangeGateConfig(StrictFrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    batch_size: int = Field(default=16_384, ge=1, le=250_000)
    mw_min: float | None = Field(default=100.0, ge=0.0, le=10_000.0)
    mw_max: float | None = Field(default=700.0, ge=0.0, le=10_000.0)
    clogp_min: float | None = Field(default=-3.0, ge=-100.0, le=100.0)
    clogp_max: float | None = Field(default=8.0, ge=-100.0, le=100.0)
    tpsa_min: float | None = Field(default=0.0, ge=0.0, le=10_000.0)
    tpsa_max: float | None = Field(default=250.0, ge=0.0, le=10_000.0)
    hbd_max: int | None = Field(default=10, ge=0, le=1_000)
    hba_max: int | None = Field(default=15, ge=0, le=1_000)
    rotatable_bonds_max: int | None = Field(default=20, ge=0, le=10_000)
    heavy_atom_min: int | None = Field(default=3, ge=0, le=100_000)
    heavy_atom_max: int | None = Field(default=100, ge=0, le=100_000)
    absolute_formal_charge_max: int | None = Field(default=3, ge=0, le=100)

    @model_validator(mode="after")
    def _ordered_ranges(self) -> Self:
        for lower_name, upper_name in (
            ("mw_min", "mw_max"),
            ("clogp_min", "clogp_max"),
            ("tpsa_min", "tpsa_max"),
            ("heavy_atom_min", "heavy_atom_max"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{lower_name} must be no greater than {upper_name}")
        return self


class RDKitPropertyRangeGatePlugin:
    descriptor = PluginDescriptor(
        id="chemistry.rdkit_property_range_gate",
        version="0.1.0",
        kind=PluginKind.GATE,
        inputs=(PARENT_V1.id,),
        outputs=(PARENT_V1.id, DECISION_V1.id),
        output_ports={"primary": PARENT_V1.id, "decisions": DECISION_V1.id},
        cardinality=Cardinality.FILTER,
        determinism=Determinism.DETERMINISTIC,
        display_name="RDKit physicochemical range gate",
        description="Project-defined MW/logP/TPSA/HBD/HBA/rotor/charge windows with full decisions.",
    )
    config_model = PropertyRangeGateConfig

    @staticmethod
    def _check(
        value: float | int,
        *,
        name: str,
        minimum: float | int | None,
        maximum: float | int | None,
    ) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []
        label = name.upper()
        if minimum is not None and value < minimum:
            findings.append(
                (f"PROPERTY_{label}_BELOW_MIN", f"{name}={value!r} is below {minimum!r}")
            )
        if maximum is not None and value > maximum:
            findings.append(
                (f"PROPERTY_{label}_ABOVE_MAX", f"{name}={value!r} is above {maximum!r}")
            )
        return findings

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        try:
            config = self.config_model.model_validate(dict(request.config))
        except ValidationError as error:
            raise PluginError(
                f"invalid property gate configuration: {error}",
                code="PLUGIN_CONFIG_INVALID",
                context={"error_count": error.error_count()},
            ) from error
        stage_input = require_single_input(request.inputs, contract_id=PARENT_V1.id)
        context.staging_root.mkdir(parents=True, exist_ok=True)
        for relative in (_PARENT_PATH, _DECISION_PATH):
            destination = context.staging_root / relative
            if destination.exists() or destination.is_symlink():
                raise PluginError(
                    f"property-gate output already exists: {relative.as_posix()}",
                    code="PLUGIN_STAGING_NOT_EMPTY",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)

        policy = config.model_dump(mode="json")
        policy.pop("batch_size", None)
        policy_id = "property-gate-policy:sha256:" + canonical_sha256(
            {
                "backend": "rdkit",
                "backend_version": rdBase.rdkitVersion,
                "policy": policy,
            }
        )
        parent_destination = context.staging_root / _PARENT_PATH
        decision_destination = context.staging_root / _DECISION_PATH
        input_count = 0
        passed_count = 0
        decision_count = 0
        try:
            with pq.ParquetWriter(parent_destination, PARENT_V1.schema, compression="zstd") as parents:
                with pq.ParquetWriter(
                    decision_destination,
                    DECISION_V1.schema,
                    compression="zstd",
                ) as decisions:
                    for batch in iter_contract_batches(
                        stage_input,
                        PARENT_V1,
                        batch_size=config.batch_size,
                    ):
                        passed: list[dict[str, Any]] = []
                        decision_rows: list[dict[str, Any]] = []
                        for row in batch.to_pylist():
                            input_count += 1
                            parent_id = row.get("parent_id")
                            molecule = Chem.MolFromSmiles(
                                row.get("parent_smiles")
                                if isinstance(row.get("parent_smiles"), str)
                                else ""
                            )
                            if molecule is None:
                                raise PluginError(
                                    "registered parent cannot be parsed by RDKit property gate",
                                    code="PROPERTY_GATE_PARENT_INVALID",
                                    context={"parent_id": str(parent_id)},
                                )
                            properties: dict[str, float | int] = {
                                "mw": float(Descriptors.MolWt(molecule)),
                                "clogp": float(Crippen.MolLogP(molecule)),
                                "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
                                "hbd": int(rdMolDescriptors.CalcNumHBD(molecule)),
                                "hba": int(rdMolDescriptors.CalcNumHBA(molecule)),
                                "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
                                "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
                                "absolute_formal_charge": abs(int(Chem.GetFormalCharge(molecule))),
                            }
                            if any(
                                isinstance(value, float) and not math.isfinite(value)
                                for value in properties.values()
                            ):
                                raise PluginError(
                                    "RDKit produced a non-finite gate property",
                                    code="PROPERTY_GATE_NON_FINITE",
                                    context={"parent_id": str(parent_id)},
                                )
                            findings: list[tuple[str, str]] = []
                            findings += self._check(
                                properties["mw"], name="mw", minimum=config.mw_min, maximum=config.mw_max
                            )
                            findings += self._check(
                                properties["clogp"],
                                name="clogp",
                                minimum=config.clogp_min,
                                maximum=config.clogp_max,
                            )
                            findings += self._check(
                                properties["tpsa"],
                                name="tpsa",
                                minimum=config.tpsa_min,
                                maximum=config.tpsa_max,
                            )
                            findings += self._check(
                                properties["hbd"], name="hbd", minimum=None, maximum=config.hbd_max
                            )
                            findings += self._check(
                                properties["hba"], name="hba", minimum=None, maximum=config.hba_max
                            )
                            findings += self._check(
                                properties["rotatable_bonds"],
                                name="rotatable_bonds",
                                minimum=None,
                                maximum=config.rotatable_bonds_max,
                            )
                            findings += self._check(
                                properties["heavy_atom_count"],
                                name="heavy_atom_count",
                                minimum=config.heavy_atom_min,
                                maximum=config.heavy_atom_max,
                            )
                            findings += self._check(
                                properties["absolute_formal_charge"],
                                name="absolute_formal_charge",
                                minimum=None,
                                maximum=config.absolute_formal_charge_max,
                            )
                            if findings:
                                for reason_code, detail in findings:
                                    decision_rows.append(
                                        {
                                            "entity_id": parent_id,
                                            "entity_kind": "PARENT",
                                            "stage_id": request.stage_id,
                                            "outcome": "REJECT",
                                            "reason_code": reason_code,
                                            "rule_id": policy_id,
                                            "detail": detail,
                                        }
                                    )
                            else:
                                passed.append(row)
                                passed_count += 1
                                decision_rows.append(
                                    {
                                        "entity_id": parent_id,
                                        "entity_kind": "PARENT",
                                        "stage_id": request.stage_id,
                                        "outcome": "PASS",
                                        "reason_code": "PROPERTY_RANGE_PASS",
                                        "rule_id": policy_id,
                                        "detail": json.dumps(
                                            properties,
                                            ensure_ascii=False,
                                            allow_nan=False,
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        ),
                                    }
                                )
                            decision_count += max(1, len(findings))
                        if passed:
                            parents.write_table(pa.Table.from_pylist(passed, schema=PARENT_V1.schema))
                        decisions.write_table(
                            pa.Table.from_pylist(decision_rows, schema=DECISION_V1.schema)
                        )
            if input_count == 0:
                raise PluginError(
                    "property-gate input contains no parents",
                    code="PROPERTY_GATE_EMPTY_INPUT",
                )
        except BaseException:
            parent_destination.unlink(missing_ok=True)
            decision_destination.unlink(missing_ok=True)
            raise
        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    PARENT_V1.id,
                    (_PARENT_PATH.as_posix(),),
                    {"row_count": passed_count},
                ),
                "decisions": PendingOutput(
                    DECISION_V1.id,
                    (_DECISION_PATH.as_posix(),),
                    {"row_count": decision_count, "policy_id": policy_id},
                ),
            },
            metadata={
                "input_count": input_count,
                "output_count": passed_count,
                "reject_count": input_count - passed_count,
                "decision_count": decision_count,
                "policy_id": policy_id,
                "backend": "rdkit",
                "backend_version": rdBase.rdkitVersion,
            },
        )


__all__ = ["PropertyRangeGateConfig", "RDKitPropertyRangeGatePlugin"]
