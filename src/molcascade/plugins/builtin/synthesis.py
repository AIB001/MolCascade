"""Local synthesis-prioritization evidence plugins.

RDKit Contrib SA_Score is a fragment-and-complexity proxy.  It does not search
for a route, estimate a step count, verify building-block availability, or
prove that a molecule can be synthesized.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from molcascade.chemistry.datasets import iter_contract_batches, require_single_input
from molcascade.config.canonical import canonical_json, canonical_sha256
from molcascade.config.models import StrictFrozenModel
from molcascade.contracts import PARENT_V1, SYNTHESIS_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import PendingOutput, StageContext, StageRequest, StageResponse
from molcascade.plugins.manifest import (
    Cardinality,
    Determinism,
    PluginDescriptor,
    PluginKind,
)

_PARENT_PATH = Path("datasets/parents/part-00000.parquet")
_SCORE_PATH = Path("datasets/synthesis_scores/part-00000.parquet")
_IMPLEMENTATION_VERSION = 1


class SAScoreConfig(StrictFrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    batch_size: int = Field(default=16_384, ge=1, le=250_000)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sa_backend() -> tuple[Callable[[Any], float] | None, dict[str, str]]:
    """Load RDKit's installed Contrib implementation and bind its two assets."""

    module: Any
    source: Path
    try:
        from rdkit.Contrib.SA_Score import sascorer as module

        source = Path(module.__file__ or "")
    except (ImportError, OSError):
        try:
            from rdkit import RDConfig

            source = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
            specification = importlib.util.spec_from_file_location(
                "_molcascade_rdkit_sa_score",
                source,
            )
            if specification is None or specification.loader is None:
                return None, {}
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
        except (ImportError, OSError, AttributeError, RuntimeError, SyntaxError):
            return None, {}

    weights = source.with_name("fpscores.pkl.gz")
    calculator = getattr(module, "calculateScore", None)
    if not callable(calculator) or not source.is_file() or not weights.is_file():
        return None, {}
    try:
        provenance = {
            "source_sha256": _sha256_file(source),
            "weights_sha256": _sha256_file(weights),
        }
    except OSError:
        return None, {}
    return calculator, provenance


def _validated_config(request: StageRequest) -> SAScoreConfig:
    try:
        return SAScoreConfig.model_validate(dict(request.config))
    except ValidationError as error:
        raise PluginError(
            f"invalid SA-score configuration: {error}",
            code="PLUGIN_CONFIG_INVALID",
            context={"error_count": error.error_count()},
        ) from error


def _prepare_outputs(context: StageContext) -> tuple[Path, Path]:
    context.staging_root.mkdir(parents=True, exist_ok=True)
    parent_destination = context.staging_root / _PARENT_PATH
    score_destination = context.staging_root / _SCORE_PATH
    destinations = (parent_destination, score_destination)
    for relative, destination in zip(
        (_PARENT_PATH, _SCORE_PATH), destinations, strict=True
    ):
        if destination.exists() or destination.is_symlink():
            raise PluginError(
                f"SA-score output already exists: {relative.as_posix()}",
                code="PLUGIN_STAGING_NOT_EMPTY",
                context={"path": relative.as_posix()},
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
    return parent_destination, score_destination


class RDKitSAScorePlugin:
    """Calculate the RDKit Contrib Ertl-Schuffenhauer SA-score proxy."""

    descriptor = PluginDescriptor(
        id="synthesis.rdkit_sa_score",
        version="0.1.0",
        kind=PluginKind.SYNTHESIS,
        inputs=(PARENT_V1.id,),
        outputs=(PARENT_V1.id, SYNTHESIS_SCORE_V1.id),
        output_ports={
            "primary": PARENT_V1.id,
            "synthesis_scores": SYNTHESIS_SCORE_V1.id,
        },
        cardinality=Cardinality.ONE_TO_ONE,
        determinism=Determinism.DETERMINISTIC,
        display_name="RDKit synthetic-accessibility score",
        description=(
            "Local Ertl-Schuffenhauer fragment/complexity prioritization proxy "
            "(higher is harder); not retrosynthesis or a route-step estimate."
        ),
    )
    config_model = SAScoreConfig

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        from rdkit import Chem, rdBase

        config = _validated_config(request)
        stage_input = require_single_input(request.inputs, contract_id=PARENT_V1.id)
        calculator, asset_provenance = _load_sa_backend()
        if calculator is None:
            raise PluginError(
                "RDKit Contrib SA_Score assets are unavailable in this installation",
                code="SYNTHESIS_SASCORE_UNAVAILABLE",
                hint=(
                    "Install an RDKit distribution containing Contrib/SA_Score/"
                    "sascorer.py and fpscores.pkl.gz, or choose another synthesis plugin."
                ),
                context={"backend": "rdkit", "component": "Contrib/SA_Score"},
            )

        parent_destination, score_destination = _prepare_outputs(context)
        method_id = "synthesis-sa:sha256:" + canonical_sha256(
            {
                "backend": "rdkit-contrib",
                "backend_version": rdBase.rdkitVersion,
                "implementation_version": _IMPLEMENTATION_VERSION,
                "method": "Ertl-Schuffenhauer SA Score (2009)",
                "doi": "10.1186/1758-2946-1-8",
                "assets": asset_provenance,
                "direction": "HIGHER_HARDER",
            }
        )
        input_count = 0
        out_of_nominal_range_count = 0
        try:
            with pq.ParquetWriter(
                parent_destination,
                PARENT_V1.schema,
                compression="zstd",
            ) as parent_writer, pq.ParquetWriter(
                score_destination,
                SYNTHESIS_SCORE_V1.schema,
                compression="zstd",
            ) as score_writer:
                for batch in iter_contract_batches(
                    stage_input,
                    PARENT_V1,
                    batch_size=config.batch_size,
                ):
                    score_rows: list[dict[str, Any]] = []
                    for row in batch.select(
                        ["parent_id", "parent_smiles"]
                    ).to_pylist():
                        input_count += 1
                        parent_id = row.get("parent_id")
                        smiles = row.get("parent_smiles")
                        molecule = Chem.MolFromSmiles(
                            smiles if isinstance(smiles, str) else ""
                        )
                        if molecule is None:
                            raise PluginError(
                                "registered parent cannot be parsed for SA score",
                                code="SYNTHESIS_PARENT_INVALID",
                                context={"parent_id": str(parent_id)},
                            )
                        try:
                            score = float(calculator(molecule))
                        except (
                            ArithmeticError,
                            KeyError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            raise PluginError(
                                "RDKit Contrib SA-score calculation failed",
                                code="SYNTHESIS_SCORE_FAILED",
                                context={"parent_id": str(parent_id)},
                            ) from error
                        if not math.isfinite(score):
                            raise PluginError(
                                "RDKit Contrib SA-score result is non-finite",
                                code="SYNTHESIS_SCORE_NON_FINITE",
                                context={"parent_id": str(parent_id)},
                            )
                        warnings = ["SYNTHESIS_PROXY_NOT_ROUTE"]
                        if not 1.0 <= score <= 10.0:
                            warnings.append("SYNTHESIS_SCORE_OUTSIDE_NOMINAL_RANGE")
                            out_of_nominal_range_count += 1
                        score_rows.append(
                            {
                                "parent_id": parent_id,
                                "method_id": method_id,
                                "score": score,
                                "direction": "HIGHER_HARDER",
                                "domain": (
                                    "RDKit Contrib SA_Score fragment-and-complexity "
                                    "proxy; nominal range 1-10"
                                ),
                                "warning_codes_json": canonical_json(warnings),
                            }
                        )
                    parent_writer.write_batch(batch)
                    score_writer.write_table(
                        pa.Table.from_pylist(
                            score_rows,
                            schema=SYNTHESIS_SCORE_V1.schema,
                        )
                    )
            if input_count == 0:
                raise PluginError(
                    "SA-score input contains no parents",
                    code="SYNTHESIS_EMPTY_INPUT",
                )
        except BaseException:
            parent_destination.unlink(missing_ok=True)
            score_destination.unlink(missing_ok=True)
            raise

        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    PARENT_V1.id,
                    (_PARENT_PATH.as_posix(),),
                    {"row_count": input_count},
                ),
                "synthesis_scores": PendingOutput(
                    SYNTHESIS_SCORE_V1.id,
                    (_SCORE_PATH.as_posix(),),
                    {"row_count": input_count, "method_id": method_id},
                ),
            },
            metadata={
                "input_count": input_count,
                "output_count": input_count,
                "method_id": method_id,
                "direction": "HIGHER_HARDER",
                "proxy_not_route": True,
                "out_of_nominal_range_count": out_of_nominal_range_count,
                "backend": "rdkit-contrib",
                "backend_version": rdBase.rdkitVersion,
                **asset_provenance,
            },
        )


__all__ = ["RDKitSAScorePlugin", "SAScoreConfig"]
