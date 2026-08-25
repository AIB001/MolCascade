from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from molcascade.config.models import PipelineConfig
from molcascade.pipeline import PipelineCompiler
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner


def _stage(
    stage_id: str,
    slot: str,
    plugin: str,
    *,
    inputs: list[dict[str, str]] | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": stage_id,
        "slot": slot,
        "plugin": plugin,
        "inputs": inputs or [],
        "config": config or {},
        "enabled": True,
    }


def _edge(stage: str, port: str = "primary", request_port: str = "primary") -> dict[str, str]:
    return {"request_port": request_port, "stage": stage, "port": port}


def _pipeline(source: Path, *, parallel: bool) -> PipelineConfig:
    property_config = {
        "mw_min": 50.0,
        "mw_max": None,
        "clogp_min": None,
        "clogp_max": None,
        "tpsa_min": None,
        "tpsa_max": None,
        "hbd_max": None,
        "hba_max": None,
        "rotatable_bonds_max": None,
        "heavy_atom_min": None,
        "heavy_atom_max": None,
        "absolute_formal_charge_max": None,
    }
    stages = [
        _stage(
            "source",
            "source",
            "source.delimited_smiles@0.1.0",
            config={"path": str(source), "smiles_column": "smiles"},
        ),
        _stage(
            "standardize",
            "standardize",
            "chemistry.rdkit_standardize@0.1.0",
            inputs=[_edge("source")],
        ),
        _stage(
            "property_gate",
            "hard_gate",
            "chemistry.rdkit_property_range_gate@0.1.0",
            inputs=[_edge("standardize")],
            config=property_config,
        ),
        _stage(
            "drug_likeness",
            "hard_gate",
            "chemistry.rdkit_drug_likeness@0.1.0",
            inputs=[_edge("standardize" if parallel else "property_gate")],
            config={
                "maximum_rule_of_five_violations": 0,
                "minimum_qed": None,
                "failure_action": "reject",
            },
        ),
    ]
    if parallel:
        stages.append(
            _stage(
                "all_policy",
                "hard_gate",
                "policy.native_decision_join@0.1.0",
                inputs=[
                    _edge("standardize", request_port="parents"),
                    _edge(
                        "property_gate",
                        "decisions",
                        "decision_property_gate",
                    ),
                    _edge(
                        "drug_likeness",
                        "decisions",
                        "decision_drug_likeness",
                    ),
                ],
                config={"mode": "all_required"},
            )
        )
    return PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "parallel" if parallel else "serial",
            "stages": stages,
        }
    )


def _response_metadata(runner: LocalRunner, result: object, stage_id: str) -> dict[str, object]:
    stage = next(stage for stage in result.stages if stage.stage_id == stage_id)  # type: ignore[attr-defined]
    assert stage.output_ref is not None
    manifest = runner.store.verify(stage.output_ref.artifact_id)
    return dict(manifest.metadata["response_metadata"])


def _parent_smiles(runner: LocalRunner, result: object) -> set[str]:
    assert result.output_ref is not None  # type: ignore[attr-defined]
    ref = result.output_ref  # type: ignore[attr-defined]
    dataset = runner.store.resolve_dataset(ref, verify=True)
    return {
        row["parent_smiles"]
        for path in ref.file_paths
        for row in pq.read_table(dataset / path).to_pylist()
    }


def test_parallel_all_and_serial_export_to_distinct_runtime_graphs(tmp_path: Path) -> None:
    source = tmp_path / "molecules.csv"
    source.write_text(
        "smiles\n"
        "CCO\n"
        "c1ccccc1\n"
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O\n"
        f"{'C' * 40}\n",
        encoding="utf-8",
    )
    registry = create_builtin_registry()
    parallel = _pipeline(source, parallel=True)
    serial = _pipeline(source, parallel=False)
    parallel_plan = PipelineCompiler(registry).compile(parallel)
    serial_plan = PipelineCompiler(registry).compile(serial)

    parallel_drug = next(
        stage for stage in parallel_plan.stages if stage.stage_id == "drug_likeness"
    )
    serial_drug = next(
        stage for stage in serial_plan.stages if stage.stage_id == "drug_likeness"
    )
    assert parallel_drug.input_bindings[0].source_stage_id == "standardize"
    assert serial_drug.input_bindings[0].source_stage_id == "property_gate"
    assert len(parallel_plan.stages[-1].input_bindings) == 3

    parallel_runner = LocalRunner(tmp_path / "parallel-workspace", plugins=registry)
    serial_runner = LocalRunner(tmp_path / "serial-workspace", plugins=registry)
    parallel_result = parallel_runner.run(parallel, run_id="parallel-run")
    serial_result = serial_runner.run(serial, run_id="serial-run")

    # Both policies compute the intersection for independent deterministic
    # predicates, but they expose materially different branch populations.
    expected = {"c1ccccc1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"}
    assert _parent_smiles(parallel_runner, parallel_result) == expected
    assert _parent_smiles(serial_runner, serial_result) == expected
    assert _response_metadata(
        parallel_runner, parallel_result, "property_gate"
    )["input_count"] == 4
    assert _response_metadata(
        parallel_runner, parallel_result, "drug_likeness"
    )["input_count"] == 4
    assert _response_metadata(
        serial_runner, serial_result, "drug_likeness"
    )["input_count"] == 3
    assert _response_metadata(
        parallel_runner, parallel_result, "all_policy"
    )["mode"] == "all_required"
