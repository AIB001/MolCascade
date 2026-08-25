from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from molcascade.artifacts import ArtifactProducer, LocalArtifactStore
from molcascade.contracts import DECISION_V1, PARENT_SOURCE_MAP_V1, PARENT_V1, RAW_MOLECULE_V1
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.hard_gate import RDKitHardGatePlugin
from molcascade.plugins.builtin.standardize import RDKitStandardizePlugin


def _producer(plugin: object) -> ArtifactProducer:
    descriptor = plugin.descriptor  # type: ignore[attr-defined]
    return ArtifactProducer(
        plugin_id=descriptor.id,
        plugin_version=descriptor.version,
        api_version=descriptor.api_version,
    )


def test_multi_output_commit_round_trip_routes_exact_port_views(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    raw_stage = store.begin_staging()
    raw_path = "datasets/raw/part-00000.parquet"
    raw_rows = [
        {
            "source_record_id": "source-a",
            "source_kind": "test",
            "source_uri": None,
            "source_index": 0,
            "generator_id": "generator-a",
            "batch_id": "batch-a",
            "source_candidate_id": "candidate-a",
            "raw_smiles": "CCO",
            "raw_molblock": None,
            "source_metadata_json": None,
        },
        {
            "source_record_id": "source-b",
            "source_kind": "test",
            "source_uri": None,
            "source_index": 1,
            "generator_id": "generator-a",
            "batch_id": "batch-a",
            "source_candidate_id": "candidate-b",
            "raw_smiles": "OCC",
            "raw_molblock": None,
            "source_metadata_json": None,
        },
    ]
    raw_destination = raw_stage / raw_path
    raw_destination.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(raw_rows, schema=RAW_MOLECULE_V1.schema), raw_destination)
    raw_manifest = store.commit(
        raw_stage,
        kind="stage-output",
        producer=ArtifactProducer(plugin_id="test.raw", plugin_version="1.0.0"),
        outputs={
            "primary": {
                "contract_id": RAW_MOLECULE_V1.id,
                "file_paths": (raw_path,),
                "metadata": {"row_count": 2},
            }
        },
    )
    raw_ref = raw_manifest.dataset_ref("primary")
    raw_root = store.resolve_dataset(raw_ref)

    standardizer = RDKitStandardizePlugin()
    standardize_stage = store.begin_staging()
    standardize_response = standardizer.execute(
        StageRequest(
            stage_id="standardize",
            inputs={"molecules": StageInput(ref=raw_ref, root=raw_root)},
            config={"batch_size": 1},
        ),
        StageContext(standardize_stage),
    )
    standardized = store.commit(
        standardize_stage,
        kind="stage-output",
        producer=_producer(standardizer),
        inputs=(raw_ref,),
        outputs=standardize_response.artifact_outputs,
        metadata=standardize_response.metadata,
    )

    assert {output.port: output.contract_id for output in standardized.outputs} == {
        "decisions": DECISION_V1.id,
        "mapping": PARENT_SOURCE_MAP_V1.id,
        "primary": PARENT_V1.id,
    }
    standardized_root = store.artifact_directory(standardized.artifact_id, verify=True)
    port_inputs = {
        port: StageInput(ref=standardized.dataset_ref(port), root=standardized_root)
        for port in ("primary", "mapping", "decisions")
    }
    assert port_inputs["primary"].file_paths == (
        "datasets/parents/part-00000.parquet",
    )
    assert port_inputs["mapping"].contract_id == PARENT_SOURCE_MAP_V1.id
    assert port_inputs["decisions"].contract_id == DECISION_V1.id

    hard_gate = RDKitHardGatePlugin()
    hard_gate_stage = store.begin_staging()
    hard_gate_response = hard_gate.execute(
        StageRequest(
            stage_id="hard_gate",
            inputs={"parents": port_inputs["primary"]},
            config={"batch_size": 1, "policy": {"flag_pains": False}},
        ),
        StageContext(hard_gate_stage),
    )
    gated = store.commit(
        hard_gate_stage,
        kind="stage-output",
        producer=_producer(hard_gate),
        inputs=(standardized.dataset_ref("primary"),),
        outputs=hard_gate_response.artifact_outputs,
        metadata=hard_gate_response.metadata,
    )

    assert gated.dataset_ref("primary").contract_id == PARENT_V1.id
    assert gated.dataset_ref("decisions").contract_id == DECISION_V1.id
    assert store.verify(gated.artifact_id) == gated
