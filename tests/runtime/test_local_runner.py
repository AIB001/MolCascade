from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import (
    ArtifactDatasetRef,
    ArtifactIntegrityError,
    InvalidStagingError,
    canonical_json_bytes,
)
from molcascade.config.models import PipelineConfig
from molcascade.contracts import DECISION_V1, PARENT_V1, RAW_MOLECULE_V1
from molcascade.errors import ContractError, ExecutionError, PipelineError, PluginError
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PendingOutput,
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    StageContext,
    StageRequest,
    StageResponse,
)
from molcascade.runtime import (
    AuditLog,
    LocalRunner,
    RunStatus,
    StageRunStatus,
    stage_cache_key,
)


def _raw_row(
    source_id: str,
    index: int,
    *,
    smiles: str = "CCO",
) -> dict[str, object]:
    return {
        "source_record_id": source_id,
        "source_kind": "test",
        "source_uri": None,
        "source_index": index,
        "generator_id": "fake-source",
        "batch_id": "batch-a",
        "source_candidate_id": source_id,
        "raw_smiles": smiles,
        "raw_molblock": None,
        "source_metadata_json": None,
    }


class FakeSourcePlugin:
    descriptor = PluginDescriptor(
        id="test.source",
        version="1.0.0",
        kind=PluginKind.SOURCE,
        outputs=(RAW_MOLECULE_V1.id,),
        cardinality=Cardinality.ONE_TO_MANY,
        determinism=Determinism.DETERMINISTIC,
    )

    def __init__(self) -> None:
        self.calls = 0
        self.smiles = "CCO"

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        assert not request.inputs
        self.calls += 1
        relative = "datasets/raw/part-00000.parquet"
        destination = context.staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [
                _raw_row("source-a", 0, smiles=self.smiles),
                _raw_row("source-b", 1, smiles=self.smiles),
            ],
            schema=RAW_MOLECULE_V1.schema,
        )
        pq.write_table(table, destination)
        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    contract_id=RAW_MOLECULE_V1.id,
                    file_paths=(relative,),
                    metadata={"row_count": table.num_rows},
                )
            },
            metadata={"source": "fake"},
        )


class FakeGatePlugin:
    descriptor = PluginDescriptor(
        id="test.gate",
        version="1.0.0",
        kind=PluginKind.GATE,
        inputs=(RAW_MOLECULE_V1.id,),
        outputs=(RAW_MOLECULE_V1.id, DECISION_V1.id),
        output_ports={"primary": RAW_MOLECULE_V1.id, "decisions": DECISION_V1.id},
        cardinality=Cardinality.FILTER,
        determinism=Determinism.DETERMINISTIC,
    )

    def __init__(self, *, mode: str = "normal") -> None:
        self.calls = 0
        self.mode = mode
        self.fail = False
        self.interrupt = False
        self.failure_message = "intentional plugin failure"
        self.hardlink_source: Path | None = None
        self.raised_error: Exception | None = None

    @staticmethod
    def _input_table(request: StageRequest) -> pa.Table:
        stage_input = next(iter(request.inputs.values()))
        return pa.concat_tables(
            [pq.read_table(stage_input.root / path) for path in stage_input.file_paths]
        )

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        self.calls += 1
        if self.interrupt:
            raise KeyboardInterrupt("simulated process interruption")
        if self.raised_error is not None:
            raise self.raised_error
        if self.fail:
            context.staging_root.mkdir(parents=True, exist_ok=True)
            (context.staging_root / "partial.tmp").write_text("partial", encoding="utf-8")
            raise RuntimeError(self.failure_message)

        table = self._input_table(request)
        context.staging_root.mkdir(parents=True, exist_ok=True)
        if self.mode == "bad_schema":
            path = "datasets/primary/part-00000.parquet"
            wrong = pa.Table.from_pylist(
                [
                    {
                        "parent_id": "parent-a",
                        "identity_policy_id": "policy-a",
                        "parent_smiles": "CCO",
                        "registration_key": "key-a",
                        "stereo_key": None,
                        "formula": "C2H6O",
                        "duplicate_count": 1,
                    }
                ],
                schema=PARENT_V1.schema,
            )
            destination = context.staging_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(wrong, destination)
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(path,),
                    )
                }
            )

        if self.mode == "duplicate_across_files":
            paths = (
                "datasets/primary/part-00000.parquet",
                "datasets/primary/part-00001.parquet",
            )
            duplicate = table.slice(0, 1)
            for path in paths:
                destination = context.staging_root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(duplicate, destination)
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=paths,
                    )
                }
            )

        if self.mode == "external_hardlink":
            assert self.hardlink_source is not None
            self.hardlink_source.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, self.hardlink_source)
            path = "datasets/primary/part-00000.parquet"
            destination = context.staging_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(self.hardlink_source, destination)
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(path,),
                    )
                }
            )

        primary_path = "datasets/primary/part-00000.parquet"
        primary_destination = context.staging_root / primary_path
        primary_destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, primary_destination)

        if self.mode == "undeclared_contract":
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=PARENT_V1.id,
                        file_paths=(primary_path,),
                    )
                }
            )
        if self.mode == "missing_primary":
            return StageResponse(
                outputs={
                    "other": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(primary_path,),
                    )
                }
            )
        if self.mode == "duplicate_contract_port":
            extra_path = "datasets/extra/part-00000.parquet"
            extra_destination = context.staging_root / extra_path
            extra_destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, extra_destination)
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(primary_path,),
                    ),
                    "extra": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(extra_path,),
                    ),
                }
            )
        if self.mode == "missing_side":
            return StageResponse(
                outputs={
                    "primary": PendingOutput(
                        contract_id=RAW_MOLECULE_V1.id,
                        file_paths=(primary_path,),
                    )
                }
            )
        if self.mode == "undeclared_file":
            (context.staging_root / "undeclared.txt").write_text("extra", encoding="utf-8")

        decision_path = "datasets/decisions/part-00000.parquet"
        decisions = pa.Table.from_pylist(
            [
                {
                    "entity_id": row["source_record_id"],
                    "entity_kind": "SOURCE_RECORD",
                    "stage_id": request.stage_id,
                    "outcome": "PASS",
                    "reason_code": "PASSED",
                    "rule_id": None,
                    "detail": None,
                }
                for row in table.to_pylist()
            ],
            schema=DECISION_V1.schema,
        )
        decision_destination = context.staging_root / decision_path
        decision_destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(decisions, decision_destination)
        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    contract_id=RAW_MOLECULE_V1.id,
                    file_paths=(primary_path,),
                    metadata={"row_count": table.num_rows},
                ),
                "decisions": PendingOutput(
                    contract_id=DECISION_V1.id,
                    file_paths=(decision_path,),
                    metadata={"row_count": decisions.num_rows},
                ),
            },
            metadata={"passed": table.num_rows},
        )


class FakeSelectorPlugin:
    descriptor = PluginDescriptor(
        id="test.selector",
        version="1.0.0",
        kind=PluginKind.SELECTOR,
        inputs=(RAW_MOLECULE_V1.id, DECISION_V1.id),
        outputs=(RAW_MOLECULE_V1.id,),
        cardinality=Cardinality.FILTER,
        determinism=Determinism.DETERMINISTIC,
    )

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        self.calls += 1
        assert set(request.inputs) == {"decisions", "molecules"}
        assert request.inputs["molecules"].contract_id == RAW_MOLECULE_V1.id
        assert request.inputs["decisions"].contract_id == DECISION_V1.id
        decision_rows = sum(
            pq.read_table(request.inputs["decisions"].root / path).num_rows
            for path in request.inputs["decisions"].file_paths
        )
        molecules = pa.concat_tables(
            [
                pq.read_table(request.inputs["molecules"].root / path)
                for path in request.inputs["molecules"].file_paths
            ]
        )
        assert decision_rows == molecules.num_rows
        relative = "datasets/selected/part-00000.parquet"
        destination = context.staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(molecules, destination)
        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    contract_id=RAW_MOLECULE_V1.id,
                    file_paths=(relative,),
                    metadata={"row_count": molecules.num_rows},
                )
            }
        )


class FakeSeededGatePlugin(FakeGatePlugin):
    descriptor = FakeGatePlugin.descriptor.model_copy(
        update={
            "id": "test.seeded_gate",
            "determinism": Determinism.SEEDED,
        }
    )


class BlockingResumeGatePlugin(FakeGatePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        if self.block:
            self.entered.set()
            if not self.release.wait(timeout=10):
                raise RuntimeError("test release timeout")
        return super().execute(request, context)


def _pipeline(*, seed: int = 7, name: str = "runtime-test") -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": name,
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "test.source@1.0.0",
                    "config": {"seed": seed},
                },
                {
                    "id": "gate",
                    "slot": "hard_gate",
                    "plugin": "test.gate@1.0.0",
                    "config": {},
                },
            ],
        }
    )


def _retained_context_pipeline() -> PipelineConfig:
    payload = _pipeline().model_dump(mode="json")
    payload["stages"].append(
        {
            "id": "selector",
            "slot": "select",
            "plugin": "test.selector@1.0.0",
            "inputs": [
                {
                    "request_port": "molecules",
                    "stage": "gate",
                    "port": "primary",
                },
                {
                    "request_port": "decisions",
                    "stage": "gate",
                    "port": "decisions",
                },
            ],
            "config": {},
        }
    )
    return PipelineConfig.model_validate(payload)


def _runner(
    tmp_path: Path,
    gate: FakeGatePlugin | None = None,
) -> tuple[LocalRunner, FakeSourcePlugin, FakeGatePlugin]:
    source = FakeSourcePlugin()
    selected_gate = FakeGatePlugin() if gate is None else gate
    registry = PluginRegistry([source, selected_gate])
    return LocalRunner(tmp_path / "workspace", plugins=registry), source, selected_gate


@pytest.mark.parametrize(
    "child",
    ["revisions", "runs", "events", "cache", ".run-locks"],
)
def test_runner_rejects_symlinked_control_plane_roots(
    tmp_path: Path,
    child: str,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / child).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        LocalRunner(
            workspace,
            plugins=PluginRegistry([FakeSourcePlugin(), FakeGatePlugin()]),
        )


def test_success_persists_revision_state_audit_and_multi_output_artifact(
    tmp_path: Path,
) -> None:
    runner, source, gate = _runner(tmp_path)

    result = runner.run(_pipeline(), run_id="run-success")

    assert result.status is RunStatus.SUCCEEDED
    assert result.output_ref is not None
    assert result.output_ref.port == "primary"
    assert source.calls == gate.calls == 1
    manifest = runner.store.verify(result.output_ref.artifact_id)
    assert {output.port for output in manifest.outputs} == {"primary", "decisions"}
    assert (runner.workspace / "revisions" / f"{result.revision_id}.json").is_file()
    assert (runner.workspace / "runs" / "run-success.json").is_file()
    events = runner.read_events("run-success")
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event_type == "RUN_STARTED"
    assert events[-1].event_type == "RUN_SUCCEEDED"
    for line in (runner.workspace / "events" / "run-success.jsonl").read_text().splitlines():
        assert isinstance(json.loads(line), dict)


def test_permitted_copyleft_backends_are_named_in_the_audit_trail(tmp_path: Path) -> None:
    """A run that linked GPL code has to be able to say which tool, under what.

    Recording the flag instead of the tools would leave a methods section with
    nothing to write: '--allow-copyleft was passed' does not tell a reader that
    the poses were rescored by GNINA under GPL-2.0-or-later.  A run that permits
    nothing records an empty mapping rather than omitting the key, so the
    absence is a positive statement too.
    """

    registry = PluginRegistry([FakeSourcePlugin(), FakeGatePlugin()])
    runner = LocalRunner(
        tmp_path / "workspace",
        plugins=registry,
        permitted_licenses={"gnina": "GPL-2.0-or-later"},
    )

    result = runner.run(_pipeline(), run_id="run-licensed")

    assert result.status is RunStatus.SUCCEEDED
    started = runner.read_events("run-licensed")[0]
    assert started.event_type == "RUN_STARTED"
    assert started.details["permitted_licenses"] == {"gnina": "GPL-2.0-or-later"}

    default_runner, _source, _gate = _runner(tmp_path / "default")
    default_runner.run(_pipeline(), run_id="run-unlicensed")
    assert default_runner.read_events("run-unlicensed")[0].details["permitted_licenses"] == {}


def test_cache_hit_force_and_cache_key_are_deterministic(tmp_path: Path) -> None:
    runner, source, gate = _runner(tmp_path)
    first = runner.run(_pipeline(), run_id="run-one")
    calls_after_first = (source.calls, gate.calls)

    cached = runner.run(_pipeline(), run_id="run-two")

    # Sources may dereference mutable external paths, so they are safely
    # re-snapshotted until the plugin API exposes pre-run content fingerprints.
    # The unchanged content-addressed source artifact still lets downstream
    # stages hit their cache.
    assert (source.calls, gate.calls) == (calls_after_first[0] + 1, calls_after_first[1])
    assert cached.stages[0].status is StageRunStatus.SUCCEEDED
    assert cached.stages[1].status is StageRunStatus.CACHED
    assert cached.output_ref == first.output_ref

    forced = runner.run(_pipeline(), run_id="run-three", force=True)
    assert (source.calls, gate.calls) == (calls_after_first[0] + 2, calls_after_first[1] + 1)
    assert all(stage.status is StageRunStatus.SUCCEEDED for stage in forced.stages)
    assert forced.output_ref == first.output_ref

    compiled = runner.compile(_pipeline())
    primary = first.output_ref
    assert primary is not None
    assert stage_cache_key(compiled.stages[-1], primary) == stage_cache_key(
        compiled.stages[-1], primary
    )
    mapping_view = primary.model_copy(
        update={
            "port": "mapping",
            "contract_id": "parent_source_map/v1",
            "file_paths": ("datasets/mapping/part.parquet",),
        }
    )
    assert stage_cache_key(compiled.stages[-1], primary) != stage_cache_key(
        compiled.stages[-1], mapping_view
    )


def test_source_is_resnapshotted_so_changed_external_content_is_not_stale(
    tmp_path: Path,
) -> None:
    runner, source, gate = _runner(tmp_path)
    first = runner.run(_pipeline(), run_id="source-before-change")

    source.smiles = "CCN"
    changed = runner.run(_pipeline(), run_id="source-after-change")

    assert source.calls == gate.calls == 2
    assert first.output_ref is not None
    assert changed.output_ref is not None
    assert changed.output_ref.artifact_id != first.output_ref.artifact_id
    assert changed.stages[0].status is StageRunStatus.SUCCEEDED
    assert changed.stages[1].status is StageRunStatus.SUCCEEDED


def test_seeded_plugin_with_null_seed_is_not_cached_across_runs(tmp_path: Path) -> None:
    source = FakeSourcePlugin()
    gate = FakeSeededGatePlugin()
    runner = LocalRunner(
        tmp_path / "workspace",
        plugins=PluginRegistry([source, gate]),
    )
    payload = _pipeline().model_dump(mode="json")
    payload["stages"][1]["plugin"] = gate.descriptor.key
    payload["stages"][1]["config"] = {"seed": None}
    pipeline = PipelineConfig.model_validate(payload)

    runner.run(pipeline, run_id="null-seed-one")
    second = runner.run(pipeline, run_id="null-seed-two")

    assert gate.calls == 2
    assert second.stages[-1].status is StageRunStatus.SUCCEEDED


def test_cache_hit_verifies_artifact_and_fails_closed_on_tamper(tmp_path: Path) -> None:
    runner, _, _ = _runner(tmp_path)
    first = runner.run(_pipeline(), run_id="run-original")
    assert first.output_ref is not None
    committed = runner.store.resolve_file(
        first.output_ref.artifact_id,
        first.output_ref.file_paths[0],
    )
    committed.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError):
        runner.run(_pipeline(), run_id="run-after-tamper")

    assert runner.load_run("run-after-tamper").status is RunStatus.FAILED


def test_invocation_key_is_bound_into_artifact_identity_metadata(tmp_path: Path) -> None:
    runner, _, _ = _runner(tmp_path)
    result = runner.run(_pipeline(), run_id="identity-key")
    source_ref = result.stages[0].output_ref
    output_ref = result.stages[1].output_ref
    assert source_ref is not None
    assert output_ref is not None

    forged_key = "cache:sha256:" + "f" * 64
    assert result.stages[1].cache_key != forged_key
    manifest_path = (
        runner.store.artifact_directory(output_ref.artifact_id) / "artifact.json"
    )
    payload = json.loads(manifest_path.read_bytes())
    payload["cache_key"] = forged_key
    manifest_path.write_bytes(canonical_json_bytes(payload))

    # The observational field is deliberately outside artifact identity.
    runner.store.verify(output_ref.artifact_id)
    compiled = runner.compile(_pipeline())
    with pytest.raises(ArtifactIntegrityError) as caught:
        runner._verify_checkpoint(
            compiled.stages[1],
            forged_key,
            output_ref,
            {"primary": source_ref},
        )

    assert "metadata.invocation_cache_key" in caught.value.context["mismatched_fields"]


def test_failed_stage_cleans_staging_and_resume_uses_verified_checkpoint(
    tmp_path: Path,
) -> None:
    gate = FakeGatePlugin()
    gate.fail = True
    runner, source, _ = _runner(tmp_path, gate)

    with pytest.raises(ExecutionError, match="intentional plugin failure"):
        runner.run(_pipeline(), run_id="run-resume")

    failed = runner.load_run("run-resume")
    assert failed.status is RunStatus.FAILED
    assert failed.stages[0].status is StageRunStatus.SUCCEEDED
    assert failed.stages[1].status is StageRunStatus.FAILED
    assert list(runner.store.staging_root.iterdir()) == []
    assert failed.stages[0].output_ref is not None
    runner.store.verify(failed.stages[0].output_ref.artifact_id)

    gate.fail = False
    resumed = runner.run(_pipeline(), run_id="run-resume", resume=True)

    assert resumed.status is RunStatus.SUCCEEDED
    assert source.calls == 1
    assert gate.calls == 2
    event_types = [event.event_type for event in runner.read_events("run-resume")]
    assert "RUN_FAILED" in event_types
    assert "RUN_RESUMED" in event_types
    assert "STAGE_RESUMED" in event_types


def test_resume_rejects_a_different_revision(tmp_path: Path) -> None:
    gate = FakeGatePlugin()
    gate.fail = True
    runner, _, _ = _runner(tmp_path, gate)
    with pytest.raises(ExecutionError):
        runner.run(_pipeline(), run_id="same-run")
    original = runner.load_run("same-run")

    with pytest.raises(PipelineError, match="revision"):
        runner.run(
            _pipeline(name="different-revision"),
            run_id="same-run",
            resume=True,
        )

    assert runner.load_run("same-run") == original


def test_revision_mismatch_does_not_destroy_a_successful_run(tmp_path: Path) -> None:
    runner, _, _ = _runner(tmp_path)
    original = runner.run(_pipeline(), run_id="successful-run")

    with pytest.raises(PipelineError) as caught:
        runner.run(
            _pipeline(name="wrong-successful-revision"),
            run_id="successful-run",
            resume=True,
        )

    assert caught.value.code == "RUN_REVISION_MISMATCH"
    assert runner.load_run("successful-run") == original


def test_same_revision_resume_invalidates_a_corrupted_successful_run(
    tmp_path: Path,
) -> None:
    runner, _, _ = _runner(tmp_path)
    succeeded = runner.run(_pipeline(), run_id="corrupted-success")
    assert succeeded.output_ref is not None
    output_file = runner.store.resolve_file(
        succeeded.output_ref.artifact_id,
        succeeded.output_ref.file_paths[0],
    )
    output_file.write_bytes(b"corrupted after success")

    with pytest.raises(ArtifactIntegrityError):
        runner.run(_pipeline(), run_id="corrupted-success", resume=True)

    invalidated = runner.load_run("corrupted-success")
    assert invalidated.status is RunStatus.FAILED
    assert invalidated.output_ref is None
    assert invalidated.error is not None
    assert invalidated.error.category.value == "ARTIFACT_INTEGRITY"


def test_resume_integrity_failure_marks_crash_left_run_failed(tmp_path: Path) -> None:
    gate = FakeGatePlugin()
    gate.interrupt = True
    runner, _, _ = _runner(tmp_path, gate)

    with pytest.raises(KeyboardInterrupt):
        runner.run(_pipeline(), run_id="crash-tamper-resume")

    crashed = runner.load_run("crash-tamper-resume")
    assert crashed.status is RunStatus.RUNNING
    source_ref = crashed.stages[0].output_ref
    assert source_ref is not None
    source_file = runner.store.resolve_file(
        source_ref.artifact_id,
        source_ref.file_paths[0],
    )
    source_file.write_bytes(b"tampered checkpoint")

    gate.interrupt = False
    with pytest.raises(ArtifactIntegrityError):
        runner.run(_pipeline(), run_id="crash-tamper-resume", resume=True)

    failed = runner.load_run("crash-tamper-resume")
    assert failed.status is RunStatus.FAILED
    assert all(stage.status is not StageRunStatus.RUNNING for stage in failed.stages)


@pytest.mark.parametrize(
    ("mode", "error_type", "error_code"),
    [
        ("undeclared_contract", PluginError, "PLUGIN_OUTPUT_CONTRACT_UNDECLARED"),
        ("missing_primary", PluginError, "PLUGIN_PRIMARY_OUTPUT_MISSING"),
        ("duplicate_contract_port", PluginError, "PLUGIN_OUTPUT_CONTRACT_DUPLICATE"),
        ("missing_side", PluginError, "PLUGIN_OUTPUT_PORT_MISSING"),
        ("bad_schema", ContractError, "CONTRACT_SCHEMA_MISMATCH"),
        ("duplicate_across_files", ContractError, "CONTRACT_PRIMARY_KEY_DUPLICATE"),
        ("undeclared_file", InvalidStagingError, "ARTIFACT_STAGING_INVALID"),
    ],
)
def test_invalid_plugin_outputs_fail_without_publishing_stage_artifact(
    tmp_path: Path,
    mode: str,
    error_type: type[Exception],
    error_code: str,
) -> None:
    runner, _, _ = _runner(tmp_path, FakeGatePlugin(mode=mode))

    with pytest.raises(error_type) as caught:
        runner.run(_pipeline(), run_id=f"run-{mode}", force=True)

    assert caught.value.code == error_code
    state = runner.load_run(f"run-{mode}")
    assert state.status is RunStatus.FAILED
    assert state.stages[-1].status is StageRunStatus.FAILED
    assert list(runner.store.staging_root.iterdir()) == []
    committed = runner.store.list_refs(verify=True)
    assert len(committed) == 1


def test_hard_linked_output_is_rejected_before_commit(tmp_path: Path) -> None:
    gate = FakeGatePlugin(mode="external_hardlink")
    gate.hardlink_source = tmp_path / "outside" / "mutable.parquet"
    runner, _, _ = _runner(tmp_path, gate)

    with pytest.raises(PluginError) as caught:
        runner.run(_pipeline(), run_id="hardlink-output")

    assert caught.value.code == "PLUGIN_OUTPUT_FILE_INVALID"
    assert "hard-linked" in caught.value.message
    assert runner.load_run("hardlink-output").status is RunStatus.FAILED
    # Only the successfully committed source artifact may exist.
    assert len(runner.store.list_refs(verify=True)) == 1


def test_force_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    runner, _, _ = _runner(tmp_path)

    with pytest.raises(PipelineError, match="force"):
        runner.run(_pipeline(), run_id="invalid", force=True, resume=True)


def test_initial_audit_failure_persists_failed_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _runner(tmp_path)

    def fail_append(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("audit unavailable")

    monkeypatch.setattr(AuditLog, "append", fail_append)
    with pytest.raises(ExecutionError, match="audit unavailable") as caught:
        runner.run(_pipeline(), run_id="audit-start-failure")

    assert caught.value.code == "RUNTIME_CONTROL_PLANE_FAILED"
    state = runner.load_run("audit-start-failure")
    assert state.status is RunStatus.FAILED
    assert all(stage.status is StageRunStatus.PENDING for stage in state.stages)
    assert state.error is not None
    assert state.error.code == "RUNTIME_CONTROL_PLANE_FAILED"


def test_resume_audit_failure_does_not_leave_run_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = FakeGatePlugin()
    gate.fail = True
    runner, _, _ = _runner(tmp_path, gate)
    with pytest.raises(ExecutionError):
        runner.run(_pipeline(), run_id="audit-resume-failure")

    def fail_append(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("resume audit unavailable")

    gate.fail = False
    monkeypatch.setattr(AuditLog, "append", fail_append)
    with pytest.raises(ExecutionError, match="resume audit unavailable"):
        runner.run(_pipeline(), run_id="audit-resume-failure", resume=True)

    state = runner.load_run("audit-resume-failure")
    assert state.status is RunStatus.FAILED
    assert state.stages[0].status is StageRunStatus.SUCCEEDED
    assert state.stages[1].status is StageRunStatus.PENDING


def test_cross_instance_run_lock_rejects_concurrent_resume(tmp_path: Path) -> None:
    gate = BlockingResumeGatePlugin()
    gate.fail = True
    source = FakeSourcePlugin()
    registry = PluginRegistry([source, gate])
    workspace = tmp_path / "workspace"
    first_runner = LocalRunner(workspace, plugins=registry)
    second_runner = LocalRunner(workspace, plugins=registry)
    with pytest.raises(ExecutionError):
        first_runner.run(_pipeline(), run_id="concurrent-resume")

    gate.fail = False
    gate.block = True
    outcome: dict[str, object] = {}

    def resume_first() -> None:
        try:
            outcome["result"] = first_runner.run(
                _pipeline(),
                run_id="concurrent-resume",
                resume=True,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            outcome["error"] = error

    worker = threading.Thread(target=resume_first)
    worker.start()
    assert gate.entered.wait(timeout=10)
    try:
        with pytest.raises(ExecutionError) as caught:
            second_runner.run(
                _pipeline(),
                run_id="concurrent-resume",
                resume=True,
            )
        assert caught.value.code == "RUN_LOCKED"
        assert caught.value.retryable
    finally:
        gate.release.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert result.status is RunStatus.SUCCEEDED
    events = first_runner.read_events("concurrent-resume")
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))


def test_oversized_plugin_exception_is_bounded_and_run_remains_loadable(
    tmp_path: Path,
) -> None:
    gate = FakeGatePlugin()
    gate.fail = True
    gate.failure_message = "sensitive-marker-" + ("x" * (17 * 1024 * 1024))
    runner, _, _ = _runner(tmp_path, gate)

    with pytest.raises(ExecutionError) as caught:
        runner.run(_pipeline(), run_id="bounded-plugin-error")

    assert len(caught.value.message) < 4_096
    assert "truncated" in caught.value.message
    state_path = runner.workspace / "runs" / "bounded-plugin-error.json"
    assert state_path.stat().st_size < 64 * 1024
    state = runner.load_run("bounded-plugin-error")
    assert state.status is RunStatus.FAILED
    assert state.error is not None
    assert len(state.error.message) < 4_096


def test_unprintable_plugin_exception_still_persists_failed_state(
    tmp_path: Path,
) -> None:
    class UnprintableError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("rendering must not escape")

    gate = FakeGatePlugin()
    gate.raised_error = UnprintableError()
    runner, _, _ = _runner(tmp_path, gate)

    with pytest.raises(ExecutionError) as caught:
        runner.run(_pipeline(), run_id="unprintable-error")

    assert "<unprintable UnprintableError>" in caught.value.message
    state = runner.load_run("unprintable-error")
    assert state.status is RunStatus.FAILED
    assert state.stages[-1].status is StageRunStatus.FAILED


def test_mutated_structured_exception_still_persists_failed_state(
    tmp_path: Path,
) -> None:
    gate = FakeGatePlugin()
    corrupted = PluginError("initially valid")
    corrupted.context["not_json"] = object()  # type: ignore[assignment]
    gate.raised_error = corrupted
    runner, _, _ = _runner(tmp_path, gate)

    with pytest.raises(ExecutionError) as caught:
        runner.run(_pipeline(), run_id="corrupted-structured-error")

    assert caught.value.code == "RUNTIME_ERROR_SERIALIZATION_FAILED"
    state = runner.load_run("corrupted-structured-error")
    assert state.status is RunStatus.FAILED
    assert state.error is not None
    assert state.error.code == "RUNTIME_ERROR_SERIALIZATION_FAILED"


def test_stage_cache_key_has_no_workspace_or_time_component(tmp_path: Path) -> None:
    runner, _, _ = _runner(tmp_path)
    compiled = runner.compile(_pipeline(seed=9))
    ref = ArtifactDatasetRef(
        artifact_id="artifact:sha256:" + "a" * 64,
        artifact_kind="stage-output/source",
        port="primary",
        contract_id=RAW_MOLECULE_V1.id,
        file_paths=("datasets/raw/part.parquet",),
    )

    first = stage_cache_key(compiled.stages[-1], ref)
    second = stage_cache_key(compiled.stages[-1], ref)

    assert first == second
    assert str(tmp_path) not in first


def test_explicit_bindings_retain_side_outputs_across_cache_hits(tmp_path: Path) -> None:
    source = FakeSourcePlugin()
    gate = FakeGatePlugin()
    selector = FakeSelectorPlugin()
    registry = PluginRegistry([source, gate, selector])
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)

    first = runner.run(_retained_context_pipeline(), run_id="retained-first")
    cached = runner.run(_retained_context_pipeline(), run_id="retained-cached")

    assert first.status is RunStatus.SUCCEEDED
    assert cached.stages[0].status is StageRunStatus.SUCCEEDED
    assert all(
        stage.status is StageRunStatus.CACHED for stage in cached.stages[1:]
    )
    assert (source.calls, gate.calls, selector.calls) == (2, 1, 1)
    selector_state = first.stages[-1]
    assert selector_state.output_ref is not None
    manifest = runner.store.verify(selector_state.output_ref.artifact_id)
    assert [(ref.port, ref.contract_id) for ref in manifest.inputs] == [
        ("decisions", DECISION_V1.id),
        ("primary", RAW_MOLECULE_V1.id),
    ]


def test_the_run_records_the_machine_it_happened_on(tmp_path: Path) -> None:
    """Provenance without an environment is provenance missing the interesting part.

    A run already measures this machine -- lane planning calls
    ``detect_environment``, which forks ``nvidia-smi`` -- and used to keep the
    worker count and the batch size and drop the rest. That matters because
    ``stage_cache_key`` hashes no third-party library version: the same
    configuration under a different RDKit is the same cache entry and a
    different answer, and nothing recorded which RDKit it was.

    Written to the audit trail rather than to a manifest, following the
    ``permitted_licenses`` precedent immediately above: ``AuditEvent.details`` is
    a free-shape mapping that enters no hash, so this costs no contract and
    leaves every run already on disk readable.
    """

    from molcascade.environment import detect_environment

    registry = PluginRegistry([FakeSourcePlugin(), FakeGatePlugin()])
    runner = LocalRunner(
        tmp_path / "workspace",
        plugins=registry,
        environment=detect_environment(),
    )

    result = runner.run(_pipeline(), run_id="run-provenance")

    assert result.status is RunStatus.SUCCEEDED
    started = runner.read_events("run-provenance")[0]
    assert started.event_type == "RUN_STARTED"
    assert started.details["molcascade_version"]
    recorded = started.details["environment"]
    assert isinstance(recorded, dict)
    assert recorded["platform"]["python_version"]
    assert "usable_cores" in recorded["cpu"]
    # GpuDevice excludes UUID and serial because this record is shared; that
    # reasoning does not stop applying when it reaches an audit log.
    for gpu in recorded["gpus"]:
        assert "uuid" not in gpu
        assert "serial" not in gpu


def test_a_runner_given_no_environment_measures_none(tmp_path: Path) -> None:
    """A runner built without one must not fork a subprocess to inventory a
    machine nobody asked about, so absent is recorded as absent -- while the
    package version, which costs nothing to know, is recorded either way."""

    runner, _source, _gate = _runner(tmp_path)
    runner.run(_pipeline(), run_id="run-bare")

    details = runner.read_events("run-bare")[0].details
    assert details["environment"] is None
    assert details["molcascade_version"]
