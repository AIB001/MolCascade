from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures.offline_html import assert_offline

from molcascade.artifacts import canonical_json_bytes
from molcascade.cli import _molecule_override, main
from molcascade.config import PipelineConfig
from molcascade.errors import PipelineError
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import AuditEvent, RunState


def _pipeline_document(path: str = "configured-input.csv") -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "cli-test",
        "stages": [
            {
                "id": "source",
                "slot": "source",
                "plugin": "source.delimited_smiles@0.1.0",
                "config": {
                    "path": path,
                    "delimiter": ",",
                    "has_header": True,
                    "smiles_column": "smiles",
                    "candidate_id_column": "id",
                    "metadata_columns": [],
                },
                "enabled": True,
            }
        ],
    }


def _write_pipeline(tmp_path: Path, *, path: str = "configured-input.csv") -> Path:
    destination = tmp_path / "pipeline.json"
    destination.write_text(
        json.dumps(_pipeline_document(path), ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def _write_model_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    graph = b"not a real graph, but the digest does not care"
    (root / "model.onnx").write_bytes(graph)
    (root / "molcascade_model.yaml").write_text(
        "schema_version: 1\n"
        "model_name: cli-demo\n"
        "endpoint_id: custom/logS\n"
        "task: regression\n"
        "representation:\n"
        "  blocks:\n"
        "    - kind: morgan\n"
        "      radius: 2\n"
        "      n_bits: 64\n"
        "model:\n"
        "  file: model.onnx\n"
        f"  sha256: {hashlib.sha256(graph).hexdigest()}\n"
        "  n_features: 64\n"
        "  dtype: float32\n",
        encoding="utf-8",
    )
    return root


def _write_molecules(tmp_path: Path) -> Path:
    destination = tmp_path / "override.csv"
    destination.write_text("id,smiles\nmol-1,CCO\nmol-2,CCN\n", encoding="utf-8")
    return destination


def _write_handoff_pipeline(tmp_path: Path) -> Path:
    molecules = _write_molecules(tmp_path)
    destination = tmp_path / "handoff-pipeline.json"
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "cli-handoff",
                "stages": [
                    {
                        "id": "source",
                        "slot": "source",
                        "plugin": "source.delimited_smiles@0.1.0",
                        "config": {
                            "path": str(molecules),
                            "smiles_column": "smiles",
                            "candidate_id_column": "id",
                        },
                    },
                    {
                        "id": "standardize",
                        "slot": "standardize",
                        "plugin": "chemistry.rdkit_standardize@0.1.0",
                        "config": {},
                    },
                    {
                        "id": "shortlist",
                        "slot": "export",
                        "plugin": "export.native_smiles_shortlist@0.1.0",
                        "config": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return destination


@pytest.mark.parametrize(
    "arguments",
    [
        ["--generate", "config"],
        ["generate", "config"],
    ],
)
def test_generate_config_compatibility_and_canonical_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "builder.html"

    exit_code = main([*arguments, "--output", str(output)])

    assert exit_code == 0
    assert output.is_file()
    document = output.read_text(encoding="utf-8")
    # ``generate config`` opens the tier-first cascade builder.  The flat graph
    # editor is still reachable, but it is a different authoring surface (it
    # writes a schema-1 pipeline directly), not an older skin of this one.
    assert "MolCascade screening builder" in document
    assert_offline(document)
    printed = capsys.readouterr().out
    assert str(output) in printed
    # A first-time user has to be told what to do with the file they just got.
    assert "molcascade screen --config" in printed


def test_generate_config_pipeline_flag_still_writes_the_graph_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "graph.html"

    exit_code = main(["generate", "config", "--pipeline", "--output", str(output)])

    assert exit_code == 0
    document = output.read_text(encoding="utf-8")
    assert "MolCascade Pipeline Builder" in document
    assert_offline(document)
    assert str(output) in capsys.readouterr().out


def test_deprecated_generate_alias_stays_frozen_at_output_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--generate config`` is a compatibility shim, so it does not grow flags.

    New surface belongs on the canonical ``generate config`` subcommand; adding
    it to the alias too would make a deprecated form look maintained.
    """

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["--generate", "config", "--pipeline"])


def test_generate_config_uses_documented_default_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--generate", "config"]) == 0

    assert (tmp_path / "molcascade-config.html").is_file()


def test_model_bundle_prints_the_digest_the_builder_asks_users_to_paste(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The custom-model criterion tells users to run this exact command.

    Both the builder's help text and the plugin's hash-mismatch hint name
    ``molcascade model-bundle``, so the two settings it prints have to be the
    two settings the criterion asks for -- otherwise the instructions send
    people looking for a field that is not there.
    """

    bundle = _write_model_bundle(tmp_path)

    assert main(["model-bundle", str(bundle)]) == 0

    printed = capsys.readouterr().out
    assert "bundle_dir: " + str(bundle) in printed
    assert "expected_bundle_sha256: " in printed

    assert main(["model-bundle", str(bundle), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["bundle_sha256"] in printed
    assert report["model_id"].startswith("cli-demo:sha256:")
    # Reading a bundle must not need the runtime that executes it, so this
    # passes on a machine where onnxruntime was never installed.
    assert report["file_count"] == 2


def test_model_bundle_refuses_a_directory_that_is_not_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["model-bundle", str(tmp_path / "absent"), "--json"]) == 1
    reported = json.loads(capsys.readouterr().err)
    assert reported["ok"] is False
    assert reported["error"]["code"] == "CUSTOM_MODEL_BUNDLE_MISSING"


def test_validate_strict_yaml_compiles_exact_builtin_plugins(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """\
schema_version: 1
name: strict-yaml
stages:
  - id: source
    slot: source
    plugin: source.delimited_smiles@0.1.0
    config:
      path: molecules.csv
      delimiter: ','
      has_header: true
      smiles_column: smiles
      candidate_id_column: id
      metadata_columns: []
""",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config), "--json"])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["name"] == "strict-yaml"
    assert len(result["revision_id"]) == 64
    assert result["stages"][0]["plugin"] == "source.delimited_smiles@0.1.0"


def test_validate_rejects_duplicate_json_keys_with_structured_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "duplicate.json"
    config.write_text(
        '{"schema_version":1,"name":"first","name":"second","stages":[]}',
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config), "--json"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["error"]["code"] == "CONFIG_DUPLICATE_KEY"


def test_legacy_vs_overrides_source_path_runs_and_resumes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path, path="must-not-be-read.csv")
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"
    arguments = [
        "--vs",
        "--config",
        str(config),
        "--molecules",
        str(molecules),
        "--workspace",
        str(workspace),
        "--run-id",
        "legacy-run",
        "--json",
    ]

    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["run_id"] == "legacy-run"
    assert first["status"] == "SUCCEEDED"
    assert first["stages"][0]["status"] == "SUCCEEDED"

    revision_path = workspace / "revisions" / f"{first['revision_id']}.json"
    revision = json.loads(revision_path.read_bytes())
    assert revision["config"]["stages"][0]["config"]["path"] == str(molecules)

    assert main([*arguments, "--resume"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["status"] == "SUCCEEDED"
    assert resumed["output_ref"] == first["output_ref"]

    forced_arguments = list(arguments)
    run_id_index = forced_arguments.index("legacy-run")
    forced_arguments[run_id_index] = "forced-run"
    forced_arguments.append("--force")
    assert main(forced_arguments) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced["status"] == "SUCCEEDED"
    events = [
        json.loads(line)
        for line in (workspace / "events" / "forced-run.jsonl").read_text().splitlines()
    ]
    bypassed = [event for event in events if event["event_type"] == "STAGE_CACHE_BYPASSED"]
    assert bypassed[0]["details"]["reason"] == "force"


def test_molecule_override_changes_only_first_enabled_source() -> None:
    payload = _pipeline_document("first.csv")
    stages = payload["stages"]
    assert isinstance(stages, list)
    stages.append(
        {
            "id": "second_source",
            "slot": "source",
            "plugin": "source.sdf@0.1.0",
            "config": {"path": "second.sdf"},
            "enabled": True,
        }
    )
    config = PipelineConfig.model_validate(payload)

    overridden = _molecule_override(
        config,
        Path("replacement.smi"),
        create_builtin_registry(),
    )

    assert overridden.stages[0].config["path"] == "replacement.smi"
    assert overridden.stages[1].config["path"] == "second.sdf"


@pytest.mark.parametrize(
    ("name", "is_directory", "expected_plugin"),
    [
        ("library.csv", False, "source.delimited_smiles@0.1.0"),
        ("library.tsv", False, "source.delimited_smiles@0.1.0"),
        ("library.smi", False, "source.delimited_smiles@0.1.0"),
        ("library.xlsx", False, "source.xlsx@0.1.0"),
        ("library.XLSM", False, "source.xlsx@0.1.0"),
        ("library.sdf", False, "source.sdf@0.1.0"),
        ("library.SD", False, "source.sdf@0.1.0"),
        ("library.parquet", False, "source.raw_molecule_parquet@0.1.0"),
        ("mol2-library", True, "source.mol2_directory@0.1.0"),
    ],
)
def test_molecule_override_auto_routes_source_and_rebuilds_strict_defaults(
    tmp_path: Path,
    name: str,
    is_directory: bool,
    expected_plugin: str,
) -> None:
    molecules = tmp_path / name
    if is_directory:
        molecules.mkdir()
    config = PipelineConfig.model_validate(_pipeline_document())

    overridden = _molecule_override(
        config,
        molecules,
        create_builtin_registry(),
    )

    source = overridden.stages[0]
    assert source.plugin == expected_plugin
    assert source.config["path"] == str(molecules)
    if expected_plugin != "source.delimited_smiles@0.1.0":
        assert "delimiter" not in source.config
        assert "metadata_columns" not in source.config or expected_plugin == "source.xlsx@0.1.0"


def test_explicit_molecule_format_precedes_directory_auto_detection(tmp_path: Path) -> None:
    molecules = tmp_path / "parquet-dataset"
    molecules.mkdir()
    config = PipelineConfig.model_validate(_pipeline_document())

    overridden = _molecule_override(
        config,
        molecules,
        create_builtin_registry(),
        molecule_format="parquet",
    )

    source = overridden.stages[0]
    assert source.plugin == "source.raw_molecule_parquet@0.1.0"
    assert source.config["path"] == str(molecules)
    assert "delimiter" not in source.config


def test_single_mol2_file_requires_directory_source(tmp_path: Path) -> None:
    config = PipelineConfig.model_validate(_pipeline_document())

    with pytest.raises(PipelineError) as caught:
        _molecule_override(
            config,
            tmp_path / "one.MOL2",
            create_builtin_registry(),
        )

    assert caught.value.code == "CLI_MOL2_DIRECTORY_REQUIRED"
    assert "directory" in str(caught.value).lower()


def test_explicit_mol2_directory_rejects_file_before_workspace_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = tmp_path / "one.mol2"
    molecules.write_text("@<TRIPOS>MOLECULE\none\n", encoding="utf-8")
    workspace = tmp_path / "must-not-exist"

    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--format",
            "mol2-directory",
            "--workspace",
            str(workspace),
            "--json",
        ]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "CLI_MOL2_DIRECTORY_REQUIRED"
    assert not workspace.exists()


def test_xlsx_override_uses_first_visible_sheet_default(tmp_path: Path) -> None:
    config = PipelineConfig.model_validate(_pipeline_document())

    overridden = _molecule_override(
        config,
        tmp_path / "library.xlsx",
        create_builtin_registry(),
    )

    source = overridden.stages[0]
    assert source.plugin == "source.xlsx@0.1.0"
    assert source.config["sheet_name"] is None
    assert source.config["has_header"] is True
    assert source.config["smiles_column"] == "smiles"


def test_canonical_run_and_status_with_events_are_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"

    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--molecule-format",
            "delimited",
            "--workspace",
            str(workspace),
            "--run-id",
            "canonical-run",
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SUCCEEDED"

    def forbidden_mkdir(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("status must not create or initialize directories")

    monkeypatch.setattr(Path, "mkdir", forbidden_mkdir)

    assert main(
        [
            "status",
            "canonical-run",
            "--workspace",
            str(workspace),
            "--events",
            "--json",
        ]
    ) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["run_id"] == "canonical-run"
    assert status["status"] == "SUCCEEDED"
    assert status["events"][0]["event_type"] == "RUN_STARTED"
    assert status["events"][-1]["event_type"] == "RUN_SUCCEEDED"


def test_legacy_single_dash_config_and_molecules_are_supported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)

    assert main(
        [
            "--vs",
            "-config",
            str(config),
            "-molecules",
            str(molecules),
            "--format",
            "delimited",
            "--workspace",
            str(tmp_path / "workspace"),
            "--run-id",
            "single-dash-run",
            "--json",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == "single-dash-run"
    assert result["status"] == "SUCCEEDED"


def test_resume_requires_explicit_run_id_and_returns_json_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)

    exit_code = main(["run", str(config), "--resume", "--json"])

    assert exit_code == 1
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["error"]["code"] == "CLI_RESUME_RUN_ID_REQUIRED"


def test_explicit_format_requires_molecule_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)

    exit_code = main(["run", str(config), "--format", "sdf", "--json"])

    assert exit_code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "CLI_MOLECULE_FORMAT_REQUIRES_INPUT"


def test_plugins_lists_only_trusted_builtins_without_environment_discovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_discovery() -> None:
        raise AssertionError("entry-point discovery must be explicit")

    monkeypatch.setattr(
        "molcascade.plugins.registry.importlib_metadata.entry_points",
        forbidden_discovery,
    )

    assert main(["plugins", "--json"]) == 0

    rows = json.loads(capsys.readouterr().out)["plugins"]
    assert rows
    assert all(row["trusted"] and row["origin"] == "builtin" for row in rows)
    assert any(row["kind"] == "source" for row in rows)


def test_doctor_never_runs_commands_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("doctor must not execute commands by default")

    monkeypatch.setattr(subprocess, "run", forbidden_run)

    assert main(["doctor", "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["backends"]
    assert sum(result["summary"].values()) == len(result["backends"])


def test_legacy_vs_requires_both_config_and_molecules() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--vs", "--config", "pipeline.yaml"])

    assert caught.value.code == 2


def test_json_usage_errors_are_single_machine_readable_objects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["run", "--json"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" not in captured.err
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["error"]["code"] == "CLI_USAGE_ERROR"


def test_legacy_json_semantic_usage_errors_are_structured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--vs", "-config", "pipeline.yaml", "--json"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"]["code"] == "CLI_USAGE_ERROR"
    assert "--molecules" in error["error"]["message"]


def test_status_missing_run_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "existing-empty-workspace"
    workspace.mkdir()

    assert main(
        ["status", "missing-run", "--workspace", str(workspace), "--json"]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "RUN_NOT_FOUND"
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("audit_damage", ["missing", "empty"])
def test_status_events_rejects_missing_or_empty_audit_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    audit_damage: str,
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"
    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--workspace",
            str(workspace),
            "--run-id",
            "missing-audit",
            "--json",
        ]
    ) == 0
    capsys.readouterr()
    audit_path = workspace / "events" / "missing-audit.jsonl"
    if audit_damage == "missing":
        audit_path.unlink()
    else:
        audit_path.write_bytes(b"")

    assert main(
        [
            "status",
            "missing-audit",
            "--workspace",
            str(workspace),
            "--events",
            "--json",
        ]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "RUNTIME_AUDIT_INVALID"


def test_status_events_rejects_terminal_event_state_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"
    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--workspace",
            str(workspace),
            "--run-id",
            "terminal-mismatch",
            "--json",
        ]
    ) == 0
    capsys.readouterr()
    audit_path = workspace / "events" / "terminal-mismatch.jsonl"
    lines = audit_path.read_bytes().splitlines()
    assert json.loads(lines[-1])["event_type"] == "RUN_SUCCEEDED"
    audit_path.write_bytes(b"\n".join(lines[:-1]) + b"\n")

    assert main(
        [
            "status",
            "terminal-mismatch",
            "--workspace",
            str(workspace),
            "--events",
            "--json",
        ]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "RUNTIME_AUDIT_INVALID"


@pytest.mark.parametrize("missing_field", ["cache_key", "finished_at"])
def test_status_rejects_incomplete_succeeded_stage_lifecycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing_field: str,
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"
    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--workspace",
            str(workspace),
            "--run-id",
            "invalid-stage-state",
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    state_path = workspace / "runs" / "invalid-stage-state.json"
    payload = json.loads(state_path.read_bytes())
    payload["stages"][0][missing_field] = None
    tampered = RunState.model_validate_json(json.dumps(payload))
    state_path.write_bytes(canonical_json_bytes(tampered))

    assert main(
        [
            "status",
            result["run_id"],
            "--workspace",
            str(workspace),
            "--json",
        ]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "RUNTIME_STATE_INVALID"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config", "pipeline.json", "--json"],
        ["--resume", "--json"],
        ["--json"],
    ],
)
def test_orphan_legacy_options_are_usage_errors(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(arguments)

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"]["code"] == "CLI_USAGE_ERROR"


def test_export_and_report_canonical_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_handoff_pipeline(tmp_path)
    workspace = tmp_path / "workspace"
    assert main(
        [
            "run",
            str(config),
            "--workspace",
            str(workspace),
            "--run-id",
            "handoff-cli",
            "--json",
        ]
    ) == 0
    capsys.readouterr()

    shortlist = tmp_path / "shortlist.smi"
    assert main(
        [
            "export",
            "handoff-cli",
            "--workspace",
            str(workspace),
            "--output",
            str(shortlist),
            "--json",
        ]
    ) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["path"] == str(shortlist)
    assert exported["record_format"] == "SMILES_TSV"
    assert exported["row_count"] == 2
    assert exported["identified_count"] == 2
    assert len(exported["sha256"]) == 64
    # The library named its rows, so the shortlist can be joined back to it.
    content = shortlist.read_text(encoding="utf-8")
    assert content.startswith("SMILES\tparent_id\tsource_id\n")
    assert {line.split("\t")[2] for line in content.splitlines()[1:]} == {"mol-1", "mol-2"}

    assert main(
        [
            "export",
            "handoff-cli",
            "--workspace",
            str(workspace),
            "--output",
            str(shortlist),
            "--overwrite",
        ]
    ) == 0
    human_export = capsys.readouterr().out
    assert "Format: SMILES_TSV" in human_export
    assert "Rows: 2" in human_export
    assert "Named rows: 2/2" in human_export
    assert "SHA-256:" in human_export
    assert f"Path: {shortlist}" in human_export

    assert main(
        ["report", "handoff-cli", "--workspace", str(workspace)]
    ) == 0
    report = workspace / "reports" / "handoff-cli.html"
    assert report.is_file()
    assert str(report) in capsys.readouterr().out
    document = report.read_text(encoding="utf-8")
    assert "MolCascade run handoff-cli" in document
    assert_offline(document)

    original_report = report.read_bytes()
    assert main(
        [
            "report",
            "handoff-cli",
            "--workspace",
            str(workspace),
            "--json",
        ]
    ) == 1
    report_error = json.loads(capsys.readouterr().err)
    assert report_error["error"]["code"] == "REPORT_OUTPUT_EXISTS"
    assert report.read_bytes() == original_report

    assert main(
        [
            "report",
            "handoff-cli",
            "--workspace",
            str(workspace),
            "--overwrite",
            "--json",
        ]
    ) == 0
    report_result = json.loads(capsys.readouterr().out)
    assert report_result == {"path": str(report), "run_id": "handoff-cli"}


@pytest.mark.parametrize("command", ["export", "report"])
@pytest.mark.parametrize("workspace_exists", [False, True])
def test_export_and_report_missing_runs_do_not_create_workspace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    workspace_exists: bool,
) -> None:
    workspace = tmp_path / "missing-workspace"
    if workspace_exists:
        workspace.mkdir()

    assert main(
        [command, "missing-run", "--workspace", str(workspace), "--json"]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    expected_code = "RUN_NOT_FOUND" if workspace_exists else "CLI_WORKSPACE_NOT_FOUND"
    assert error["error"]["code"] == expected_code
    if workspace_exists:
        assert list(workspace.iterdir()) == []
    else:
        assert not workspace.exists()


def test_report_rejects_state_revision_stage_mismatch_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_pipeline(tmp_path)
    molecules = _write_molecules(tmp_path)
    workspace = tmp_path / "workspace"
    assert main(
        [
            "run",
            str(config),
            "--molecules",
            str(molecules),
            "--workspace",
            str(workspace),
            "--run-id",
            "revision-mismatch",
            "--json",
        ]
    ) == 0
    capsys.readouterr()
    state_path = workspace / "runs" / "revision-mismatch.json"
    payload = json.loads(state_path.read_bytes())
    payload["stages"][0]["stage_id"] = "tampered-source"
    tampered = RunState.model_validate_json(json.dumps(payload))
    state_path.write_bytes(canonical_json_bytes(tampered))

    assert main(
        [
            "report",
            "revision-mismatch",
            "--workspace",
            str(workspace),
            "--json",
        ]
    ) == 1

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    error = json.loads(captured.err)
    assert error["error"]["code"] == "RUNTIME_STATE_INVALID"


def test_report_binds_checkpoints_to_exact_revision_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    for label, smiles in (("one", "CCO"), ("two", "CCN")):
        molecules = tmp_path / f"{label}.csv"
        molecules.write_text(f"id,smiles\n{label},{smiles}\n", encoding="utf-8")
        config = tmp_path / f"{label}.json"
        config.write_text(
            json.dumps(_pipeline_document(str(molecules))),
            encoding="utf-8",
        )
        assert main(
            [
                "run",
                str(config),
                "--workspace",
                str(workspace),
                "--run-id",
                f"run-{label}",
                "--json",
            ]
        ) == 0
        capsys.readouterr()

    first_state_path = workspace / "runs" / "run-one.json"
    second_state_path = workspace / "runs" / "run-two.json"
    first_payload = json.loads(first_state_path.read_bytes())
    second_payload = json.loads(second_state_path.read_bytes())
    first_payload["revision_id"] = second_payload["revision_id"]
    tampered_state = RunState.model_validate_json(json.dumps(first_payload))
    first_state_path.write_bytes(canonical_json_bytes(tampered_state))

    audit_path = workspace / "events" / "run-one.jsonl"
    tampered_events: list[bytes] = []
    for line in audit_path.read_bytes().splitlines():
        payload = json.loads(line)
        payload["revision_id"] = second_payload["revision_id"]
        event = AuditEvent.model_validate_json(json.dumps(payload))
        tampered_events.append(canonical_json_bytes(event))
    audit_path.write_bytes(b"\n".join(tampered_events) + b"\n")

    assert main(
        [
            "report",
            "run-one",
            "--workspace",
            str(workspace),
            "--json",
        ]
    ) == 1

    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "RUNTIME_CHECKPOINT_INVALID"
    assert not (workspace / "reports" / "run-one.html").exists()


def test_generate_compatibility_rejects_run_only_options() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--generate", "config", "--resume"])

    assert caught.value.code == 2


def test_module_entrypoint_propagates_version_exit_code() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "molcascade", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("molcascade ")
    assert completed.stderr == ""
