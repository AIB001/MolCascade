from __future__ import annotations

from pathlib import Path

from fixtures.offline_html import assert_offline

from molcascade.config import PipelineConfig
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner
from molcascade.ui.report import build_run_report_payload, generate_run_report


def _completed_run(tmp_path: Path) -> tuple[LocalRunner, str]:
    molecules = tmp_path / "molecules.csv"
    molecules.write_text("id,smiles\nfirst,CCO\nsecond,c1ccccc1\n", encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "report-test",
            "description": "</textarea><script>globalThis.pwned=true</script>",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(molecules),
                        "delimiter": ",",
                        "has_header": True,
                        "smiles_column": "smiles",
                        "candidate_id_column": "id",
                        "metadata_columns": [],
                    },
                }
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    result = runner.run(config, run_id="report-run")
    return runner, result.run_id


def test_report_payload_contains_only_control_plane_summaries(tmp_path: Path) -> None:
    runner, run_id = _completed_run(tmp_path)
    payload = build_run_report_payload(runner, run_id)

    assert payload["run"]["status"] == "SUCCEEDED"
    assert payload["molecule_rows_embedded"] == 0
    assert payload["stages"][0]["response_metadata"]["input_count"] == 2
    assert payload["stages"][0]["row_counts"]["primary"] == 2
    assert "raw_smiles" not in payload


def test_generated_report_is_offline_csp_hardened_and_escapes_payload(tmp_path: Path) -> None:
    runner, run_id = _completed_run(tmp_path)
    destination = generate_run_report(runner, run_id, tmp_path / "report.html")
    document = destination.read_text(encoding="utf-8")

    assert "MolCascade local screening report" in document
    assert "connect-src &#x27;none&#x27;" in document
    assert_offline(document)
    assert "</textarea><script>globalThis.pwned" not in document
    assert "&lt;/textarea&gt;&lt;script&gt;globalThis.pwned" in document
    assert "Download next configuration" in document
    assert 'lang="en"' in document
