from __future__ import annotations

import json
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


def _run_with_decisions(tmp_path: Path) -> tuple[LocalRunner, str]:
    """A run whose standardizer warns and whose gate rejects.

    ``_completed_run`` above is a source stage alone, which records nothing but
    PASS rows -- so it cannot tell whether the projection below actually has
    anything to withhold.
    """

    molecules = tmp_path / "panel.csv"
    molecules.write_text(
        "id,smiles\n"
        "benzene,c1ccccc1\n"
        "benzene_again,c1ccccc1\n"
        "ethanol,CCO\n"
        "alanine,CC(N)C(=O)O\n"
        "salt,CC[NH3+].[Cl-]\n"
        "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n",
        encoding="utf-8",
    )
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "decision-report",
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
                    "inputs": [
                        {"request_port": "primary", "stage": "source", "port": "primary"}
                    ],
                },
                {
                    "id": "mass_gate",
                    "slot": "hard_gate",
                    "plugin": "chemistry.rdkit_property_range_gate@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "standardize", "port": "primary"}
                    ],
                    "config": {"mw_min": 50.0},
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    return runner, runner.run(config, run_id="decision-run").run_id


def test_the_report_says_why_molecules_left(tmp_path: Path) -> None:
    """The reason codes were always written; until now nothing displayed them.

    Without this the page said ``reject: 0`` for the standardizer -- which is
    what its manifest metadata reports -- while its decision dataset held three
    warnings: a duplicate, a fragment selection and an unassigned stereocentre.
    """

    runner, run_id = _run_with_decisions(tmp_path)
    payload = build_run_report_payload(runner, run_id)

    assert payload["report_schema_version"] == 2
    stages = {stage["stage_id"]: stage for stage in payload["decisions"]["stages"]}
    codes = {
        bucket["reason_code"] for bucket in stages["standardize"]["buckets"]
    }
    assert {"EXACT_DUPLICATE", "FRAGMENTS_REMOVED", "UNDEFINED_STEREO"} <= codes
    assert payload["decisions"]["rejected_rows"] == 2


def test_the_projection_withholds_entity_ids_that_the_digest_carries(
    tmp_path: Path,
) -> None:
    """The allow-list, tested on the path it exists to guard.

    Going through ``build_run_report_payload`` cannot catch a projection bug,
    because that call already asks for ``sample_size=0``. So this builds a
    digest that genuinely has sample ids and projects *that*.
    """

    from molcascade.decisions import build_decision_digest
    from molcascade.ui.report import _decision_summary

    runner, run_id = _run_with_decisions(tmp_path)
    digest = build_decision_digest(runner, run_id, sample_size=5)
    populated = [
        bucket
        for stage in digest.stages_with_decisions
        for bucket in stage.buckets
        if bucket.sample_entity_ids
    ]
    assert populated, "the digest under test must actually carry sample ids"

    blob = json.dumps(_decision_summary(digest))
    assert "sample_entity_ids" not in blob
    for bucket in populated:
        for entity_id in bucket.sample_entity_ids:
            assert entity_id not in blob


def test_no_entity_id_reaches_the_document(tmp_path: Path) -> None:
    """``molecule_rows_embedded: 0`` is a claim about the whole file."""

    runner, run_id = _run_with_decisions(tmp_path)
    document = generate_run_report(runner, run_id, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "parent:sha256:" not in document
    assert "source-record:sha256:" not in document
    assert 'id="decisions"' in document
    assert_offline(document)


def test_an_unreadable_stage_does_not_carry_a_local_path_into_the_file(
    tmp_path: Path,
) -> None:
    """A report is emailed; an absolute path from the machine that made it is not
    an aggregate and has no business travelling with one."""

    from molcascade.decisions import StageDecisions
    from molcascade.ui.report import _decision_summary

    class _Digest:
        rejected_rows = 0
        warned_rows = 0
        stages_with_decisions = (
            StageDecisions(
                stage_id="mass_gate",
                port="decisions",
                outcome_totals=(),
                buckets=(),
                rows_total=0,
                untracked_rows=0,
                unavailable="OSError: [Errno 2] /home/someone/secret/ws/artifacts/x.parquet",
            ),
        )

    summary = _decision_summary(_Digest())
    assert summary["stages"][0]["unavailable"] == "OSError"
    assert "/home/someone" not in json.dumps(summary)
