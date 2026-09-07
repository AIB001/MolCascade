"""Reading back the reasons a run recorded for the molecules it removed.

What is under test is a reader, and a reader of this shape goes wrong in four
specific ways: it can report decision *rows* as if they were molecules, it can
sum ``SOURCE_RECORD`` and ``PARENT`` decisions into a number that means nothing,
it can understate a total when its bucket table fills, and it can refuse a whole
run because one artifact is unreadable.

The library is chosen so that the four reason codes this module exists for --
the ones the pipeline has always written and nothing has ever read -- each fire
exactly once, and so that a stage that attributed another stage's rows to itself
would be visible as a wrong count rather than hidden in a plausible total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.config import PipelineConfig
from molcascade.decisions import build_decision_digest
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner

#: Six records, five parents. Each of the four warn/reject codes fires once:
#: ``benzene_again`` duplicates ``benzene`` (EXACT_DUPLICATE), the hydrochloride
#: is multi-component (FRAGMENTS_REMOVED), alanine has an unassigned stereocentre
#: (UNDEFINED_STEREO), and ethanol at 46 Da plus the 45 Da ethylamine parent both
#: fall under the 50 Da floor (PROPERTY_MW_BELOW_MIN, twice).
_LIBRARY = (
    "id,smiles\n"
    "benzene,c1ccccc1\n"
    "benzene_again,c1ccccc1\n"
    "ethanol,CCO\n"
    "alanine,CC(N)C(=O)O\n"
    "ethylamine_hcl,CC[NH3+].[Cl-]\n"
    "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
)

_MW_FLOOR = 50.0


def _run(tmp_path: Path, *, run_id: str = "decisions-run") -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "decisions-test",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(source),
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
                    "config": {"mw_min": _MW_FLOOR},
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    return runner, runner.run(config, run_id=run_id).run_id


def _stage(digest, stage_id: str):
    for stage in digest.stages:
        if stage.stage_id == stage_id:
            return stage
    raise AssertionError(f"no stage {stage_id} in digest")


def _codes(stage) -> dict[str, int]:
    return {bucket.reason_code: bucket.rows for bucket in stage.buckets}


def test_the_four_codes_nothing_read_are_all_reported(tmp_path: Path) -> None:
    """The point of the module: these were always written and never surfaced."""

    runner, run_id = _run(tmp_path)
    digest = build_decision_digest(runner, run_id)

    standardize = _codes(_stage(digest, "standardize"))
    assert standardize["EXACT_DUPLICATE"] == 1
    assert standardize["FRAGMENTS_REMOVED"] == 1
    assert standardize["UNDEFINED_STEREO"] == 1

    gate = _codes(_stage(digest, "mass_gate"))
    assert gate["PROPERTY_MW_BELOW_MIN"] == 2


def test_entity_kinds_are_not_summed_together(tmp_path: Path) -> None:
    """A rejected source record and a rejected molecule are different events.

    The standardizer decides about ``SOURCE_RECORD``s and the gate decides about
    ``PARENT``s. Collapsing the two would produce a total that is arithmetically
    fine and scientifically meaningless, so the kind is part of every key.
    """

    runner, run_id = _run(tmp_path)
    digest = build_decision_digest(runner, run_id)

    standardize_kinds = {kind for kind, _, _ in _stage(digest, "standardize").outcome_totals}
    gate_kinds = {kind for kind, _, _ in _stage(digest, "mass_gate").outcome_totals}
    assert standardize_kinds == {"SOURCE_RECORD"}
    assert gate_kinds == {"PARENT"}
    for bucket in _stage(digest, "standardize").buckets:
        assert bucket.entity_kind == "SOURCE_RECORD"


def test_totals_stay_exact_when_the_bucket_table_fills(tmp_path: Path) -> None:
    """Capping the reason table must not cost a row from the totals.

    ``rd_filters_alerts`` mixes a rule hash into its reason code, so the number
    of distinct codes is driven by input data rather than bounded by the plugin.
    The cap keeps that from becoming an unbounded allocation; what it must never
    do is make the headline number smaller than the truth.
    """

    runner, run_id = _run(tmp_path)
    full = build_decision_digest(runner, run_id)
    capped = build_decision_digest(runner, run_id, max_buckets=1)

    for stage_id in ("standardize", "mass_gate"):
        wide = _stage(full, stage_id)
        tight = _stage(capped, stage_id)
        assert tight.rows_total == wide.rows_total
        assert tight.outcome_totals == wide.outcome_totals
        assert len(tight.buckets) == 1
        assert sum(b.rows for b in tight.buckets) + tight.untracked_rows == tight.rows_total
        assert tight.untracked_rows > 0


def test_rows_are_rows_and_the_digest_says_so(tmp_path: Path) -> None:
    """A molecule failing two rules in one stage is two rows, and that is fine.

    The guard is that the count is never presented as a molecule count: the
    rendering carries the sentence, and the survivor counts a reader would want
    instead live in the funnel, computed from manifest metadata.
    """

    runner, run_id = _run(tmp_path)
    digest = build_decision_digest(runner, run_id)

    gate = _stage(digest, "mass_gate")
    assert gate.rows_total == sum(rows for _, _, rows in gate.outcome_totals)
    assert "Rows, not molecules" in digest.render()


def test_a_run_with_no_decisions_is_empty_rather_than_absent(tmp_path: Path) -> None:
    """A stage that published a decision port and wrote nothing still appears.

    Distinguishing "this stage recorded nothing" from "this stage is missing" is
    the whole reason the port is reported separately from the row count.
    """

    runner, run_id = _run(tmp_path)
    digest = build_decision_digest(runner, run_id)

    source = _stage(digest, "source")
    assert source.port is not None
    assert source.rows_total == 0
    assert source.unavailable is None


def test_one_unreadable_artifact_does_not_refuse_the_whole_run(tmp_path: Path) -> None:
    """A digest of a damaged workspace shows what is readable.

    A run whose artifacts were partly pruned is exactly when the reasons above
    the damage are worth reading, so an unreadable stage is marked and skipped
    rather than raised -- the same bargain ``trace`` makes.
    """

    runner, run_id = _run(tmp_path)
    intact = build_decision_digest(runner, run_id)
    gate_rows = _stage(intact, "mass_gate").rows_total
    assert gate_rows > 0

    state = runner.load_run(run_id)
    victim = next(stage for stage in state.stages if stage.stage_id == "mass_gate")
    assert victim.output_ref is not None
    root = runner.store.resolve_dataset(victim.output_ref, verify=False)
    for path in root.rglob("*.parquet"):
        path.write_bytes(b"not a parquet file")

    damaged = build_decision_digest(runner, run_id)
    assert _stage(damaged, "standardize").rows_total > 0
    assert _stage(damaged, "mass_gate").unavailable is not None


def test_invalid_bounds_are_refused(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    with pytest.raises(ValueError):
        build_decision_digest(runner, run_id, max_buckets=0)
    with pytest.raises(ValueError):
        build_decision_digest(runner, run_id, sample_size=-1)
