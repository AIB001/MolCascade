"""Measuring what a funnel keeps of molecules known to be worth keeping.

Recall multiplies across a cascade, so the number this produces is the one
per-tier calibration cannot show. That also makes it the number most worth
lying about, and the lies are specific: an unreadable tier that reports as
having kept everything, a partial run whose tiers are each honest and whose
product is meaningless, and a panel whose names never arrived being reported as
a funnel that lost all of them.

Each of those is tested here directly, because each produces a plausible number
rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import (
    CascadeConfig,
    CriterionConfig,
    FinalizeConfig,
    LibraryConfig,
    StepConfig,
    TierConfig,
)
from molcascade.errors import PipelineError
from molcascade.plugins import create_builtin_registry
from molcascade.recall import measure_recall
from molcascade.runtime import LocalRunner

#: Ethanol and the ethylamine from the hydrochloride both fall under the mass
#: floor; the rest clear it. Names are what the report is for, so they are the
#: thing the assertions read.
_PANEL = (
    "id,smiles\n"
    "benzene,c1ccccc1\n"
    "ethanol,CCO\n"
    "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
    "salt,CC[NH3+].[Cl-]\n"
    "aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
)

_MW_FLOOR = 50.0


def _cascade(*, id_column: str | None = "id") -> CascadeConfig:
    """Two tiers, so the product of retentions is not the same as one tier's."""

    return CascadeConfig(
        name="recall-test",
        library=LibraryConfig(smiles_column="smiles", id_column=id_column),
        ingest=StepConfig(id="library", backend="source.delimited_smiles@0.1.0"),
        standardize=StepConfig(
            id="standardize", backend="chemistry.rdkit_standardize@0.1.0"
        ),
        tiers=(
            TierConfig(
                id="t1_mass",
                title="Mass floor",
                mode="all",
                criteria=(
                    CriterionConfig(
                        id="mass_gate",
                        criterion="physchem_window",
                        backend="chemistry.rdkit_property_range_gate@0.1.0",
                        settings={"mw_min": _MW_FLOOR},
                    ),
                ),
            ),
            TierConfig(
                id="t2_rings",
                title="Ring ceiling",
                mode="all",
                criteria=(
                    CriterionConfig(
                        id="ring_gate",
                        criterion="ring_topology",
                        backend="chemistry.rdkit_ring_topology_gate@0.1.0",
                        settings={"max_rings": 0},
                    ),
                ),
            ),
        ),
    )


def _run(
    tmp_path: Path,
    *,
    id_column: str | None = "id",
    run_id: str = "recall-run",
) -> tuple[LocalRunner, str]:
    panel = tmp_path / "panel.csv"
    panel.write_text(_PANEL, encoding="utf-8")
    registry = create_builtin_registry()
    lowered = lower_cascade(
        _cascade(id_column=id_column),
        registry=registry,
        library_path=str(panel),
        allow_missing_target=True,
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    return runner, runner.run(lowered.pipeline, run_id=run_id).run_id


def _tier(report, tier_id: str):
    return next(tier for tier in report.tiers if tier.tier_id == tier_id)


def test_the_product_is_reported_and_the_lost_molecules_are_named(
    tmp_path: Path,
) -> None:
    """Per-tier retention is what calibration already had; the product is new.

    Two tiers each keeping most of what they see keep noticeably less between
    them, and naming the losses is what makes the number actionable rather than
    merely alarming.
    """

    runner, run_id = _run(tmp_path)
    report = measure_recall(runner, run_id)

    assert report.registered == 5
    mass = _tier(report, "t1_mass")
    rings = _tier(report, "t2_rings")
    assert set(mass.lost) == {"ethanol", "salt"}
    assert set(rings.lost) == {"benzene", "caffeine", "aspirin"}

    assert report.end_to_end == pytest.approx(
        mass.retention * rings.retention
    )
    assert report.survivors == 0


def test_an_unreadable_tier_makes_the_product_unknown_not_perfect(
    tmp_path: Path,
) -> None:
    """The failure that would make a broken run look like a flawless one.

    Treating survivors as equal to arrivals when a stage cannot be read is the
    obvious implementation and the worst one: the end-to-end figure would then
    rise the more of the run was damaged.
    """

    runner, run_id = _run(tmp_path)
    intact = measure_recall(runner, run_id)
    assert intact.end_to_end is not None

    state = runner.load_run(run_id)
    victim = next(s for s in state.stages if s.stage_id == "ring_gate")
    assert victim.output_ref is not None
    root = runner.store.resolve_dataset(victim.output_ref, verify=False)
    for path in root.rglob("*.parquet"):
        path.write_bytes(b"not a parquet file")

    damaged = measure_recall(runner, run_id)
    rings = _tier(damaged, "t2_rings")
    assert rings.unavailable is not None
    assert rings.retention is None
    assert damaged.end_to_end is None
    # render() points at the notes whenever it withholds the figure, so a run
    # that withheld it must have one. Without this the JSON reads complete=true
    # with an empty notes list, which a consumer takes for "nothing to report".
    assert any("could not be read" in note for note in damaged.notes)
    # The tier above it is still measured: one broken artifact costs its own
    # tier and the product, not the whole report.
    assert _tier(damaged, "t1_mass").retention is not None


def test_a_panel_with_no_names_is_an_error_not_a_total_loss(tmp_path: Path) -> None:
    """"Every molecule was lost" and "no molecule was named" look identical.

    One is the most alarming finding this command can produce and the other is a
    missing ``--id-column``, so they must not print the same way.
    """

    runner, run_id = _run(tmp_path, id_column=None)
    with pytest.raises(PipelineError) as caught:
        measure_recall(runner, run_id)
    assert caught.value.code == "RECALL_NAMES_UNAVAILABLE"
    assert "id" in (caught.value.hint or "").lower()


def test_a_flat_pipeline_is_refused_rather_than_measured(tmp_path: Path) -> None:
    """Recall is per tier, and a hand-written pipeline has stages but no tiers."""

    from molcascade.config import PipelineConfig

    panel = tmp_path / "panel.csv"
    panel.write_text(_PANEL, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "flat",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(panel),
                        "smiles_column": "smiles",
                        "candidate_id_column": "id",
                    },
                }
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    run_id = runner.run(config, run_id="flat-run").run_id

    with pytest.raises(PipelineError) as caught:
        measure_recall(runner, run_id)
    assert caught.value.code == "RECALL_NOT_A_CASCADE"


def test_the_tier_structure_comes_from_the_run_not_from_a_config(
    tmp_path: Path,
) -> None:
    """Reading the funnel from the revision is what makes it unmistakable.

    A recall measured against a configuration file the caller supplied could be
    measured against a different funnel than the one that ran, and would look
    exactly as authoritative.
    """

    runner, run_id = _run(tmp_path)
    report = measure_recall(runner, run_id)

    assert [tier.tier_id for tier in report.tiers] == ["t1_mass", "t2_rings"]
    assert [tier.title for tier in report.tiers] == ["Mass floor", "Ring ceiling"]


def test_a_selector_that_cuts_a_congeneric_panel_is_not_hidden(tmp_path: Path) -> None:
    """The loss that happens after the last tier, which is not a tier.

    A panel of measured molecules is a congeneric series -- that is what a
    project's own actives look like -- so RDKit finds one or two Murcko
    scaffolds across the whole panel, and the shipped shortlist selector caps
    each scaffold at 25.  Molecules leave that no gate rejected.

    Reporting only the tiers here is not a small approximation.  On the panel
    below the tiers keep everything and the file somebody opens holds three
    fifths of it, so the tier-only figure is 100% for a run that delivered 60%.
    """

    panel = "id,smiles\n" + "".join(
        f"AN-{index:03d},O=C(Nc1ccc({tail})cc1)c1ccccc1\n"
        for index, tail in enumerate(
            ("C", "CC", "CCC", "CCCC", "F", "Cl", "Br", "I", "O", "OC")
        )
    )
    source = tmp_path / "panel.csv"
    source.write_text(panel, encoding="utf-8")

    cascade = CascadeConfig(
        name="congeneric",
        library=LibraryConfig(smiles_column="smiles", id_column="id"),
        tiers=(
            TierConfig(
                id="t1_mass",
                title="Mass",
                criteria=(
                    CriterionConfig(
                        id="mass_gate",
                        criterion="physchem_window",
                        backend="chemistry.rdkit_property_range_gate@0.1.0",
                        settings={"mw_min": _MW_FLOOR},
                    ),
                ),
            ),
        ),
        ingest=StepConfig(id="library", backend="source.delimited_smiles@0.1.0"),
        standardize=StepConfig(
            id="standardize", backend="chemistry.rdkit_standardize@0.1.0"
        ),
        finalize=FinalizeConfig(
            target_count=45_000,
            seed=7,
            steps=(
                StepConfig(id="scaffolds", backend="scaffold.rdkit_murcko@0.1.0"),
                StepConfig(
                    id="shortlist_select",
                    backend="select.native_scaffold_round_robin@0.1.0",
                    settings={"max_per_scaffold": 6},
                ),
            ),
        ),
    )
    registry = create_builtin_registry()
    lowered = lower_cascade(
        cascade, registry=registry, library_path=str(source), allow_missing_target=True
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=registry)
    run_id = runner.run(lowered.pipeline, run_id="congeneric").run_id

    report = measure_recall(runner, run_id)
    assert _tier(report, "t1_mass").retention == 1.0
    # The cap, not a gate, is what removed them.
    cut = next(row for row in report.finalize if row.tier_id == "shortlist_select")
    assert cut.surviving == 6
    assert len(cut.lost) == 4
    assert report.survivors == 6
    assert report.end_to_end == 0.6
    assert any("after the last tier" in note for note in report.notes)
    assert "not screening criteria" in report.render()
