"""What the shipped defaults keep of molecules already known to be worth keeping.

``test_shipped_defaults.py`` beside this file calibrates each tier on its own,
and its history is the argument for the style both files use: pinning the *set*
rather than the count is what caught the sulfoxide-notation bug that was
deleting every proton-pump inhibitor, and the NIBR compound-class rules that
were deleting steroids. A count would have moved and nobody would have known
which molecule moved it.

What no per-tier test can show is that **recall multiplies**. Six gates that
each keep most of what they see keep noticeably less between them, and no single
tier looks wrong at any point along the way. Measured here, on the 77 approved
oral drugs:

===================  =====  =====  ========
tier                    in    out      kept
===================  =====  =====  ========
t1_chemistry            77     77    100.0%
t2_physchem             77     58     75.3%
t3_druglike             58     54     93.1%
t4_alerts               54     51     94.4%
t5_synthesis            51     51    100.0%
t7_structure_qc         51     51    100.0%
**end to end**          77     51  **66.2%**
===================  =====  =====  ========

Retention against this panel is a *floor*, never a target: these molecules
survived years of optimisation, so a funnel that keeps all of them is almost
certainly keeping far too much else. What the panel can say is the opposite
thing -- which tier is deleting something it should not, and which molecule.

Three properties are pinned, and they do different jobs. ``_LOST`` is the ledger:
a legitimate change to a default is *expected* to move it, and the failure
message prints a replacement block to paste. The in-window budget is a ceiling
that a legitimate change is *not* expected to move, so raising it needs its own
argument. And every molecule the panel marks as deliberately outside oral
small-molecule space must still be deleted -- a funnel that starts keeping
metformin has not become more generous, it has stopped making that scope
decision at all.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fixtures.approved_drugs import APPROVED_ORAL_DRUGS, IN_WINDOW, OUT_OF_WINDOW

from molcascade.cascade import defaults as defaults_module
from molcascade.cascade.defaults import default_cascade
from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import CascadeConfig, LibraryConfig
from molcascade.plugins import PluginRegistry, create_builtin_registry
from molcascade.recall import RecallReport, measure_recall
from molcascade.runtime import LocalRunner

_RUN_ID = "shipped-funnel-recall"

#: The tiers measured here: every tier of the shipped cascade that needs no
#: trained model, no receptor and no 3D coordinates. The rest are excused by
#: name below, so that a tier added later is neither measured by accident nor
#: skipped by accident.
_MEASURED_TIERS: tuple[str, ...] = (
    "t1_chemistry",
    "t2_physchem",
    "t3_druglike",
    "t4_alerts",
    "t5_synthesis",
    "t7_structure_qc",
)

#: Why each remaining tier is out of scope. Kept as prose rather than a set so
#: that adding a tier forces someone to write down which side it belongs on.
_EXCUSED: dict[str, str] = {
    "t6_admet": (
        "runs two neural predictors; both arms exist wherever admet-ai and "
        "openadmet are installed, so including it would pin a number that "
        "changes with the machine's packages rather than with the defaults"
    ),
    "t8_ligand_conformers": "embeds 3D geometry",
    "t9_docking": "needs a receptor and out-of-process engines",
    "t9_redock": "needs retained target poses and independent GPU re-search; calibrate per target",
    "t10_docking_metrics": "reads docked poses",
    "t12_affinity": "needs Boltz-2, a target FASTA and a card",
}

#: The measured tiers' shape at the time the numbers below were taken. Compared
#: rather than assumed: a tier that grew an arm, lost one, or changed how it
#: joins is measuring something else, and its pin would then be a number about a
#: different funnel wearing the same name.
_MEASURED_SHAPE: dict[str, tuple[str, tuple[str, ...]]] = {
    "t1_chemistry": ("all", ("structure_validity",)),
    "t2_physchem": ("all", ("physchem_window", "ring_topology", "aromatic_ring_count")),
    "t3_druglike": ("all", ("druglike_ro5_qed", "druglike_oral_consensus")),
    "t4_alerts": ("all", ("alerts_rdkit", "alerts_medchem", "alerts_lilly_demerits")),
    "t5_synthesis": ("any", ("synthesis_sa_score", "synthesis_scscore")),
    "t7_structure_qc": ("all", ("structure_qc_chembl",)),
}

#: A measured tier can legitimately be absent only when its single backend is an
#: optional install. Every other tier in ``_MEASURED_TIERS`` is built from RDKit
#: alone or from a package the shape check already accounts for, so "the whole
#: tier is gone" is a source change and must fail rather than skip.
_TIERS_THAT_CAN_BE_ABSENT = frozenset({"t7_structure_qc"})

#: Who each tier removes. The ledger: a legitimate change to a default is
#: expected to move this, and the failure message prints its replacement.
_LOST: dict[str, tuple[str, ...]] = {
    "t1_chemistry": (),
    "t2_physchem": (
        "allopurinol",
        "amitriptyline",
        "atorvastatin",
        "carbamazepine",
        "codeine",
        "dexamethasone",
        "isoniazid",
        "lapatinib",
        "levofloxacin",
        "levothyroxine",
        "loratadine",
        "metformin",
        "methotrexate",
        "montelukast",
        "morphine",
        "olanzapine",
        "prednisolone",
        "quetiapine",
        "verapamil",
    ),
    "t3_druglike": ("famotidine", "sertraline", "sorafenib", "tamoxifen"),
    "t4_alerts": ("amoxicillin", "ibrutinib", "ranitidine"),
    "t5_synthesis": (),
    "t7_structure_qc": (),
}

#: Measured: 20 of the 71 in-window drugs, 28.2%. The headroom to 0.35 is
#: deliberate. ``_LOST`` is the ledger and moves with any legitimate change;
#: this is the budget, and moving it is a separate argument someone has to make.
_IN_WINDOW_LOSS_CEILING = 0.35

_EXPECTED_SURVIVORS = 51
_EXPECTED_END_TO_END = 0.6623376623376623


@dataclass(frozen=True, slots=True)
class PanelRun:
    workspace: Path
    #: Excluded from ``repr`` so a fixture argument in a pytest header is one
    #: short path rather than a truncated RecallReport.
    report: RecallReport = field(repr=False)


def _source_tier_ids() -> tuple[str, ...]:
    """Every tier id ``defaults.py`` can construct, read from its source.

    Not from ``default_cascade()``: that returns only the tiers this machine can
    build, so a tier whose backend is absent here would slip past unmeasured and
    unexcused on exactly the machine where nobody notices. Reading the source
    catches one that is guarded, optional, or not yet reachable at all.

    ``ast.parse(inspect.getsource(...))`` is the repository's existing way of
    asserting something about a module without executing it -- five test files
    already do it.
    """

    tree = ast.parse(inspect.getsource(defaults_module))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name != "TierConfig":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "id"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                found.append(keyword.value.value)
    return tuple(dict.fromkeys(found))


def _shape(cascade: CascadeConfig) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        tier.id: (str(tier.mode), tuple(c.id for c in tier.active_criteria))
        for tier in cascade.tiers
    }


def _lost_by_tier(report: RecallReport) -> dict[str, tuple[str, ...] | str]:
    """Losses per tier, with an unreadable tier folded into the value itself.

    An unreadable tier also has ``lost == ()``, so a pin comparing only loss sets
    would report "this artifact could not be read" as "this tier no longer
    removes anything" -- the same failure ``recall.py`` refuses to make when it
    returns ``None`` rather than a retention of 1.0. Returning the explanation
    string instead of a tuple makes the two impossible to confuse: a ``str``
    never equals a ``tuple``, so no pin can be satisfied by a broken artifact.
    """

    return {
        tier.tier_id: (tier.lost if tier.unavailable is None else tier.unavailable)
        for tier in report.tiers
    }


@pytest.fixture(scope="module")
def registry(hermetic_installation_paths: None) -> PluginRegistry:
    del hermetic_installation_paths  # ordering only; see tests/conftest.py
    return create_builtin_registry()


@pytest.fixture(scope="module")
def cascade(registry: PluginRegistry) -> CascadeConfig:
    """The shipped defaults, restricted to the tiers measured here.

    ``allow_missing_target`` is deliberately not passed to ``lower_cascade``
    below: with the docking tiers removed nothing needs a receptor, so a
    criterion that starts needing one would fail loudly at lowering instead of
    running quietly against a placeholder target.
    """

    shipped = default_cascade(registry=registry)
    return shipped.model_copy(
        update={
            "tiers": tuple(t for t in shipped.tiers if t.id in _MEASURED_TIERS),
            # recall is measured by name, so the panel must be read with its id
            # column or ``measure_recall`` raises RECALL_NAMES_UNAVAILABLE.
            "library": LibraryConfig(smiles_column="smiles", id_column="id"),
        }
    )


@pytest.fixture(scope="module")
def panel_run(
    cascade: CascadeConfig,
    registry: PluginRegistry,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PanelRun]:
    missing = {
        tier_id: expected
        for tier_id, expected in _MEASURED_SHAPE.items()
        if tier_id in _shape(cascade)
        and set(expected[1]) - set(_shape(cascade)[tier_id][1])
    }
    absent = [
        tier_id
        for tier_id in _MEASURED_TIERS
        if tier_id not in _shape(cascade) and tier_id in _TIERS_THAT_CAN_BE_ABSENT
    ]
    if missing or absent:
        pytest.skip(
            "this installation cannot build the funnel these numbers were "
            f"measured on: missing arms {missing}, absent tiers {absent}"
        )

    workspace = tmp_path_factory.mktemp("shipped-funnel-recall")
    panel = workspace / "approved-oral.csv"
    panel.write_text(
        "id,smiles\n"
        + "".join(f"{name},{smiles}\n" for name, smiles in APPROVED_ORAL_DRUGS.items()),
        encoding="utf-8",
    )
    lowered = lower_cascade(cascade, registry=registry, library_path=str(panel))
    runner = LocalRunner(workspace / "ws", plugins=registry)
    runner.run(lowered.pipeline, run_id=_RUN_ID)
    yield PanelRun(workspace=workspace / "ws", report=measure_recall(runner, _RUN_ID))


def _lost_names(report: RecallReport) -> set[str]:
    """Every panel name any tier removed, skipping tiers that could not be read."""

    return {
        name
        for value in _lost_by_tier(report).values()
        if not isinstance(value, str)
        for name in value
    }


def _recall_drift(run: PanelRun) -> str:
    """One tier and one drug per line, and a command that says why.

    Never prints a reason code or a detail. Measured on this panel, the alert
    tiers' reason codes carry a 64-character hex hash (``structural_alerts`` and
    ``medchem_alerts`` both mix a sha256 into the code) and the drug-likeness
    details are 738-761 character JSON objects, so any fixed-width format cuts
    through the middle of one. ``molcascade explain`` already answers "why" per
    molecule and is quoted instead.
    """

    observed = _lost_by_tier(run.report)
    lines: list[str] = ["whole-funnel recall against the approved-oral panel moved."]
    newly, no_longer = [], []
    for tier_id, expected in _LOST.items():
        seen = observed.get(tier_id)
        if isinstance(seen, str):
            lines.append(f"  {tier_id}: unreadable -- {seen}")
            continue
        seen = seen or ()
        newly.extend((tier_id, name) for name in sorted(set(seen) - set(expected)))
        no_longer.extend((tier_id, name) for name in sorted(set(expected) - set(seen)))
    if newly:
        lines.append("")
        lines.append("newly removed:")
        lines.extend(f"  {tier_id}  {name}" for tier_id, name in newly)
    if no_longer:
        lines.append("")
        lines.append("no longer removed:")
        lines.extend(f"  {tier_id}  {name}" for tier_id, name in no_longer)

    moved = [name for _tier, name in newly + no_longer]
    if moved:
        lines.append("")
        lines.append("why, for the first few (SMILES single-quoted: two carry a backslash):")
        for name in moved[:5]:
            smiles = APPROVED_ORAL_DRUGS[name]
            lines.append(
                f"  molcascade explain {_RUN_ID} --workspace {run.workspace} "
                f"--smiles '{smiles}'"
            )
        if len(moved) > 5:
            lines.append(f"  ... and {len(moved) - 5} more, all named above")
    lines.append("")
    lines.append("If the new answer is the right one, paste this back into _LOST and say,")
    lines.append("per molecule, which setting you changed and why the new answer is correct:")
    lines.append("_LOST = {")
    for tier_id in _LOST:
        seen = observed.get(tier_id)
        lines.append(f"    {tier_id!r}: {tuple(seen) if not isinstance(seen, str) else seen!r},")
    lines.append("}")
    return "\n".join(lines)


def test_every_tier_the_defaults_can_build_is_measured_or_excused() -> None:
    """A tier added later must not slip past unmeasured.

    Runs on every machine, with no skip and no pipeline: the tier ids come from
    ``defaults.py``'s source rather than from a constructed cascade, so a tier
    this installation cannot build -- or one behind a guard -- is still caught.
    """

    classified = set(_MEASURED_TIERS) | set(_EXCUSED)
    unclassified = [t for t in _source_tier_ids() if t not in classified]
    assert not unclassified, (
        f"{unclassified} is new in cascade/defaults.py. Decide whether the funnel "
        "recall guard measures it (add it to _MEASURED_TIERS, re-measure _LOST) "
        "or not (add it to _EXCUSED with the reason)."
    )
    assert not (set(_MEASURED_TIERS) & set(_EXCUSED))


def test_no_measured_tier_has_grown_an_arm_or_changed_how_it_joins(
    cascade: CascadeConfig,
) -> None:
    """A tier measuring something else must fail rather than skip.

    Missing arms are an environment fact and skip the run below; extra arms and
    a changed join are source changes, and a guard that let them skip would be
    silent about exactly what it exists to catch. Losing a whole tier is a source
    change too, except for the one tier whose only backend is an optional
    install.
    """

    observed = _shape(cascade)
    for tier_id, (mode, criteria) in _MEASURED_SHAPE.items():
        if tier_id not in observed:
            assert tier_id in _TIERS_THAT_CAN_BE_ABSENT, (
                f"{tier_id} is no longer built by the shipped defaults; it is not a "
                "tier that can go missing for environment reasons, so this is a "
                "change to defaults.py rather than to this machine."
            )
            continue
        seen_mode, seen_criteria = observed[tier_id]
        assert not set(seen_criteria) - set(criteria), (
            f"{tier_id} gained {sorted(set(seen_criteria) - set(criteria))}; the pins "
            "below describe a funnel without it. Re-measure _LOST and _MEASURED_SHAPE."
        )
        if not set(criteria) - set(seen_criteria):
            # Only meaningful with every arm present: t5 and t6 are declared
            # "any" if union else "all", so a missing arm flips the mode as a
            # fact about the machine rather than about the defaults.
            assert seen_mode == mode, f"{tier_id} now joins as {seen_mode}, not {mode}"


def test_an_unreadable_tier_does_not_read_as_a_tier_that_removed_nothing() -> None:
    """The distinction folded into the compared value, tested on its own.

    Needs no pipeline, so it runs everywhere -- including on a machine that
    skips the measurement below, which is where a silent regression here would
    otherwise live.
    """

    from molcascade.recall import TierRecall

    readable = TierRecall(
        tier_id="t4_alerts", title="", mode="all", entering=10, surviving=10, lost=()
    )
    broken = TierRecall(
        tier_id="t4_alerts",
        title="",
        mode="all",
        entering=10,
        surviving=None,
        lost=(),
        unavailable="stage published no parent population",
    )
    assert _lost_by_tier(RecallReport("r", "SUCCEEDED", 10, 10, (readable,))) == {
        "t4_alerts": ()
    }
    broken_view = _lost_by_tier(RecallReport("r", "SUCCEEDED", 10, 10, (broken,)))
    assert broken_view == {"t4_alerts": "stage published no parent population"}
    assert broken_view != {"t4_alerts": ()}
    assert RecallReport("r", "SUCCEEDED", 10, 10, (broken,)).end_to_end is None


def test_the_panel_arrives_whole_and_the_funnel_finishes(panel_run: PanelRun) -> None:
    """The denominator, checked before anything is divided by it.

    Two panel rows standardizing to one parent would shrink ``registered``
    silently, and every retention below would still look unchanged.
    """

    report = panel_run.report
    assert report.status == "SUCCEEDED"
    assert report.registered == len(APPROVED_ORAL_DRUGS) == 77
    assert report.notes == ()
    assert report.unaccounted == ()


def test_each_tier_removes_the_drugs_it_was_measured_removing(
    panel_run: PanelRun,
) -> None:
    assert _lost_by_tier(panel_run.report) == _LOST, "\n" + _recall_drift(panel_run)


def test_the_end_to_end_number_is_the_one_the_docstring_quotes(
    panel_run: PanelRun,
) -> None:
    """The only literal number here, and it guards prose rather than behaviour.

    If this moves, the table at the top of this file is wrong and must move with
    it -- which is the point of asserting it at all.
    """

    report = panel_run.report
    assert report.survivors == _EXPECTED_SURVIVORS
    assert report.end_to_end == pytest.approx(_EXPECTED_END_TO_END, abs=1e-4)
    assert report.end_to_end == pytest.approx(_EXPECTED_SURVIVORS / 77, abs=1e-4)


def test_the_funnel_stays_within_its_budget_for_in_window_drugs(
    panel_run: PanelRun,
) -> None:
    """A ceiling, not a ledger. Raising it is a separate argument."""

    lost = _lost_names(panel_run.report)
    in_window_lost = lost & set(IN_WINDOW)
    fraction = len(in_window_lost) / len(IN_WINDOW)
    assert fraction <= _IN_WINDOW_LOSS_CEILING, (
        f"{len(in_window_lost)} of {len(IN_WINDOW)} in-window drugs removed "
        f"({fraction:.1%}), above the {_IN_WINDOW_LOSS_CEILING:.0%} budget: "
        f"{sorted(in_window_lost)}"
    )


def test_the_drugs_the_panel_calls_out_of_scope_are_still_removed(
    panel_run: PanelRun,
) -> None:
    """The discriminator. Without it the budget above could be met by a funnel
    that had simply stopped making the scope decision -- one that keeps
    metformin at 129 Da is not more generous, it is no longer deciding."""

    lost = _lost_names(panel_run.report)
    assert set(OUT_OF_WINDOW) <= lost, (
        f"still kept: {sorted(set(OUT_OF_WINDOW) - lost)}"
    )
