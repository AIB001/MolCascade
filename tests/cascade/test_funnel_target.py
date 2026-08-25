"""A shortlist count means nothing until it is read against the target.

The cascade is configured to hand a fixed number of molecules to whatever runs
next -- docking, or a finer screen such as PRISM -- and the budget selector that
produces it is a ``LIMIT``: it truncates a population and can never enlarge one.
So the configured target is a ceiling, and a run can quietly deliver a tenth of
it.  "12,000 shortlisted" is a good result from a 20,000-molecule library and a
bad one from two million, and the bare number does not say which happened.

There are exactly two ways to fall short and they need opposite fixes, which is
why the report diagnoses them separately rather than printing one warning:

* too few molecules survived the tiers -- screen a larger library, or loosen a
  tier;
* enough survived, but the per-scaffold cap bound first -- the library is
  concentrated in too few scaffolds, and enlarging it changes nothing.

Confusing the second for the first sends the user to generate another million
molecules that will be capped exactly the same way.
"""

from __future__ import annotations

from molcascade.cascade.funnel import FunnelReport, FunnelStep


def _prose(report: FunnelReport) -> str:
    """The rendered report with its line wrapping undone.

    The verdict is wrapped to the table's width, so any phrase long enough to
    be worth asserting on is also long enough to be split by it.  Asserting on
    the joined text checks what the sentence says rather than where the wrap
    happened to fall.
    """

    return " ".join(report.render().split())


def _step(stage_id: str, rows: int, entering: int | None = None) -> FunnelStep:
    return FunnelStep(
        stage_id=stage_id,
        label=stage_id,
        role="finalize",
        criterion_id=None,
        rows=rows,
        entering=entering,
    )


def _report(
    *,
    ingested: int,
    shortlisted: int,
    target: int | None,
    selected_from: int | None = None,
    max_per_scaffold: int | None = None,
) -> FunnelReport:
    return FunnelReport(
        run_id="run-test",
        status="succeeded",
        ingest=FunnelStep(
            stage_id="ingest",
            label="Molecule library",
            role="ingest",
            criterion_id=None,
            rows=ingested,
            entering=None,
        ),
        preparation=(),
        tiers=(),
        finalize=(_step("budget", shortlisted),),
        target_count=target,
        selected_from=selected_from,
        max_per_scaffold=max_per_scaffold,
    )


def test_a_run_that_fills_its_budget_says_so_and_names_what_it_chose_from() -> None:
    report = _report(
        ingested=1_000_000, shortlisted=45_000, target=45_000, selected_from=180_000
    )

    assert report.met_target is True
    rendered = _prose(report)
    assert "At the 45,000 target" in rendered
    assert "180,000 survivors" in rendered


def test_falling_short_on_survivors_reports_the_library_size_that_would_not() -> None:
    """The actionable number is how many molecules in, not how many short."""

    report = _report(ingested=608, shortlisted=420, target=45_000, selected_from=422)

    assert report.met_target is False
    rendered = _prose(report)
    assert "Short of the 45,000 target by 44,580" in rendered
    # 608 * 45000 / 420, rounded up.
    assert "65,143" in rendered
    assert "max_per_scaffold" not in rendered


def test_being_capped_per_scaffold_is_reported_as_its_own_cause() -> None:
    """Survivors to spare and still short: only the scaffold cap does that."""

    report = _report(
        ingested=2_000_000,
        shortlisted=30_000,
        target=45_000,
        selected_from=600_000,
        max_per_scaffold=25,
    )

    rendered = _prose(report)
    assert "no more than 25 per scaffold" in rendered
    assert "max_per_scaffold" in rendered
    # The other diagnosis would send the user to generate more molecules, which
    # is the one thing that cannot help here.
    assert "needs roughly" not in rendered


def test_a_report_with_no_target_says_nothing_about_one() -> None:
    """Flat pipelines have no finalize target, and inventing one would mislead."""

    report = _report(ingested=1_000, shortlisted=800, target=None)

    assert report.met_target is None
    rendered = _prose(report)
    assert "target" not in rendered.lower()


def test_the_verdict_is_machine_readable_too() -> None:
    """A caller driving MolCascade from a script should not parse prose."""

    payload = _report(
        ingested=1_000_000, shortlisted=45_000, target=45_000, selected_from=90_000
    ).as_dict()

    assert payload["target_count"] == 45_000
    assert payload["selected_from"] == 90_000
    assert payload["met_target"] is True


def test_the_prose_lines_stay_inside_the_width_the_table_uses() -> None:
    """The counted rows are columnar; unwrapped prose beside them looks broken."""

    rendered = _report(
        ingested=608, shortlisted=420, target=45_000, selected_from=422
    ).render()

    assert all(len(line) <= 78 for line in rendered.splitlines())
