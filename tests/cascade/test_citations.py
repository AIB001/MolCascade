"""The checked-in bibliography has to keep matching the catalogue it describes.

A stale citation file is worse than no citation file: it reads as a complete
record while quietly omitting whichever tool was added last, and the omission is
invisible at review time because nothing about the diff mentions it.  So the
document is generated, and this module fails the build when the copy on disk and
the generator disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.cascade.catalog import CRITERIA
from molcascade.cascade.citations import render_citations, uncited_options

_DOCUMENT = Path(__file__).parents[2] / "docs" / "citations.md"


def test_a_local_copy_of_the_document_is_not_left_stale() -> None:
    """``docs/`` is not published, so the copy is local convenience, not source.

    It is still worth checking when it is there: a bibliography that reads as
    complete while omitting whichever tool was added last is worse than none,
    and someone keeping a copy beside a manuscript wants to know it drifted.
    A clone that never generated one has nothing to be stale, so this skips
    rather than failing -- the guarantee that every option *has* a reference
    lives in the next test, which needs no file at all.
    """

    if not _DOCUMENT.is_file():
        pytest.skip("no local citations.md; generate one with: molcascade cite -o PATH")
    on_disk = _DOCUMENT.read_text(encoding="utf-8")
    assert on_disk == render_citations(), (
        f"{_DOCUMENT} is out of date with src/molcascade/cascade/catalog.py. "
        f"Regenerate it with: molcascade cite --output {_DOCUMENT}"
    )


def test_every_option_the_builder_offers_carries_a_reference() -> None:
    """A tool that decided which molecules survived has to be nameable in a paper.

    This is the check that makes the bibliography worth having.  Without it the
    generator would happily emit "_No reference recorded._" forever and the file
    would still look thorough.
    """

    assert uncited_options() == ()


@pytest.mark.parametrize("spec", CRITERIA, ids=lambda spec: spec.id)
def test_each_criterion_appears_under_its_own_heading(spec) -> None:
    document = render_citations()
    assert f"### {spec.label}" in document
    for option in spec.options:
        assert f"#### {option.label}" in document


def test_options_without_an_adapter_are_marked_as_such() -> None:
    """The menu lists tools that cannot run here, and must say which those are.

    Reviewed-but-unimplemented entries exist so the catalogue records what was
    considered, not only what shipped.  Presenting them identically to working
    backends would turn an honest menu into a false claim of capability.
    """

    document = render_citations()
    unrunnable = [
        option for spec in CRITERIA for option in spec.options if not option.executable
    ]
    assert unrunnable, "the catalogue no longer lists any reviewed-only options"
    assert document.count("**Status** — Reviewed, no adapter") == len(unrunnable)


def test_vendored_weights_are_listed_with_their_papers() -> None:
    document = render_citations()
    assert "## Weights and rule tables" in document
    # The two assets that ship real bytes. Named explicitly rather than derived,
    # so that dropping one from the registry fails here instead of silently
    # shrinking the bibliography.
    assert "SCScore published weights" in document
    assert "rd_filters structural alert collection" in document
    assert "10.1021/acs.jcim.7b00622" in document
    assert "10.1021/jm901137j" in document
