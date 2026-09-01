"""The wordmark in the corner of every generated page, and why it is inlined.

Three renderers write standalone HTML, and all three now open with the project's
mark.  The interesting property is not that it is there -- it is that it travels
*inside* the document.  A ``src`` pointing at a sibling file would look correct
in the checkout, pass every existing test, and then be a broken-image icon the
first time somebody mailed the file to a colleague or opened it off a share.  So
the assertions below are about the form of the reference, not its presence.

The second trap is packaging, and it is invisible from here in the ordinary
sense: the tests run against an editable install, where ``importlib.resources``
reads straight out of the source tree and finds the PNG whether or not the build
backend was ever told to ship it.  Only a wheel-installed MolCascade would raise
-- on a user's machine, at render time.  Hence the assertion that reads
``pyproject.toml``.
"""

from __future__ import annotations

import base64
import re
import tomllib
from pathlib import Path

import pytest

from molcascade.ui.assets import png_data_uri
from molcascade.ui.builder import build_builder_payload, render_config_builder
from molcascade.ui.cascade_builder import build_cascade_builder_payload, render_cascade_builder
from molcascade.ui.report import render_run_report

_MARK = "molcascade-mark.png"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_IMG = re.compile(r"<img\b[^>]*>")


def _documents() -> dict[str, str]:
    """One rendered page from each renderer, keyed by what it is."""

    return {
        "cascade builder": render_cascade_builder(build_cascade_builder_payload()),
        "config builder": render_config_builder(build_builder_payload()),
        # The report renderer reads only these two keys out of the payload, so a
        # real run would cost minutes to assert something about a header.
        "run report": render_run_report({"run": {"run_id": "r", "revision_id": "rev"}}),
    }


def test_the_helper_returns_a_decodable_png_and_not_a_path() -> None:
    uri = png_data_uri(_MARK)
    prefix = "data:image/png;base64,"
    assert uri.startswith(prefix)
    assert base64.b64decode(uri[len(prefix) :], validate=True).startswith(_PNG_MAGIC)


def test_the_base64_is_computed_once_per_process() -> None:
    """A served builder re-renders per request; 42 KB of base64 each time is not free."""

    png_data_uri(_MARK)
    before = png_data_uri.cache_info().hits
    png_data_uri(_MARK)
    assert png_data_uri.cache_info().hits == before + 1


def test_a_missing_asset_fails_loudly_rather_than_rendering_a_blank() -> None:
    with pytest.raises(FileNotFoundError):
        png_data_uri("no-such-mark.png")


@pytest.mark.parametrize("page", sorted(_documents()))
def test_every_generated_page_carries_the_mark_inside_itself(page: str) -> None:
    """No image in any of these documents may reference something outside it."""

    document = _documents()[page]
    images = _IMG.findall(document)
    assert images, f"{page} has no image; the wordmark is missing"
    for tag in images:
        source = re.search(r'src="([^"]*)"', tag)
        assert source, f"{page} has an <img> with no src"
        assert source.group(1).startswith("data:image/png;base64,"), (
            f"{page} references an image by path; the document is no longer self-contained"
        )
        # Both attributes, so the box is reserved before 42 KB of base64 decodes
        # and the text beside the mark does not jump on load.
        assert 'width="' in tag and 'height="' in tag, f"{page} omits the intrinsic ratio"
        # The adjacent "MolCascade" text is the accessible name; a duplicate here
        # would make a screen reader say it twice.
        assert 'alt=""' in tag, f"{page} gives the decorative mark an alt text"


@pytest.mark.parametrize("page", sorted(_documents()))
def test_the_mark_precedes_the_wordmark_text_it_belongs_to(page: str) -> None:
    """Top-left means before the title in source order, which is what a reader gets."""

    document = _documents()[page]
    image = _IMG.search(document)
    assert image
    assert image.start() < document.index("MolCascade", document.index("<body>"))


def test_the_asset_is_declared_as_package_data() -> None:
    """The one thing an editable install can never catch; see the module docstring."""

    root = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = manifest["tool"]["setuptools"]["package-data"]["molcascade.ui.assets"]
    assert "*.png" in patterns
