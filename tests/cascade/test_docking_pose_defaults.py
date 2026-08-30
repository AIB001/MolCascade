"""Docking ships storing its poses, and says so in two places that can drift.

``keep_poses`` is written down twice.  The plugin's config model holds the value
a cascade gets when it stays silent about the setting; the builder catalogue
holds the value it writes into a generated ``cascade.json``.  Nothing makes the
two agree, and they disagreed for as long as the setting existed -- both off,
which meant a docking run threw its structures away unless the operator knew to
ask, and ``molcascade trace`` then wrote an empty SDF with nothing to say about
why.  A screen that only wants numbers can still turn it off; that is the point
of it being a setting rather than the reason to default it off.

So this pins the on-state at both copies, and pins that they are the *same*
copy of it.  A future edit to one of them is then a failure here rather than a
run that silently stops keeping geometry.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.catalog import CRITERIA, BackendOption
from molcascade.plugins import create_builtin_registry

#: Every catalogue option that offers the setting at all, named by the plugin it
#: configures so a failure says which engine changed.
_OPTIONS: list[tuple[str, BackendOption]] = [
    (option.plugin_ref, option)
    for criterion in CRITERIA
    for option in criterion.options
    if "keep_poses" in option.defaults
]


def test_every_docking_engine_offers_the_setting() -> None:
    """A new engine that forgot the key would otherwise be skipped in silence."""

    assert {reference.split("@", 1)[0] for reference, _option in _OPTIONS} == {
        "docking.unidock",
        "docking.karmadock",
        "docking.gnina",
    }


@pytest.mark.parametrize(
    ("reference", "option"), _OPTIONS, ids=[reference for reference, _ in _OPTIONS]
)
def test_the_builder_writes_pose_storage_on(reference: str, option: BackendOption) -> None:
    assert option.defaults["keep_poses"] is True, reference


@pytest.mark.parametrize(
    ("reference", "option"), _OPTIONS, ids=[reference for reference, _ in _OPTIONS]
)
def test_a_silent_cascade_gets_what_the_builder_would_have_written(
    reference: str, option: BackendOption
) -> None:
    """The two defaults are the same default, not two that happen to match today.

    A cascade written by hand omits most keys, so it takes the model's default;
    one written by the builder carries the catalogue's.  If those parted the
    same file would mean different things depending on which produced it.
    """

    plugin = create_builtin_registry().get(reference)
    field = plugin.config_model.model_fields["keep_poses"]

    assert field.default is option.defaults["keep_poses"], reference
