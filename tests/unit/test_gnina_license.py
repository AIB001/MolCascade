"""GNINA is refused for its licence, not for being absent, until someone says so.

Every other backend MolCascade declines to run is declined because this machine
cannot run it: a package is not installed, a weight file was never fetched, a
CUDA build is missing.  GNINA is the first one that is declined for a reason
that has nothing to do with the machine.  It is GPL-2.0-or-later -- only because
it links Open Babel -- and linking a screening pipeline to GPL code is a
decision about the pipeline, one an organisation may have already made either
way.  MolCascade cannot make it, so it takes the reversible answer and says
which flag reverses it.

The failure mode these tests exist to prevent is not "GNINA runs when it should
not".  It is the message: a licence refusal that reads like a missing package
sends the user to run ``pip install``, which cannot possibly help, and leaves
them with no idea that a one-word flag was all that stood in the way.  So the
assertions are as much about what the hint *does not* say as what it does.

The audit half of the same guarantee -- that a run which was permitted records
which tool it linked and under what licence -- lives with the runner it belongs
to, in ``tests/runtime/test_local_runner.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from molcascade.backends.models import ProbePolicy
from molcascade.backends.preflight import copyleft_backends, preflight_backends
from molcascade.cascade.availability import option_availability
from molcascade.cascade.catalog import BackendOption
from molcascade.cli import main
from molcascade.errors import PluginError
from molcascade.pipeline.compiler import CompiledStage
from molcascade.plugins import create_builtin_registry

_GNINA = "docking.gnina@0.1.0"

# GNINA declares Linux only, and a platform refusal is reported before the
# licence is ever consulted -- correctly, since no flag makes a Linux binary run
# on Windows.  These cases are about what happens once the platform is right.
_linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="GNINA declares platforms=('linux',), so elsewhere the platform answers first",
)


def _stage(
    stage_id: str,
    plugin_key: str,
    config: dict[str, object] | None = None,
) -> CompiledStage:
    """A compiled stage carrying only the fields the preflight reads.

    The same shape ``tests/unit/test_backend_preflight.py`` uses, and for the
    same reason: the preflight looks at ``stage_id``, ``plugin_key`` and the
    stage config, and building a whole valid pipeline to supply a descriptor it
    never touches would obscure what is being tested.
    """

    descriptor = create_builtin_registry().entries()[0].descriptor
    return CompiledStage(
        stage_id=stage_id,
        slot="dock",
        plugin_key=plugin_key,
        descriptor=descriptor,
        config=config or {},
        input_contract=None,
        output_contract="molcascade.decision.v1",
        input_port=None,
    )


def _executable(tmp_path: Path) -> str:
    """An absolute path to something that would satisfy the isolated check."""

    path = tmp_path / "gnina"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@_linux_only
def test_default_policy_refuses_gnina_and_says_the_licence_is_why(tmp_path: Path) -> None:
    stages = [_stage("t6_rescore", _GNINA, {"executable": _executable(tmp_path)})]

    with pytest.raises(PluginError) as caught:
        preflight_backends(stages)

    error = caught.value
    assert error.code == "BACKEND_PREFLIGHT_FAILED"
    assert "t6_rescore" in str(error)
    assert "copyleft" in str(error)
    hint = error.hint or ""
    assert "GPL-2.0-or-later" in hint
    assert "--allow-copyleft" in hint
    # The executable is present and the package half is irrelevant, so any
    # instruction to install something would be an instruction to fix a machine
    # that is not broken.
    assert "pip install" not in hint


@_linux_only
def test_permitting_copyleft_moves_the_question_on_to_the_executable(tmp_path: Path) -> None:
    """With the flag, the next answer must be about this machine again."""

    del tmp_path
    policy = ProbePolicy(allow_copyleft=True, run_version_commands=False)

    with pytest.raises(PluginError) as caught:
        preflight_backends([_stage("t6_rescore", _GNINA)], policy=policy)

    error = caught.value
    assert "does not say where to find gnina" in str(error)
    assert "copyleft" not in str(error)
    hint = error.hint or ""
    assert "--allow-copyleft" not in hint
    # An isolated backend is not a pip install into this environment, and saying
    # so is the whole point of the tier.
    assert "environment of its own" in hint


@_linux_only
def test_a_permitted_gnina_with_its_executable_configured_is_ready(tmp_path: Path) -> None:
    policy = ProbePolicy(allow_copyleft=True, run_version_commands=False)
    stages = [_stage("t6_rescore", _GNINA, {"executable": _executable(tmp_path)})]

    assert preflight_backends(stages, policy=policy) == ("gnina",)


@_linux_only
def test_the_permission_is_recorded_by_name_not_as_a_boolean(tmp_path: Path) -> None:
    """A methods section has to name the tool and the licence, not the flag."""

    executable = _executable(tmp_path)
    stages = [
        _stage("t6_rescore", _GNINA, {"executable": executable}),
        _stage("t7_rescore_again", _GNINA, {"executable": executable}),
        _stage("t5_dock", "docking.unidock@0.1.0", {"executable": executable}),
    ]

    permitted = copyleft_backends(stages)

    # Deduplicated: two stages running one tool is one licence decision, and a
    # provenance record that named GNINA twice would invite the reader to
    # wonder what the second one was.
    assert [spec.id for spec in permitted] == ["gnina"]
    assert permitted[0].license_spdx == "GPL-2.0-or-later"


def test_a_cascade_without_copyleft_tools_permits_nothing(tmp_path: Path) -> None:
    stages = [_stage("t5_dock", "docking.unidock@0.1.0", {"executable": _executable(tmp_path)})]

    assert copyleft_backends(stages) == ()


def test_the_builder_offers_gnina_but_names_the_flag_instead_of_a_package() -> None:
    """The builder cannot pass the flag, so it has to explain rather than fail.

    ``option_availability`` is what greys an option out in the exported HTML.
    Reporting GNINA as an uninstalled package would be wrong twice over: the
    adapter *is* installed, and installing anything at all would not make the
    option selectable.
    """

    option = BackendOption(
        id="gnina",
        label="GNINA 1.3 CNN rescoring",
        engine="gnina",
        summary="Rescore docked poses with a convolutional neural network.",
        plugin_ref=_GNINA,
        license_spdx="GPL-2.0-or-later",
        requires_license_optin=True,
    )

    availability = option_availability(option, create_builtin_registry())

    assert not availability.runnable
    reason = availability.reason or ""
    assert "GPL-2.0-or-later" in reason
    assert "--allow-copyleft" in reason
    # The two reasons it is *not*, either of which would send the user to do
    # something that cannot help.
    assert availability.missing_packages == ()
    assert availability.missing_assets == ()
    assert "pip install" not in reason
    assert "molcascade assets fetch" not in reason


@_linux_only
def test_doctor_reports_gnina_blocked_until_the_flag_is_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``doctor`` answers the licence question before anyone starts a run."""

    assert main(["doctor", "--json"]) == 0
    blocked = _backend_row(capsys, "gnina")
    assert blocked["availability"] == "blocked"
    assert "copyleft" in blocked["reason"]

    assert main(["doctor", "--json", "--allow-copyleft"]) == 0
    permitted = _backend_row(capsys, "gnina")
    assert permitted["availability"] != "blocked"
    # Permitted, not relicensed: the recorded terms are the ones the binary
    # actually ships under, whatever this installation decided to allow.
    assert permitted["spec"]["license_spdx"] == "GPL-2.0-or-later"


def _backend_row(capsys: pytest.CaptureFixture[str], backend_id: str) -> dict[str, object]:
    rows = json.loads(capsys.readouterr().out)["backends"]
    for row in rows:
        if row["spec"]["id"] == backend_id:
            return dict(row)
    raise AssertionError(f"{backend_id} is not in the backend catalogue")


@pytest.mark.parametrize("command", ["run", "screen", "doctor"])
def test_every_entry_point_that_can_start_a_run_accepts_the_flag(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag on ``screen`` and not on ``run`` would be a trap, not a policy.

    ``doctor`` is included because the whole value of asking beforehand is that
    it answers the question the run will ask, under the same terms.
    """

    with pytest.raises(SystemExit) as caught:
        main([command, "--help"])

    assert caught.value.code == 0
    assert "--allow-copyleft" in capsys.readouterr().out
