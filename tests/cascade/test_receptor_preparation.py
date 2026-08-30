"""A failed receptor preparation has to arrive carrying meeko's own diagnosis.

Meeko says two different things when it cannot build a receptor, on two
different streams.  stderr gets a residue-by-residue template-matching trace --
several thousand characters on a real protein, and almost none of it actionable.
stdout gets the summary: which residues failed, which carry an alternate
location, and the flags that get past each.

``prepare_receptor_pdbqt`` used to report ``stderr or stdout``, which on a real
failure is always stderr, because a meeko that failed always wrote a trace.  It
then kept the trace's last 4000 characters, so the operator received four
kilobytes of per-residue debug output and none of the sentence naming the
residues.  These tests pin the two halves of that: both streams survive, and the
summary is the half that survives truncation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from molcascade.cascade.target import _MEEKO_DETAIL_BUDGET, prepare_receptor_pdbqt
from molcascade.errors import ConfigError

# Shortened from a real meeko refusal on STK17B: the summary is 806 characters
# on stdout, the trace 6594 on stderr.
_SUMMARY = (
    "Error: Creation of data structure for receptor failed.\n"
    "Template matching failed for: ['A:43', 'A:67', 'A:76', 'A:125']\n"
    "Residues with alternate location: ['A:26', 'A:97', 'A:285']\n"
    "Recommendations: use --allow_bad_res and --default_altloc"
)
_TRACE_LINE = "matched with excess inter-residue bond(s): A:{index}"


def _trace(lines: int) -> str:
    return "\n".join(_TRACE_LINE.format(index=index) for index in range(lines))


@pytest.fixture
def receptor(tmp_path: Path) -> Path:
    path = tmp_path / "protein.pdb"
    path.write_text("ATOM      1  N   MET A   1       0.000   0.000   0.000\nEND\n")
    return path


def _refuse(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    stderr: str,
) -> None:
    """Make meeko exit 1 with the given streams and write no PDBQT."""

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            returncode=1,
            stdout=stdout.encode("utf-8"),
            stderr=stderr.encode("utf-8"),
        )

    monkeypatch.setattr("molcascade.cascade.target._meeko_command", lambda: ["meeko"])
    monkeypatch.setattr(subprocess, "run", fake_run)


def _detail(receptor: Path, tmp_path: Path) -> str:
    with pytest.raises(ConfigError) as raised:
        prepare_receptor_pdbqt(receptor, workspace=tmp_path / "workspace")
    assert raised.value.code == "DOCKING_RECEPTOR_PREPARATION_FAILED"
    output = raised.value.context["meeko_output"]
    assert isinstance(output, str)
    return output


def test_the_summary_only_stdout_carries_is_not_dropped_for_the_trace(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression itself: ``stderr or stdout`` never reached stdout."""

    _refuse(monkeypatch, stdout=_SUMMARY, stderr=_trace(40))

    detail = _detail(receptor, tmp_path)

    assert "Recommendations: use --allow_bad_res and --default_altloc" in detail
    assert "'A:43', 'A:67'" in detail
    # And the trace is still there -- keeping stdout instead of stderr would
    # only have moved the loss to the other stream.
    assert "excess inter-residue bond(s): A:39" in detail


def test_the_summary_leads_because_a_reader_stops_at_the_first_screenful(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refuse(monkeypatch, stdout=_SUMMARY, stderr=_trace(40))

    detail = _detail(receptor, tmp_path)

    assert detail.index("Creation of data structure") < detail.index("excess inter-residue")


def test_a_trace_larger_than_the_budget_cannot_crowd_out_the_summary(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size that made this invisible in the first place.

    A real trace exceeds the budget on its own, so the two streams cannot both
    be kept whole.  stdout is budgeted first and stderr takes the remainder from
    its end, the end of a trace being the part nearest the failure.
    """

    overflowing = _trace(2_000)
    assert len(overflowing) > _MEEKO_DETAIL_BUDGET

    _refuse(monkeypatch, stdout=_SUMMARY, stderr=overflowing)

    detail = _detail(receptor, tmp_path)

    assert _SUMMARY in detail
    assert len(detail) <= _MEEKO_DETAIL_BUDGET + 32
    assert "excess inter-residue bond(s): A:1999" in detail
    assert "excess inter-residue bond(s): A:0" not in detail


def test_one_stream_speaking_alone_is_reported_without_a_separator(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meeko is not the only thing that can fail here.

    An out-of-memory kill or a missing shared library leaves stdout empty, and
    the report should then be the stream that spoke rather than a heading over
    nothing.
    """

    _refuse(monkeypatch, stdout="", stderr="ImportError: libGL.so.1: cannot open")

    detail = _detail(receptor, tmp_path)

    assert detail == "ImportError: libGL.so.1: cannot open"


def test_the_receptor_path_and_exit_code_stay_in_the_context(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refuse(monkeypatch, stdout=_SUMMARY, stderr=_trace(4))

    with pytest.raises(ConfigError) as raised:
        prepare_receptor_pdbqt(receptor, workspace=tmp_path / "workspace")

    assert raised.value.context["receptor_path"] == str(receptor)
    assert raised.value.context["returncode"] == 1
    assert "--receptor-pdbqt" in (raised.value.hint or "")


def test_a_refused_preparation_leaves_no_half_written_pdbqt(
    receptor: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the digest cache would serve the wreck to the next run."""

    workspace = tmp_path / "workspace"
    _refuse(monkeypatch, stdout=_SUMMARY, stderr=_trace(4))

    with pytest.raises(ConfigError):
        prepare_receptor_pdbqt(receptor, workspace=workspace)

    assert list(workspace.rglob("*.pdbqt")) == []
