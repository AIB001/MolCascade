from __future__ import annotations

import pytest

from molcascade.runtime.models import validate_run_id


@pytest.mark.parametrize(
    "value",
    [
        "CON",
        "con.json",
        "PrN.report",
        "AUX",
        "NUL",
        "COM1",
        "com9.anything",
        "LPT1",
        "lpt9.log",
        "portable-but-trailing.",
    ],
)
def test_run_id_rejects_windows_device_and_ambiguous_names(value: str) -> None:
    with pytest.raises(ValueError, match="portable filename"):
        validate_run_id(value)


@pytest.mark.parametrize("value", ["run-001", "screen.v2", "AUXiliary", "COM10"])
def test_run_id_accepts_portable_names(value: str) -> None:
    assert validate_run_id(value) == value

