"""The MCP surface, and the two properties of it that are easy to lose.

Most of what this package does is call library functions that are tested where
they live, so these tests are about the adapter layer rather than about
screening: the error envelope an agent classifies on, the path rule, and two
structural properties that a well-meant simplification would silently undo.

The first is that ``screen`` is a coroutine function.  Under stdio transport a
synchronous tool is invoked inline on the event loop, so rewriting ``screen`` as
a plain ``def`` would freeze the whole server for the duration of a run -- no
second call, no ping, no cancellation -- and nothing about the result would look
different in a passing test.  The assertion below is the only thing standing
between that change and a server that appears to work until somebody screens a
real library.

The second is that every tool carries a docstring.  A tool's docstring *is* its
description over the wire; a tool without one is invisible to the model that has
to choose it.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

pytest.importorskip("mcp", reason="the MCP SDK is an optional extra")

from molcascade.mcp import mcp
from molcascade.mcp._common import absolute_path, dumps, tool

#: Five groups, counted here so that adding a tool without deciding which
#: question it answers is a failing test rather than a quiet drift.
_EXPECTED_TOOLS = {
    # environment -- can this machine do it
    "doctor",
    "environment",
    "plugins",
    "citations",
    "assets",
    # configuration -- is this valid, what would it build
    "validate_config",
    "inspect_bundle",
    "generate_builder",
    # screening -- run it, how did it go
    "plan_screen",
    "screen",
    "run_status",
    # results -- the product
    "export_shortlist",
    "trace_run_stages",
    "generate_report",
    # audit -- why did it decide that
    "summarise_decisions",
    "explain_molecule",
    "audit_stereochemistry",
    "measure_recall",
}


def test_every_tool_is_registered_exactly_once() -> None:
    # asyncio.run rather than pytest-asyncio: the listing is the only async call
    # in this file's reach, and a plugin dependency for three awaits would be a
    # new install for everyone who runs the suite.
    tools = asyncio.run(mcp.list_tools())
    names = [entry.name for entry in tools]

    assert sorted(names) == sorted(_EXPECTED_TOOLS)
    assert len(names) == len(set(names)), "a tool name is registered twice"


def test_every_tool_describes_itself() -> None:
    """A docstring is the tool's description on the wire, not documentation."""

    for entry in asyncio.run(mcp.list_tools()):
        description = (entry.description or "").strip()
        assert description, f"{entry.name} has no description"
        # Long enough to say what the tool needs and what it returns. The shortest
        # real one here is generate_builder; a one-line stub would not reach this.
        assert len(description) > 120, f"{entry.name}'s description is a stub"


def test_the_resources_and_prompt_are_registered() -> None:
    uris = {str(entry.uri) for entry in asyncio.run(mcp.list_resources())}
    assert uris == {"molcascade://overview", "molcascade://contracts"}
    names = [entry.name for entry in asyncio.run(mcp.list_prompts())]
    assert names == ["calibrate_a_funnel"]


def test_screen_is_a_coroutine_function() -> None:
    """The property that keeps a long run from freezing the server.

    A synchronous tool function is called inline on the event loop. ``screen``
    takes minutes to hours, so it must be awaited and must do its work off the
    loop; a plain ``def`` here would be a server that stops answering for the
    length of a screening campaign.
    """

    from molcascade.mcp import screening

    captured: dict[str, object] = {}

    class Collector:
        def tool(self, *args: object, **kwargs: object):
            def decorate(function):
                captured[function.__name__] = function
                return function

            return decorate

    screening.register(Collector())

    assert inspect.iscoroutinefunction(captured["screen"])
    # The two cheap ones are deliberately synchronous: planning is seconds and
    # reading run state is milliseconds, so a thread hop would only add latency.
    assert not inspect.iscoroutinefunction(captured["plan_screen"])
    assert not inspect.iscoroutinefunction(captured["run_status"])


def test_a_relative_path_is_refused_with_a_code() -> None:
    from molcascade.errors import MolCascadeError

    with pytest.raises(MolCascadeError) as raised:
        absolute_path("relative/workspace", field="workspace")

    assert raised.value.code == "MCP_PATH_NOT_ABSOLUTE"
    # The hint has to explain *why*, because "use an absolute path" invites the
    # caller to think the adapter is being fussy rather than that a relative path
    # is genuinely ambiguous here.
    assert "working directory" in (raised.value.hint or "")


def test_a_missing_output_parent_is_refused_rather_than_created() -> None:
    from molcascade.errors import MolCascadeError

    with pytest.raises(MolCascadeError) as raised:
        absolute_path("/nonexistent-root-xyz/out.html", field="output_path", must_exist=False)

    assert raised.value.code == "MCP_PARENT_NOT_FOUND"


def test_the_error_envelope_carries_the_project_s_own_fields() -> None:
    """An agent classifies on ``code`` and ``retryable``; both must survive."""

    from molcascade.errors import MolCascadeError

    @tool
    def failing() -> str:
        raise MolCascadeError(
            "the receptor moved",
            code="TEST_CODE",
            hint="put it back",
            retryable=True,
            context={"where": "here"},
        )

    payload = json.loads(failing())

    assert payload["ok"] is False
    error = payload["error"]
    assert error["code"] == "TEST_CODE"
    assert error["retryable"] is True
    assert error["hint"] == "put it back"
    assert error["context"]["where"] == "here"


def test_an_unexpected_exception_is_still_structured() -> None:
    """A bare traceback reaching an agent is unclassifiable; this prevents it."""

    @tool
    def exploding() -> str:
        raise ZeroDivisionError("nope")

    payload = json.loads(exploding())

    assert payload["ok"] is False
    assert payload["error"]["code"] == "ZeroDivisionError"
    assert payload["error"]["retryable"] is False
    assert payload["error"]["context"]["tool"] == "exploding"


def test_results_serialise_past_datetimes_and_paths() -> None:
    """Run state carries both, and failing on them would fail after the work."""

    from datetime import datetime
    from pathlib import Path

    text = dumps({"when": datetime(2026, 9, 11), "where": Path("/tmp/x")})
    decoded = json.loads(text)

    assert decoded["when"].startswith("2026-09-11")
    assert decoded["where"] == "/tmp/x"
