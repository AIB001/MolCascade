"""The loopback route that puts the exported cascade beside the HTML.

Every export used to land in the browser's downloads folder and be moved by
hand, because a page opened from ``file://`` has no other option: it cannot
choose a directory, and it cannot ask for one either -- ``showSaveFilePicker``
is unavailable on an origin the browser does not consider secure.  So the file
stayed the same and the delivery changed: ``generate config --serve`` hands out
a second render of the same document from 127.0.0.1 and writes what that page
posts back.

Two things have to stay true for that to be worth having, and both are asserted
here.  The document written to ``--output`` must not change at all -- it is
still the artifact that works with the network cable out, and it must carry no
token and no endpoint.  And the endpoint must be narrow, because a loopback
port is reachable by every other page in the browser: the destination comes
from the command line rather than the request, the token cannot be guessed, and
a body that is not a cascade is refused before anything is written.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from fixtures.offline_html import assert_offline

from molcascade.ui.cascade_builder import (
    DEFAULT_CONFIG_FILENAME,
    build_cascade_builder_payload,
    generate_cascade_builder,
    render_cascade_builder,
)
from molcascade.ui.serve import (
    MAX_BODY_BYTES,
    SAVE_PATH,
    TOKEN_HEADER,
    BuilderServer,
    serve_cascade_builder,
)

_PAYLOAD = re.compile(r'<textarea id="builder-payload"[^>]*>(.*?)</textarea>', re.DOTALL)

_CASCADE = json.dumps(
    {
        "kind": "cascade",
        "schema_version": 2,
        "name": "written from the browser",
        "tiers": [],
    },
    indent=2,
).encode("utf-8")


def _embedded_payload(document: str) -> dict[str, Any]:
    match = _PAYLOAD.search(document)
    assert match is not None, "the builder document always carries its payload"
    return json.loads(html.unescape(match.group(1)))


def _connect_src(document: str) -> str:
    match = re.search(r"connect-src ([^;]+);", html.unescape(document))
    assert match is not None, "the builder document always pins a connect-src"
    return match.group(1)


@pytest.fixture
def served(tmp_path: Path) -> Any:
    """A running server whose exports land next to a generated HTML file."""

    destination = generate_cascade_builder(tmp_path / "molcascade-config.html")
    server = serve_cascade_builder(destination)
    thread = server.start_background()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _post(
    server: BuilderServer,
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    sent = {"Content-Type": "application/json", TOKEN_HEADER: server.token}
    sent.update(headers or {})
    # A header explicitly set to None is one the caller wants absent; that is
    # how the checks below say "arrived with no token" rather than "arrived
    # with an empty one", which is a different request.
    sent = {key: value for key, value in sent.items() if value is not None}
    request = urllib.request.Request(
        server.url.rstrip("/") + SAVE_PATH, data=body, headers=sent, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# -- the file on disk is unchanged ------------------------------------------


def test_the_document_written_to_disk_still_reaches_for_nothing(tmp_path: Path) -> None:
    """``--serve`` changes the delivery, not the artifact.

    Someone who runs the command, exports their cascade and then mails the HTML
    to a colleague has sent the offline builder, not a page that expects a
    server on a port that no longer exists and a token that was never valid.
    """

    with_serving = generate_cascade_builder(tmp_path / "served.html").read_bytes()
    plain = render_cascade_builder(build_cascade_builder_payload()).encode("utf-8")

    assert with_serving == plain
    document = with_serving.decode("utf-8")
    assert _embedded_payload(document)["save_endpoint"] is None
    assert _connect_src(document) == "'none'"
    assert_offline(document)


def test_the_served_document_may_post_home_and_nowhere_else(served: BuilderServer) -> None:
    """The one thing the served render relaxes, and how far.

    ``connect-src 'self'`` is the served origin only, so the page may return its
    export to the command that produced it and still may not reach anything on
    the network.  The endpoint it posts to is relative for the same reason the
    offline document has no URLs at all: there is no host named anywhere in the
    script, so the same code is inert when it is opened from a file.
    """

    document = urllib.request.urlopen(served.url).read().decode("utf-8")
    endpoint = _embedded_payload(document)["save_endpoint"]

    assert _connect_src(document) == "'self'"
    assert endpoint["url"] == SAVE_PATH
    assert not endpoint["url"].startswith("http")
    assert endpoint["token"] == served.token
    assert endpoint["header"] == TOKEN_HEADER
    # The confirmation names the file the operator is about to pass to --config.
    assert endpoint["path"] == str(served.destination)
    assert_offline(document)


def test_the_export_lands_beside_the_html_under_the_name_the_run_command_prints(
    tmp_path: Path,
) -> None:
    """Where the file goes is fixed by the command line, not by the request.

    There is no filename in the POST at all, which is what makes the endpoint
    uninteresting to anything else on the machine: no traversal, no overwrite of
    a file the operator did not name, no argument to get wrong.
    """

    destination = generate_cascade_builder(tmp_path / "nested" / "molcascade-config.html")
    server = serve_cascade_builder(destination)

    assert server.destination == destination.parent / DEFAULT_CONFIG_FILENAME
    assert server.destination.name in build_cascade_builder_payload()["run_command"]


# -- what the endpoint accepts ----------------------------------------------


def test_a_posted_cascade_is_written_byte_for_byte(served: BuilderServer) -> None:
    """The file beside the HTML is the file the download would have produced.

    The bytes are written as they arrived rather than re-serialised from the
    parsed object, so indentation, key order and the trailing newline survive
    the trip.  A file that differed from the downloaded one would be a second
    format to keep working, invisibly.
    """

    status, body = _post(served, _CASCADE)

    assert status == 200
    assert body == {"ok": True, "path": str(served.destination)}
    assert served.destination.read_bytes() == _CASCADE
    assert [save.size for save in served.saves] == [len(_CASCADE)]


def test_exporting_again_replaces_the_file_rather_than_accumulating(
    served: BuilderServer,
) -> None:
    """Designing a cascade means exporting it repeatedly.

    Each export supersedes the last under the one name ``--config`` is given, so
    the operator never has to work out which of ``cascade(3).json`` is current.
    """

    _post(served, _CASCADE)
    second = _CASCADE.replace(b"written from the browser", b"revised in the browser")
    status, _ = _post(served, second)

    assert status == 200
    assert served.destination.read_bytes() == second
    assert len(served.saves) == 2
    assert sorted(path.name for path in served.destination.parent.iterdir()) == [
        DEFAULT_CONFIG_FILENAME,
        "molcascade-config.html",
    ]


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        pytest.param({TOKEN_HEADER: None}, _CASCADE, 403, id="no token"),
        pytest.param({TOKEN_HEADER: "guessed"}, _CASCADE, 403, id="wrong token"),
        pytest.param({"Origin": "http://elsewhere.example"}, _CASCADE, 403, id="another page"),
        pytest.param({"Content-Type": "text/plain"}, _CASCADE, 415, id="not json"),
        pytest.param({}, b"{ not json", 400, id="unparseable"),
        pytest.param({}, b'["a", "list"]', 400, id="not an object"),
        pytest.param({}, b'{"kind": "pipeline"}', 400, id="not a cascade"),
        pytest.param({}, "{}".encode("utf-16"), 400, id="not utf-8"),
    ],
)
def test_a_request_that_is_not_this_page_exporting_writes_nothing(
    served: BuilderServer, headers: dict[str, str], body: bytes, status: int
) -> None:
    """A loopback port is reachable by every other page in the browser.

    So each layer is checked before a byte is written, and the file that is
    already on disk is what proves it: a refused request must not truncate,
    replace or create it.  The token is the layer that matters most -- a
    cross-origin ``fetch`` cannot set a custom header without a preflight this
    server never answers -- and the JSON check is the layer that stops a
    well-formed request from replacing a good cascade with rubbish.
    """

    _post(served, _CASCADE)
    assert served.destination.read_bytes() == _CASCADE

    code, answer = _post(served, body, headers=headers)

    assert code == status
    assert answer["ok"] is False
    assert answer["error"]
    assert served.destination.read_bytes() == _CASCADE
    assert len(served.saves) == 1


def test_an_oversized_body_is_refused_without_being_read(served: BuilderServer) -> None:
    """The cap exists so the length header alone decides.

    Nothing is read off the socket until the declared size is known to be
    plausible, so a client that promises a gigabyte cannot make the process hold
    one while the checks that would have rejected it run.
    """

    code, _ = _post(served, b"", headers={"Content-Length": str(MAX_BODY_BYTES + 1)})

    assert code == 413
    assert not served.destination.exists()


def test_the_server_answers_one_page_and_one_endpoint(served: BuilderServer) -> None:
    """Nothing else on the filesystem is reachable through this port.

    The document is held in memory and returned for ``/`` alone; there is no
    directory being served, so no path can walk out of one.
    """

    for path in ("/", "/index.html"):
        with urllib.request.urlopen(served.url.rstrip("/") + path) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/html")
            # The token is in this body; keeping it out of the caches is free.
            assert response.headers["Cache-Control"] == "no-store"

    for path in ("/cascade.json", "/../cascade.json", "/molcascade-config.html"):
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(served.url.rstrip("/") + path)
        assert raised.value.code == 404


def test_each_server_mints_its_own_token(tmp_path: Path) -> None:
    """A token that outlived its command would authorise the next one.

    It is generated per server and never written to disk, so the HTML the
    command leaves behind cannot be used to reach a later invocation.
    """

    first = BuilderServer(tmp_path / "one.json")
    second = BuilderServer(tmp_path / "two.json")
    try:
        assert first.token != second.token
        assert len(first.token) >= 32
        assert first.token not in (tmp_path / "one.json").parent.as_posix()
    finally:
        first.shutdown()
        second.shutdown()


def test_a_request_claiming_to_be_from_elsewhere_is_turned_away(served: BuilderServer) -> None:
    """DNS rebinding is the attack this closes.

    A remote page that points its own hostname at 127.0.0.1 reaches this port
    with someone else's ``Host``.  It would fail the token check regardless;
    failing on the ``Host`` first means the token is never even compared against
    a request that had no business arriving.
    """

    request = urllib.request.Request(
        served.url.rstrip("/") + SAVE_PATH,
        data=_CASCADE,
        headers={
            "Host": "attacker.example",
            "Content-Type": "application/json",
            TOKEN_HEADER: served.token,
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request)

    assert raised.value.code == 421
    assert not served.destination.exists()


def test_it_binds_loopback_only(served: BuilderServer) -> None:
    """The port is for this machine; nothing advertises it beyond it."""

    assert served.host == "127.0.0.1"
    assert served.url.startswith("http://127.0.0.1:")
    assert served.port > 0


def test_saves_are_serialised_when_two_tabs_export_at_once(served: BuilderServer) -> None:
    """The server is threaded, so two exports can overlap.

    The write is atomic -- a rename over the destination -- so whichever lands
    second is the whole file, and neither can leave a half-written cascade for
    ``screen`` to read.
    """

    bodies = [
        _CASCADE.replace(b"written from the browser", f"export {index}".encode())
        for index in range(8)
    ]
    results: list[int] = []
    lock = threading.Lock()

    def export(body: bytes) -> None:
        status, _ = _post(served, body)
        with lock:
            results.append(status)

    threads = [threading.Thread(target=export, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert results == [200] * len(bodies)
    assert served.destination.read_bytes() in bodies
    assert len(served.saves) == len(bodies)


# -- the page's half of the exchange ----------------------------------------
#
# The server above is only useful if the shipped button actually talks to it, so
# these run the real ``cascade.js`` in Node against a payload carrying an
# endpoint.  The three runs are the three outcomes an operator can get: the file
# is written, the server refuses it, and the server is not there at all.  The
# last two matter most -- whatever happens, the cascade must not evaporate.

_UI = Path(__file__).parents[2] / "src" / "molcascade" / "ui" / "assets"
_HERE = Path(__file__).parent
_ID_PATTERN = re.compile(r'\sid="([a-zA-Z0-9_-]+)"')
_ENDPOINT = {
    "url": SAVE_PATH,
    "token": "a-token-only-the-served-page-has",
    "header": TOKEN_HEADER,
    "path": "/campaigns/stk17b/cascade.json",
}


def _drive_export(
    tmp_path: Path, *, endpoint: dict[str, str] | None, failure: str = ""
) -> dict[str, Any]:
    """Click Download in a scripted browser and report what left the page."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the scripted browser session cannot run")
    payload = build_cascade_builder_payload()
    payload["save_endpoint"] = endpoint
    document = render_cascade_builder(payload, save_endpoint=endpoint)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ids.json").write_text(
        json.dumps(sorted(set(_ID_PATTERN.findall(document)))), encoding="utf-8"
    )
    (tmp_path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            node,
            str(_HERE / "cascade_driver.js"),
            str(_HERE / "cascade_dom_stub.js"),
            str(_UI / "cascade.js"),
            str(tmp_path / "ids.json"),
            str(tmp_path / "payload.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "DRIVER_FETCH_FAILURE": failure},
    )
    return json.loads(completed.stdout)["export"]


def test_without_an_endpoint_the_button_still_only_makes_a_download(tmp_path: Path) -> None:
    """The document on disk behaves exactly as it did before any of this.

    Same code, same button; the branch that posts is simply unreachable, which
    is what makes it safe for the two renders to share one script.
    """

    export = _drive_export(tmp_path, endpoint=None)

    assert export["endpoint"] is None
    assert export["posted"] == []
    assert export["downloadedAfterPost"] == 1
    assert json.loads(export["settledFile"])["kind"] == "cascade"
    assert "Saved cascade.json" in export["settledStatus"]["text"]


def test_the_served_page_posts_the_config_and_does_not_download_it(tmp_path: Path) -> None:
    """One export, one request, no file in the downloads folder.

    The body is the same text the download would have contained, which is what
    lets the server write the bytes through untouched, and the token travels in
    a custom header -- a header a cross-origin page cannot set without a
    preflight, which is most of why the endpoint is safe to expose.
    """

    export = _drive_export(tmp_path, endpoint=_ENDPOINT)

    assert len(export["posted"]) == 1
    posted = export["posted"][0]
    assert posted["url"] == SAVE_PATH
    assert posted["method"] == "POST"
    assert posted["headers"]["Content-Type"] == "application/json"
    assert posted["headers"][TOKEN_HEADER] == _ENDPOINT["token"]
    assert json.loads(posted["body"])["kind"] == "cascade"
    # Nothing reached the downloads folder, which is the entire point.
    assert export["settledDownloads"] == 0
    assert export["settledStatus"]["className"] == "status-bar ok"
    assert _ENDPOINT["path"] in export["settledStatus"]["text"]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param("refused", "wrong or missing builder token", id="the server said no"),
        pytest.param("network", "connection refused", id="the command was stopped"),
    ],
)
def test_a_failed_save_falls_back_to_the_download_rather_than_losing_the_file(
    tmp_path: Path, failure: str, expected: str
) -> None:
    """A cascade someone spent an hour assembling must survive a bad port.

    So every failure ends in the route that always works, and says which file
    could not be written and why -- a silent success would be the worst outcome
    here, and a silent failure only slightly better.
    """

    export = _drive_export(tmp_path, endpoint=_ENDPOINT, failure=failure)

    assert len(export["posted"]) == 1
    assert export["settledDownloads"] == 1
    assert json.loads(export["settledFile"])["kind"] == "cascade"
    status = export["settledStatus"]
    assert status["className"] == "status-bar warn"
    assert _ENDPOINT["path"] in status["text"]
    assert expected in status["text"]
    assert "Saved cascade.json" in status["text"]
