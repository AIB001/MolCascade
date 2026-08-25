"""Vendored weights: declaration, verification, resolution, and download.

The property under test throughout is *fails closed*.  An asset that is absent,
truncated, replaced, symlinked somewhere else, or served by a host having a bad
day must stop the caller with a message naming the fix -- never hand back a path
to bytes nobody checked.

The download tests drive :func:`molcascade.assets.fetch.fetch_asset` against a
fake ``urlopen`` rather than the network.  That is not only for speed: it is the
only way to exercise the failure paths that matter, since a well-behaved host
never serves the wrong digest on purpose.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.error
from pathlib import Path

import pytest

from molcascade.assets import catalog as catalog_module
from molcascade.assets import fetch as fetch_module
from molcascade.assets import store as store_module
from molcascade.assets.fetch import (
    fetch_asset,
    render_citation_document,
    render_manifest,
)
from molcascade.assets.models import (
    AssetFile,
    AssetKind,
    AssetSpec,
    AssetState,
    Citation,
    PayloadTrust,
)
from molcascade.assets.store import (
    asset_status,
    default_asset_root,
    is_asset_reference,
    resolve_path_or_reference,
    resolve_reference,
    split_reference,
)
from molcascade.errors import AssetError

_PAYLOAD = b"molcascade test payload\n"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()
_BLOB_SHA1 = hashlib.sha1(
    b"blob %d\0" % len(_PAYLOAD) + _PAYLOAD
).hexdigest()

_CITATION = Citation(
    title="A Fictional Model",
    authors="Doe, J.; Roe, R.",
    venue="Journal of Nothing",
    year=2024,
    doi="10.1000/fictional",
)


def _file(
    name: str = "weights.bin", *, payload: bytes = _PAYLOAD, **overrides: object
) -> AssetFile:
    fields: dict[str, object] = {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "url": f"https://example.invalid/{name}",
    }
    fields.update(overrides)
    return AssetFile(**fields)  # type: ignore[arg-type]


def _spec(*files: AssetFile, **overrides: object) -> AssetSpec:
    fields: dict[str, object] = {
        "id": "fictional",
        "kind": AssetKind.WEIGHTS,
        "display_name": "Fictional model weights",
        "summary": "Weights used only by the test suite.",
        "version": "1.0",
        "homepage": "https://example.invalid/",
        "license_spdx": "MIT",
        "files": files or (_file(),),
        "citations": (_CITATION,),
        "used_by": ("synthesis.fictional@0.1.0",),
    }
    fields.update(overrides)
    return AssetSpec(**fields)  # type: ignore[arg-type]


def _install(spec: AssetSpec, root: Path, contents: dict[str, bytes]) -> Path:
    directory = root / spec.id
    for name, payload in contents.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return directory


# --------------------------------------------------------------- declarations


@pytest.mark.parametrize(
    "name",
    [
        "../escape.bin",
        "models/../../escape.bin",
        "/absolute.bin",
        "C:/windows.bin",
        "models//double.bin",
        "./here.bin",
        "trailing/",
        "with space.bin",
    ],
)
def test_a_member_path_that_could_escape_the_asset_directory_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        _file(name)


def test_a_nested_member_path_is_accepted() -> None:
    entry = _file("models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz")

    assert entry.name.endswith(".json.gz")


@pytest.mark.parametrize(
    "digest",
    ["", "abc", _DIGEST.upper(), _DIGEST[:-1] + "g", _DIGEST + "0"],
)
def test_a_digest_that_is_not_64_lowercase_hex_characters_is_refused(digest: str) -> None:
    with pytest.raises(ValueError):
        _file(sha256=digest)


def test_a_git_object_id_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError):
        _file(git_blob_sha1="not-a-sha1" + "0" * 30)


@pytest.mark.parametrize(
    "url",
    ["http://example.invalid/w.bin", "ftp://example.invalid/w.bin", "file:///w.bin"],
)
def test_assets_may_only_be_declared_over_https(url: str) -> None:
    with pytest.raises(ValueError):
        _file(url=url)


def test_two_members_differing_only_in_case_are_refused() -> None:
    """On a case-insensitive filesystem the second would overwrite the first."""

    with pytest.raises(ValueError, match="duplicate file name"):
        _spec(_file("Weights.bin"), _file("weights.bin", payload=b"other"))


@pytest.mark.parametrize("asset_id", ["Fictional", "1fictional", "has space", "trailing-", ""])
def test_an_asset_id_that_is_not_a_safe_directory_name_is_refused(asset_id: str) -> None:
    with pytest.raises(ValueError):
        _spec(id=asset_id)


def test_the_declared_size_is_the_sum_of_its_files() -> None:
    spec = _spec(_file("a.bin"), _file("b.bin", payload=b"xy"))

    assert spec.total_bytes == len(_PAYLOAD) + 2


def test_asking_for_an_undeclared_member_raises() -> None:
    with pytest.raises(KeyError):
        _spec().file("absent.bin")


def test_a_citation_does_not_double_the_period_after_the_final_initial() -> None:
    citation = Citation(
        title="SCScore",
        authors="Coley, C. W.; Rogers, L.; Green, W. H.; Jensen, K. F.",
        venue="J. Chem. Inf. Model.",
        year=2018,
        doi="10.1021/acs.jcim.7b00622",
    )

    assert "Jensen, K. F. SCScore." in citation.reference
    assert ".." not in citation.reference.replace("J. Chem. Inf. Model.", "")
    assert citation.reference.endswith("DOI: 10.1021/acs.jcim.7b00622")


def test_a_citation_without_a_doi_ends_at_the_year() -> None:
    citation = Citation(title="T", authors="A", venue="V", year=2020)

    assert citation.reference == "A. T. V 2020."


# -------------------------------------------------------------- verification


def test_an_asset_that_was_never_downloaded_reports_missing(tmp_path: Path) -> None:
    status = asset_status(_spec(), root=tmp_path)

    assert status.state is AssetState.MISSING
    assert status.ready is False
    assert status.files[0].present is False


def test_an_asset_whose_bytes_match_reports_ready(tmp_path: Path) -> None:
    spec = _spec()
    _install(spec, tmp_path, {"weights.bin": _PAYLOAD})

    status = asset_status(spec, root=tmp_path)

    assert status.state is AssetState.READY
    assert status.files[0].verified is True


def test_a_partly_downloaded_asset_reports_incomplete(tmp_path: Path) -> None:
    spec = _spec(_file("a.bin"), _file("b.bin", payload=b"xy"))
    _install(spec, tmp_path, {"a.bin": _PAYLOAD})

    assert asset_status(spec, root=tmp_path).state is AssetState.INCOMPLETE


def test_a_file_of_the_right_size_but_the_wrong_bytes_reports_corrupt(
    tmp_path: Path,
) -> None:
    spec = _spec()
    tampered = bytes(len(_PAYLOAD))
    _install(spec, tmp_path, {"weights.bin": tampered})

    status = asset_status(spec, root=tmp_path)

    assert status.state is AssetState.CORRUPT
    assert "digest mismatch" in status.files[0].detail


def test_a_truncated_file_is_caught_on_size_before_it_is_hashed(tmp_path: Path) -> None:
    spec = _spec()
    _install(spec, tmp_path, {"weights.bin": _PAYLOAD[:-1]})

    status = asset_status(spec, root=tmp_path)

    assert status.state is AssetState.CORRUPT
    assert status.files[0].observed_size_bytes == len(_PAYLOAD) - 1
    assert "expected" in status.files[0].detail


def test_a_symlink_is_refused_even_when_it_points_at_the_right_bytes(
    tmp_path: Path,
) -> None:
    """Otherwise the verified path and the read path could be different files."""

    spec = _spec()
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(_PAYLOAD)
    directory = tmp_path / spec.id
    directory.mkdir(parents=True)
    try:
        (directory / "weights.bin").symlink_to(elsewhere)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("this filesystem does not support symbolic links")

    status = asset_status(spec, root=tmp_path)

    assert status.state is AssetState.CORRUPT
    assert "symbolic link" in status.files[0].detail


def test_a_directory_standing_in_for_a_file_is_not_verified(tmp_path: Path) -> None:
    spec = _spec()
    (tmp_path / spec.id / "weights.bin").mkdir(parents=True)

    assert asset_status(spec, root=tmp_path).ready is False


# ------------------------------------------------------------- the stamp cache


def test_verifying_writes_a_stamp_so_the_next_check_need_not_rehash(
    tmp_path: Path,
) -> None:
    spec = _spec()
    directory = _install(spec, tmp_path, {"weights.bin": _PAYLOAD})

    asset_status(spec, root=tmp_path)
    stamp = json.loads((directory / ".molcascade-verified.json").read_text())

    assert stamp["files"]["weights.bin"]["sha256"] == _DIGEST


def test_the_cache_is_consulted_rather_than_the_bytes_on_a_repeat_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _install(spec, tmp_path, {"weights.bin": _PAYLOAD})
    asset_status(spec, root=tmp_path)

    def explode(_: Path) -> tuple[str, int]:  # pragma: no cover - must not run
        raise AssertionError("a cached, unchanged file was hashed again")

    monkeypatch.setattr(store_module, "_digest", explode)

    assert asset_status(spec, root=tmp_path).ready is True


def test_a_deep_check_ignores_the_cache_and_reads_the_bytes(tmp_path: Path) -> None:
    """This is what catches an in-place edit that preserved size and mtime."""

    spec = _spec()
    directory = _install(spec, tmp_path, {"weights.bin": _PAYLOAD})
    asset_status(spec, root=tmp_path)
    target = directory / "weights.bin"
    original = os.stat(target)
    target.write_bytes(bytes(len(_PAYLOAD)))
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert asset_status(spec, root=tmp_path).ready is True
    assert asset_status(spec, root=tmp_path, deep=True).state is AssetState.CORRUPT


def test_a_changed_modification_time_invalidates_the_cache(tmp_path: Path) -> None:
    spec = _spec()
    directory = _install(spec, tmp_path, {"weights.bin": _PAYLOAD})
    asset_status(spec, root=tmp_path)
    target = directory / "weights.bin"
    target.write_bytes(bytes(len(_PAYLOAD)))
    # Set the timestamp explicitly: many filesystems record mtime at timer-tick
    # granularity, so two writes milliseconds apart can share one value and the
    # test would be asserting the clock rather than the cache.
    stamped = os.stat(target)
    os.utime(target, ns=(stamped.st_atime_ns, stamped.st_mtime_ns + 10**9))

    assert asset_status(spec, root=tmp_path).state is AssetState.CORRUPT


def test_an_unreadable_stamp_is_treated_as_a_cold_cache_not_an_error(
    tmp_path: Path,
) -> None:
    spec = _spec()
    directory = _install(spec, tmp_path, {"weights.bin": _PAYLOAD})
    (directory / ".molcascade-verified.json").write_text("{ this is not json")

    assert asset_status(spec, root=tmp_path).ready is True


# ------------------------------------------------------------------ resolution


@pytest.fixture
def catalogued(monkeypatch: pytest.MonkeyPatch) -> AssetSpec:
    """Make the fictional spec the only asset the store knows about."""

    spec = _spec()

    def lookup(asset_id: str) -> AssetSpec:
        if asset_id != spec.id:
            raise KeyError(asset_id)
        return spec

    monkeypatch.setattr(store_module, "asset_spec", lookup)
    monkeypatch.setattr(store_module, "iter_assets", lambda: (spec,))
    return spec


@pytest.mark.parametrize(
    "value",
    ["asset:", "asset:fictional", "asset:fictional/", "asset:/weights.bin", "asset:/"],
)
def test_a_malformed_reference_is_refused(value: str) -> None:
    with pytest.raises(AssetError, match=r"ASSET_REFERENCE_INVALID|asset reference"):
        split_reference(value)


def test_a_plain_path_is_not_mistaken_for_a_reference() -> None:
    assert is_asset_reference("/opt/weights.bin") is False
    assert is_asset_reference("asset:fictional/weights.bin") is True


def test_a_reference_splits_at_the_first_slash_only() -> None:
    assert split_reference("asset:scscore/models/a/b.gz") == (
        "scscore",
        "models/a/b.gz",
    )


def test_resolving_a_ready_asset_yields_an_absolute_verified_path(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    _install(catalogued, tmp_path, {"weights.bin": _PAYLOAD})

    path = resolve_reference("asset:fictional/weights.bin", root=tmp_path)

    assert path.is_absolute()
    assert path.read_bytes() == _PAYLOAD


def test_resolving_an_unknown_asset_lists_the_ones_that_exist(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    with pytest.raises(AssetError) as caught:
        resolve_reference("asset:imaginary/weights.bin", root=tmp_path)

    assert caught.value.code == "ASSET_UNKNOWN"
    assert "fictional" in (caught.value.hint or "")


def test_resolving_an_undeclared_member_lists_the_declared_ones(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    with pytest.raises(AssetError) as caught:
        resolve_reference("asset:fictional/not-declared.bin", root=tmp_path)

    assert caught.value.code == "ASSET_MEMBER_UNKNOWN"
    assert "weights.bin" in (caught.value.hint or "")


def test_resolving_a_missing_asset_names_the_command_that_would_fix_it(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    with pytest.raises(AssetError) as caught:
        resolve_reference("asset:fictional/weights.bin", root=tmp_path)

    assert caught.value.code == "ASSET_NOT_READY"
    assert "molcascade assets fetch fictional" in (caught.value.hint or "")
    assert "never downloads during a screening run" in (caught.value.hint or "")
    assert caught.value.context["expected_sha256"] == _DIGEST


def test_resolving_a_tampered_asset_fails_rather_than_returning_the_path(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    _install(catalogued, tmp_path, {"weights.bin": bytes(len(_PAYLOAD))})

    with pytest.raises(AssetError, match="digest mismatch"):
        resolve_reference("asset:fictional/weights.bin", root=tmp_path)


def test_resolution_never_downloads(tmp_path: Path, catalogued: AssetSpec) -> None:
    """The store module must not even be able to reach the network."""

    assert not hasattr(store_module, "urlopen")
    assert "urllib" not in dir(store_module)


def test_a_relative_path_in_a_configuration_is_refused() -> None:
    with pytest.raises(AssetError) as caught:
        resolve_path_or_reference("models/weights.bin")

    assert caught.value.code == "ASSET_PATH_NOT_ABSOLUTE"


def test_an_absolute_path_is_passed_through_unverified(tmp_path: Path) -> None:
    """A path the user typed is theirs to vouch for; only assets are pinned."""

    target = tmp_path / "mine.bin"

    assert resolve_path_or_reference(str(target)) == target


def test_an_asset_reference_still_resolves_through_the_combined_helper(
    tmp_path: Path, catalogued: AssetSpec
) -> None:
    _install(catalogued, tmp_path, {"weights.bin": _PAYLOAD})

    resolved = resolve_path_or_reference("asset:fictional/weights.bin", root=tmp_path)

    assert resolved.read_bytes() == _PAYLOAD


def test_the_asset_root_can_be_moved_with_an_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOLCASCADE_ASSET_ROOT", str(tmp_path / "elsewhere"))

    assert default_asset_root() == tmp_path / "elsewhere"


def test_without_an_override_assets_live_beside_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOLCASCADE_ASSET_ROOT", raising=False)

    assert default_asset_root().name == "vendor"


# ------------------------------------------------------------------ downloading


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _serve(
    monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]
) -> list[str]:
    """Answer downloads from a dictionary keyed by file name; record the calls."""

    requested: list[str] = []

    def urlopen(request: object, timeout: float = 0.0) -> _FakeResponse:
        url = request.full_url  # type: ignore[attr-defined]
        name = url.rsplit("/", 1)[-1]
        requested.append(name)
        if name not in payloads:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        return _FakeResponse(payloads[name])

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", urlopen)
    return requested


def test_a_successful_fetch_verifies_stamps_and_documents_the_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(_file(git_blob_sha1=_BLOB_SHA1))
    _serve(monkeypatch, {"weights.bin": _PAYLOAD})

    installed = fetch_asset(spec, root=tmp_path)

    directory = tmp_path / spec.id
    assert installed == ("weights.bin",)
    assert (directory / "weights.bin").read_bytes() == _PAYLOAD
    assert (directory / ".molcascade-verified.json").is_file()
    assert "10.1000/fictional" in (directory / "CITATION.md").read_text()
    assert asset_status(spec, root=tmp_path, deep=True).ready is True


def test_a_second_fetch_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    requested = _serve(monkeypatch, {"weights.bin": _PAYLOAD})
    fetch_asset(spec, root=tmp_path)
    requested.clear()

    assert fetch_asset(spec, root=tmp_path) == ()
    assert requested == []


def test_forcing_a_fetch_downloads_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    requested = _serve(monkeypatch, {"weights.bin": _PAYLOAD})
    fetch_asset(spec, root=tmp_path)
    requested.clear()

    assert fetch_asset(spec, root=tmp_path, force=True) == ("weights.bin",)
    assert requested == ["weights.bin"]


def test_an_interrupted_fetch_resumes_with_only_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(_file("a.bin"), _file("b.bin", payload=b"xy"))
    _install(spec, tmp_path, {"a.bin": _PAYLOAD})
    requested = _serve(monkeypatch, {"a.bin": _PAYLOAD, "b.bin": b"xy"})

    assert fetch_asset(spec, root=tmp_path) == ("b.bin",)
    assert requested == ["b.bin"]


def test_bytes_whose_digest_disagrees_are_deleted_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _serve(monkeypatch, {"weights.bin": b"substituted content of the same length"})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code in ("ASSET_DIGEST_MISMATCH", "ASSET_DOWNLOAD_SIZE_MISMATCH")
    assert list((tmp_path / spec.id).glob("*")) == []


def test_a_digest_mismatch_leaves_no_partial_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same length, different bytes: the digest is the only thing that catches it."""

    spec = _spec()
    _serve(monkeypatch, {"weights.bin": bytes(len(_PAYLOAD))})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code == "ASSET_DIGEST_MISMATCH"
    assert list((tmp_path / spec.id).glob("*")) == []


def test_a_git_object_id_that_disagrees_rejects_bytes_the_sha256_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent checks; agreeing with one of them is not enough."""

    spec = _spec(_file(git_blob_sha1="0" * 40))
    _serve(monkeypatch, {"weights.bin": _PAYLOAD})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code == "ASSET_GIT_SHA_MISMATCH"
    assert list((tmp_path / spec.id).glob("*")) == []


def test_a_response_larger_than_declared_is_abandoned_mid_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _serve(monkeypatch, {"weights.bin": b"x" * (len(_PAYLOAD) + 10_000)})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code == "ASSET_DOWNLOAD_TOO_LARGE"


def test_a_short_response_is_reported_as_a_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _serve(monkeypatch, {"weights.bin": _PAYLOAD[:-3]})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code == "ASSET_DOWNLOAD_SIZE_MISMATCH"
    assert caught.value.retryable is True


def test_an_http_error_is_retryable_and_names_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _serve(monkeypatch, {})

    with pytest.raises(AssetError) as caught:
        fetch_asset(spec, root=tmp_path)

    assert caught.value.code == "ASSET_DOWNLOAD_FAILED"
    assert caught.value.retryable is True
    assert caught.value.context["status"] == 404


def test_a_network_failure_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(request: object, timeout: float = 0.0) -> _FakeResponse:
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(AssetError) as caught:
        fetch_asset(_spec(), root=tmp_path)

    assert caught.value.retryable is True


def test_progress_is_reported_against_the_declared_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    _serve(monkeypatch, {"weights.bin": _PAYLOAD})
    seen: list[tuple[str, int, int]] = []

    fetch_asset(spec, root=tmp_path, progress=lambda *args: seen.append(args))

    assert seen[-1] == ("weights.bin", len(_PAYLOAD), len(_PAYLOAD))


# ------------------------------------------------------------- documentation


def test_the_citation_document_carries_paper_licence_and_digests() -> None:
    document = render_citation_document(_spec())

    assert "A Fictional Model" in document
    assert "10.1000/fictional" in document
    assert "```bibtex" in document
    assert _DIGEST in document
    assert "asset:fictional/weights.bin" in document
    assert "synthesis.fictional@0.1.0" in document


def test_a_data_payload_carries_no_execution_warning() -> None:
    document = render_citation_document(_spec())

    assert "This payload is executable" not in document


def test_an_executable_payload_says_so_prominently() -> None:
    """A pickle or a SavedModel runs code; a digest does not change that."""

    spec = _spec(trust=PayloadTrust.EXECUTABLE)

    document = render_citation_document(spec)

    assert "**This payload is executable.**" in document
    assert "does not make deserialising them safe" in document


def test_the_manifest_indexes_every_asset() -> None:
    manifest = render_manifest([_spec()])

    assert "[`fictional`](fictional/CITATION.md)" in manifest
    assert "MOLCASCADE_ASSET_ROOT" in manifest
    assert "Download. Ever." in manifest


def test_the_provenance_documents_do_not_depend_on_where_they_were_written() -> None:
    """Both files are checked in, so both have to read the same everywhere.

    They used to embed the absolute install directory.  That is defensible on
    the machine that ran ``assets fetch`` and wrong on every other one: the
    committed copy named a path that did not exist for any reader, and
    regenerating it produced a spurious diff for whoever's home directory was
    different.  Neither document needs the path -- each sits in the directory
    it describes.
    """

    spec = _spec()

    assert render_citation_document(spec) == render_citation_document(spec)
    assert render_manifest([spec]) == render_manifest([spec])

    # No absolute path of any shape: POSIX home directories, mounted Windows
    # drives under WSL, and bare drive letters.
    for name, document in (
        ("CITATION.md", render_citation_document(spec)),
        ("MANIFEST.md", render_manifest([spec])),
    ):
        offenders = re.findall(r"(?:/home/|/Users/|/mnt/[a-z]/|[A-Z]:\\\\)\S*", document)
        assert not offenders, f"{name} embeds a machine-specific path: {offenders}"


# ---------------------------------------------------------- the real catalogue


def test_every_shipped_asset_declaration_is_valid_and_cited() -> None:
    specs = list(catalog_module.iter_assets())

    assert specs
    for spec in specs:
        assert spec.citations, f"{spec.id} has no citation"
        assert spec.used_by, f"{spec.id} is declared but nothing uses it"
        assert all(entry.url.startswith("https://") for entry in spec.files)


def test_asset_ids_are_unique() -> None:
    ids = [spec.id for spec in catalog_module.iter_assets()]

    assert len(ids) == len(set(ids))


def test_looking_up_an_asset_that_does_not_exist_raises_key_error() -> None:
    with pytest.raises(KeyError):
        catalog_module.asset_spec("no-such-asset")


def test_scscore_names_the_plugin_that_reads_it() -> None:
    spec = catalog_module.asset_spec("scscore")

    assert spec.trust is PayloadTrust.DATA
    assert "synthesis.scscore@0.1.0" in spec.used_by
    assert any(c.doi == "10.1021/acs.jcim.7b00622" for c in spec.citations)
