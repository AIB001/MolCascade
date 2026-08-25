from __future__ import annotations

import contextlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError

from molcascade.artifacts import (
    ArtifactConflictError,
    ArtifactDatasetRef,
    ArtifactFile,
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactOutput,
    ArtifactProducer,
    CanonicalizationError,
    InvalidStagingError,
    LocalArtifactStore,
    canonical_json_bytes,
    make_artifact_id,
    make_cache_key,
)


@pytest.fixture
def producer() -> ArtifactProducer:
    return ArtifactProducer(plugin_id="builtin.test", plugin_version="1.0.0")


@pytest.fixture
def empty_file() -> ArtifactFile:
    return ArtifactFile(
        path="tables/empty.txt",
        size_bytes=0,
        checksum="sha256:" + "0" * 64,
        media_type="text/plain",
    )


def _output(
    port: str,
    contract_id: str,
    *file_paths: str,
    metadata: dict[str, object] | None = None,
) -> ArtifactOutput:
    return ArtifactOutput(
        port=port,
        contract_id=contract_id,
        file_paths=file_paths,
        metadata={} if metadata is None else metadata,
    )


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    left = {"name": "筛选", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "name": "筛选"}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert make_cache_key(left) == make_cache_key(right)


def test_canonical_hash_rejects_non_finite_numbers() -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json_bytes({"score": float("nan")})


def test_canonical_hash_rejects_recursive_values() -> None:
    recursive: dict[str, object] = {}
    recursive["self"] = recursive

    with pytest.raises(CanonicalizationError, match="recursive"):
        canonical_json_bytes(recursive)


def test_cache_key_and_artifact_id_have_distinct_namespaces() -> None:
    content = {"same": "canonical content"}

    cache_key = make_cache_key(content)
    artifact_id = make_artifact_id(content)

    assert cache_key.startswith("cache:sha256:")
    assert artifact_id.startswith("artifact:sha256:")
    assert cache_key.removeprefix("cache:") == artifact_id.removeprefix("artifact:")


def test_created_at_and_cache_key_do_not_change_artifact_id(
    producer: ArtifactProducer,
    empty_file: ArtifactFile,
) -> None:
    first_time = datetime(2026, 8, 22, tzinfo=UTC)
    second_time = first_time + timedelta(days=1)
    first = ArtifactManifest.create(
        kind="test-output",
        producer=producer,
        files=[empty_file],
        created_at=first_time,
        cache_key=make_cache_key({"invocation": 1}),
    )
    second = ArtifactManifest.create(
        kind="test-output",
        producer=producer,
        files=[empty_file],
        created_at=second_time,
        cache_key=make_cache_key({"invocation": 2}),
    )

    assert first.artifact_id == second.artifact_id
    assert first.created_at != second.created_at
    assert first.cache_key != second.cache_key


def test_manifest_ports_are_identity_bearing_and_produce_exact_dataset_refs(
    producer: ArtifactProducer,
) -> None:
    parents = ArtifactFile(
        path="datasets/parents/part-00000.parquet",
        size_bytes=0,
        checksum="sha256:" + "1" * 64,
    )
    decisions = ArtifactFile(
        path="datasets/decisions/part-00000.parquet",
        size_bytes=0,
        checksum="sha256:" + "2" * 64,
    )
    outputs = (
        _output(
            "decisions",
            "decision/v1",
            decisions.path,
            metadata={"row_count": 3},
        ),
        _output("primary", "parent/v1", parents.path),
    )

    manifest = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=[parents, decisions],
        outputs=outputs,
    )
    changed_port_metadata = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=[parents, decisions],
        outputs=(
            _output(
                "decisions",
                "decision/v1",
                decisions.path,
                metadata={"row_count": 4},
            ),
            _output("primary", "parent/v1", parents.path),
        ),
    )

    assert tuple(output.port for output in manifest.outputs) == ("decisions", "primary")
    assert manifest.artifact_id != changed_port_metadata.artifact_id
    primary = manifest.dataset_ref("primary")
    assert primary == ArtifactDatasetRef(
        artifact_id=manifest.artifact_id,
        artifact_kind="stage-output",
        port="primary",
        contract_id="parent/v1",
        file_paths=(parents.path,),
    )
    assert manifest.validate_dataset_ref(primary) == manifest.output("primary")
    with pytest.raises(TypeError, match="immutable"):
        manifest.output("decisions").metadata["row_count"] = 4

    changed_contract = primary.model_copy(update={"contract_id": "decision/v1"})
    with pytest.raises(ValueError, match="does not match manifest output"):
        manifest.validate_dataset_ref(changed_contract)
    changed_paths = primary.model_copy(
        update={"file_paths": ("datasets/other/part-00000.parquet",)}
    )
    with pytest.raises(ValueError, match="does not match manifest output"):
        manifest.validate_dataset_ref(changed_paths)
    with pytest.raises(ValueError, match="no output port"):
        manifest.dataset_ref("missing")


def test_manifest_ports_must_claim_the_complete_non_overlapping_file_inventory(
    producer: ArtifactProducer,
) -> None:
    files = [
        ArtifactFile(
            path=f"datasets/{name}.parquet",
            size_bytes=0,
            checksum="sha256:" + digit * 64,
        )
        for name, digit in (("parents", "1"), ("decisions", "2"))
    ]

    with pytest.raises(ValidationError, match="unclaimed"):
        ArtifactManifest.create(
            kind="stage-output",
            producer=producer,
            files=files,
            outputs=(_output("primary", "parent/v1", files[0].path),),
        )
    with pytest.raises(ValidationError, match="claimed by multiple"):
        ArtifactManifest.create(
            kind="stage-output",
            producer=producer,
            files=files,
            outputs=(
                _output("primary", "parent/v1", files[0].path, files[1].path),
                _output("decisions", "decision/v1", files[1].path),
            ),
        )


def test_input_lineage_distinguishes_ports_of_the_same_artifact(
    producer: ArtifactProducer,
    empty_file: ArtifactFile,
) -> None:
    upstream_files = [
        ArtifactFile(
            path=f"datasets/{name}.parquet",
            size_bytes=0,
            checksum="sha256:" + digit * 64,
        )
        for name, digit in (("parents", "1"), ("mapping", "2"))
    ]
    upstream = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=upstream_files,
        outputs=(
            _output("mapping", "parent_source_map/v1", upstream_files[1].path),
            _output("primary", "parent/v1", upstream_files[0].path),
        ),
    )
    downstream_output = (_output("primary", "artifact_payload/v1", empty_file.path),)

    from_parents = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=[empty_file],
        outputs=downstream_output,
        inputs=[upstream.dataset_ref("primary")],
    )
    from_mapping = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=[empty_file],
        outputs=downstream_output,
        inputs=[upstream.dataset_ref("mapping")],
    )

    assert from_parents.artifact_id != from_mapping.artifact_id
    assert make_cache_key({"inputs": [upstream.dataset_ref("primary")]}) != make_cache_key(
        {"inputs": [upstream.dataset_ref("mapping")]}
    )
    with pytest.raises(ValidationError):
        ArtifactManifest.create(
            kind="stage-output",
            producer=producer,
            files=[empty_file],
            outputs=downstream_output,
            inputs=[upstream.as_ref()],  # type: ignore[list-item]
        )


def test_manifest_identity_metadata_is_deeply_immutable(
    producer: ArtifactProducer,
    empty_file: ArtifactFile,
) -> None:
    manifest = ArtifactManifest.create(
        kind="test-output",
        producer=producer,
        files=[empty_file],
        metadata={"nested": {"thresholds": [0.1, 0.2]}},
    )

    with pytest.raises(TypeError, match="immutable"):
        manifest.metadata["new"] = True
    nested = manifest.metadata["nested"]
    assert isinstance(nested, dict)
    thresholds = nested["thresholds"]
    assert isinstance(thresholds, list)
    with pytest.raises(TypeError, match="immutable"):
        thresholds.append(0.3)


def test_manifest_rejects_pre_port_v1_documents(
    producer: ArtifactProducer,
    empty_file: ArtifactFile,
) -> None:
    manifest = ArtifactManifest.create(
        kind="stage-output",
        producer=producer,
        files=[empty_file],
        outputs=(_output("primary", "artifact_payload/v1", empty_file.path),),
    )
    document = manifest.model_dump(mode="json")
    document["manifest_version"] = "molcascade.artifact/v1"

    with pytest.raises(ValidationError, match=re.escape("molcascade.artifact/v2")):
        ArtifactManifest.model_validate(document)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "..",
        "../outside.parquet",
        "data/../../outside.parquet",
        "/absolute/data.parquet",
        "C:/absolute/data.parquet",
        r"C:\absolute\data.parquet",
        r"..\outside.parquet",
        "data//part.parquet",
        "data/./part.parquet",
        "data/part.parquet/",
        "data:stream",
        "data/query?.json",
        "NUL.txt",
    ],
)
def test_manifest_rejects_unsafe_relative_paths(unsafe_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(
            path=unsafe_path,
            size_bytes=0,
            checksum="sha256:" + "0" * 64,
        )


def test_uncommitted_staging_is_not_visible(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "partial.txt").write_text("not complete", encoding="utf-8")

    assert store.list_refs() == ()
    assert not any(store.artifacts_root.iterdir())
    assert staging.exists()


def test_commit_verifies_and_duplicate_content_is_reused(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first_stage = store.begin_staging()
    (first_stage / "data").mkdir()
    (first_stage / "data" / "rows.jsonl").write_text('{"id":1}\n', encoding="utf-8")

    first = store.commit(
        first_stage,
        kind="molecule-table",
        producer=producer,
        metadata={"rows": 1},
        file_paths=["data/rows.jsonl"],
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
    )
    assert not first_stage.exists()
    assert store.verify(first.artifact_id) == first

    second_stage = store.begin_staging()
    (second_stage / "data").mkdir()
    (second_stage / "data" / "rows.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    duplicate = store.commit(
        second_stage,
        kind="molecule-table",
        producer=producer,
        metadata={"rows": 1},
        file_paths=["data/rows.jsonl"],
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert duplicate == first
    assert not second_stage.exists()
    assert store.list_refs() == (first.as_ref(),)


def test_concurrent_duplicate_commits_converge_on_one_artifact(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    stages = [store.begin_staging(), store.begin_staging()]
    for staging in stages:
        (staging / "payload.txt").write_text("same bytes", encoding="utf-8")

    def publish(staging: Path) -> ArtifactManifest:
        return store.commit(staging, kind="test", producer=producer)

    with ThreadPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(publish, stages))

    assert manifests[0].artifact_id == manifests[1].artifact_id
    assert store.list_refs() == (manifests[0].as_ref(),)
    assert all(not staging.exists() for staging in stages)


def test_publication_lock_keeps_its_descriptor_without_closing_unrelated_files(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    destination = store.artifacts_root / ("a" * 64)
    probe = tmp_path / "probe.txt"
    probe.write_text("open", encoding="utf-8")
    probe_descriptor: int | None = None

    try:
        with store._publication_lock("a" * 64, destination):
            # The advisory lock descriptor remains open for the whole critical
            # section.  A separately opened descriptor must remain untouched.
            probe_descriptor = os.open(probe, os.O_RDONLY)
        assert probe_descriptor is not None
        os.fstat(probe_descriptor)
    finally:
        if probe_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(probe_descriptor)


def test_visible_destination_does_not_bypass_an_active_publication_lock(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    digest = "b" * 64
    destination = store.artifacts_root / digest
    first_published = Event()
    release_first = Event()
    second_entered = Event()

    def first() -> None:
        with store._publication_lock(digest, destination):
            destination.mkdir()
            first_published.set()
            assert release_first.wait(timeout=2)

    def second() -> None:
        assert first_published.wait(timeout=2)
        with store._publication_lock(digest, destination):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first)
        assert first_published.wait(timeout=2)
        second_future = pool.submit(second)
        assert not second_entered.wait(timeout=0.2)
        release_first.set()
        first_future.result(timeout=2)
        second_future.result(timeout=2)
    assert second_entered.is_set()


def test_stale_publication_lock_file_does_not_block_crash_recovery(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    payload = staging / "payload.txt"
    payload.write_text("recoverable", encoding="utf-8")
    file_entry = store._describe_file(staging, "payload.txt")
    requested = ArtifactManifest.create(
        kind="test",
        producer=producer,
        files=(file_entry,),
    )
    digest = requested.artifact_id.rsplit(":", 1)[1]
    stale_lock = store.lock_root / f"{digest}.lock"
    stale_lock.write_text("pid=crashed\n", encoding="ascii")

    published = store.commit_manifest(staging, requested)

    assert published == requested
    assert stale_lock.is_file()
    assert store.verify(published.artifact_id) == published


def test_integrity_verification_detects_file_tampering(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "payload.txt").write_text("trusted", encoding="utf-8")
    manifest = store.commit(staging, kind="test", producer=producer)
    committed_file = store.resolve_file(manifest.artifact_id, "payload.txt")

    committed_file.write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        store.verify(manifest.artifact_id)


def test_integrity_verification_detects_manifest_identity_tampering(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "payload.txt").write_text("trusted", encoding="utf-8")
    manifest = store.commit(staging, kind="test", producer=producer)
    manifest_path = store.artifacts_root / manifest.artifact_id.rsplit(":", 1)[1] / "artifact.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["kind"] = "altered"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="invalid manifest"):
        store.verify(manifest.artifact_id)


def test_store_rejects_forged_port_contract_or_paths(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "parents.txt").write_text("parent", encoding="utf-8")
    manifest = store.commit(
        staging,
        kind="stage-output",
        producer=producer,
        outputs={
            "primary": {
                "contract_id": "parent/v1",
                "file_paths": ("parents.txt",),
            }
        },
    )
    trusted = manifest.dataset_ref("primary")

    assert store.resolve_dataset(trusted) == store.artifact_directory(manifest.artifact_id)
    for forged in (
        trusted.model_copy(update={"contract_id": "decision/v1"}),
        trusted.model_copy(update={"file_paths": ("other.txt",)}),
        trusted.model_copy(update={"artifact_kind": "other-bundle"}),
    ):
        with pytest.raises(ArtifactIntegrityError, match="does not match artifact manifest"):
            store.resolve_dataset(forged)


def test_integrity_verification_detects_output_port_tampering(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "payload.txt").write_text("trusted", encoding="utf-8")
    manifest = store.commit(
        staging,
        kind="stage-output",
        producer=producer,
        outputs={
            "primary": {
                "contract_id": "parent/v1",
                "file_paths": ("payload.txt",),
            }
        },
    )
    manifest_path = store.artifacts_root / manifest.artifact_id.rsplit(":", 1)[1] / "artifact.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["outputs"][0]["contract_id"] = "decision/v1"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="invalid manifest"):
        store.verify(manifest.artifact_id)


def test_declared_file_set_must_match_staging(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "declared.txt").write_text("one", encoding="utf-8")
    (staging / "undeclared.txt").write_text("two", encoding="utf-8")

    with pytest.raises(InvalidStagingError, match="unexpected"):
        store.commit(
            staging,
            kind="test",
            producer=producer,
            file_paths=["declared.txt"],
        )
    assert staging.exists()


def test_store_ports_must_match_the_complete_staging_inventory(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "parents.parquet").write_bytes(b"not inspected because union fails")
    (staging / "decisions.parquet").write_bytes(b"undeclared")

    with pytest.raises(InvalidStagingError, match="port files do not match staging"):
        store.commit(
            staging,
            kind="stage-output",
            producer=producer,
            outputs={
                "primary": {
                    "contract_id": "parent/v1",
                    "file_paths": ("parents.parquet",),
                }
            },
        )
    assert staging.exists()


def test_store_rejects_symlinked_staged_files(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of staging", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "escape.txt").symlink_to(outside)

    with pytest.raises(InvalidStagingError, match="symlink"):
        store.commit(staging, kind="test", producer=producer)


def test_store_rejects_hard_linked_staged_files(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("mutable alias", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    os.link(outside, staging / "payload.txt")

    with pytest.raises(InvalidStagingError, match="hard-linked"):
        store.commit(staging, kind="test", producer=producer)

    assert store.list_refs() == ()


def test_oversized_manifest_is_never_published(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with (
        pytest.raises(InvalidStagingError, match="size limit"),
        store.staging_area() as staging,
    ):
        (staging / "payload.txt").write_text("small", encoding="utf-8")
        store.commit(
            staging,
            kind="test",
            producer=producer,
            metadata={"oversized": "x" * (17 * 1024 * 1024)},
        )

    assert store.list_refs() == ()
    assert list(store.staging_root.iterdir()) == []


def test_staged_retry_manifest_cannot_replace_requested_cache_key(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "payload.txt").write_text("stable", encoding="utf-8")
    file_entry = store._describe_file(staging, "payload.txt")
    requested = ArtifactManifest.create(
        kind="test",
        producer=producer,
        files=(file_entry,),
        cache_key=make_cache_key({"invocation": "requested"}),
    )
    forged_retry = ArtifactManifest.create(
        kind="test",
        producer=producer,
        files=(file_entry,),
        cache_key=make_cache_key({"invocation": "forged"}),
        created_at=requested.created_at,
    )
    assert forged_retry.artifact_id == requested.artifact_id
    (staging / "artifact.json").write_bytes(forged_retry.to_json_bytes())

    with pytest.raises(InvalidStagingError, match="cache_key"):
        store.commit_manifest(staging, requested)

    assert store.list_refs() == ()


@pytest.mark.parametrize("child", [".staging", ".locks", "artifacts"])
def test_store_initialization_rejects_symlinked_child_roots(
    tmp_path: Path,
    child: str,
) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / child).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        LocalArtifactStore(root)


def test_corrupt_existing_artifact_is_never_overwritten(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first_stage = store.begin_staging()
    (first_stage / "payload.txt").write_text("stable", encoding="utf-8")
    manifest = store.commit(first_stage, kind="test", producer=producer)
    committed = store.resolve_file(manifest.artifact_id, "payload.txt")
    committed.write_text("broken", encoding="utf-8")

    retry_stage = store.begin_staging()
    (retry_stage / "payload.txt").write_text("stable", encoding="utf-8")
    with pytest.raises(ArtifactConflictError, match="refusing to overwrite"):
        store.commit(retry_stage, kind="test", producer=producer)

    assert retry_stage.exists()
    assert committed.read_text(encoding="utf-8") == "broken"

    committed.write_text("stable", encoding="utf-8")
    retried = store.commit(retry_stage, kind="test", producer=producer)
    assert retried == manifest
    assert not retry_stage.exists()


def test_resolve_file_revalidates_user_path(
    tmp_path: Path,
    producer: ArtifactProducer,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    staging = store.begin_staging()
    (staging / "payload.txt").write_text("stable", encoding="utf-8")
    manifest = store.commit(staging, kind="test", producer=producer)

    with pytest.raises(ValueError, match="traverse"):
        store.resolve_file(manifest.artifact_id, "../artifact.json")
