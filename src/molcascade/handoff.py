"""Materialize a verified shortlist artifact for downstream local software."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from molcascade.artifacts import ArtifactDatasetRef
from molcascade.contracts import (
    PARENT_SOURCE_MAP_V1,
    RAW_MOLECULE_V1,
    RAW_MOLECULE_V2,
    SHORTLIST_EXPORT_V1,
)
from molcascade.errors import PipelineError
from molcascade.io.atomic import DestinationExistsError, atomic_publish_file
from molcascade.io.parquet import iter_parquet_batches
from molcascade.runtime import LocalRunner, RunState, RunStatus

#: Contracts a source reader may emit.  Both carry ``source_candidate_id``.
_SOURCE_CONTRACTS = frozenset({RAW_MOLECULE_V1.id, RAW_MOLECULE_V2.id})

#: Batch size for the two projected scans that build the identifier index.
#: Larger than the export batch because only two narrow string columns are read.
_INDEX_BATCH_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class MaterializedShortlist:
    path: Path
    record_format: str
    row_count: int
    sha256: str
    source_artifact_id: str
    export_spec_id: str
    #: Rows carrying an identifier from the user's library.  Zero when the
    #: library had no identifier column, which is also when no column is added.
    identified_count: int = 0


def _property_value(value: str) -> str:
    """Keep an SDF property single-line without changing molecular structure."""

    return value.replace("\r", " ").replace("\n", " ")


def _field_value(value: str) -> str:
    """Keep a library identifier inside one tab-separated field.

    Identifiers come from a spreadsheet cell, so they can hold anything a user
    typed -- including a tab.  Passing one through unchanged would silently add
    a column to that row and misalign every parser downstream.
    """

    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _dataset_files(runner: LocalRunner, ref: ArtifactDatasetRef) -> tuple[Path, ...]:
    """The verified files behind one port-qualified dataset view."""

    root = runner.store.resolve_dataset(ref, verify=True)
    return tuple(
        root.joinpath(*PurePosixPath(relative).parts) for relative in ref.file_paths
    )


def _ports_with_contract(
    runner: LocalRunner,
    state: RunState,
    contract_ids: Collection[str],
) -> tuple[ArtifactDatasetRef, ...]:
    """Every dataset in the run whose manifest declares one of ``contract_ids``.

    Stages are matched by the contract they emit rather than by plugin key, so
    this keeps working when the cascade selects a different reader or a
    different standardizer -- which is the entire point of letting the user
    assemble the flow.  Manifests are read without re-hashing; only the datasets
    actually opened are verified, by :func:`_dataset_files`.
    """

    refs: list[ArtifactDatasetRef] = []
    seen: set[str] = set()
    for stage in state.stages:
        if stage.output_ref is None or stage.output_ref.artifact_id in seen:
            continue
        seen.add(stage.output_ref.artifact_id)
        manifest = runner.store.get_manifest(stage.output_ref.artifact_id)
        for output in manifest.outputs:
            if output.contract_id in contract_ids:
                refs.append(manifest.dataset_ref(output.port))
    return tuple(refs)


def _identifier_index(
    runner: LocalRunner,
    state: RunState,
    wanted: Collection[str],
) -> dict[str, str]:
    """Map each shortlisted parent back to the identifier its library row carried.

    A shortlist keyed only by content hash cannot be joined to the spreadsheet it
    came from, which is what anyone screening a named library actually needs.
    The identifier is already recorded -- ``source_candidate_id`` on the raw
    records, and ``parent_source_map/v1`` links those records to parents -- so
    this is a join over data the run already holds, not new information.  Doing
    it here rather than threading a column through the cascade leaves every
    contract and every plugin untouched.

    Both scans are bounded by ``wanted``: one PRIMARY mapping row exists per
    parent, so the intermediate index never exceeds the shortlist, whatever the
    size of the library behind it.

    A parent that several library rows collapsed onto is named by exactly one of
    them -- the standardizer's PRIMARY, chosen deterministically but by record
    digest rather than by position in the file.  The occurrences it stands in for
    are in the run's decision log as ``EXACT_DUPLICATE``; they are left out of a
    handoff file, which is a handoff and not an audit trail.
    """

    if not wanted:
        return {}
    mapping_refs = _ports_with_contract(runner, state, {PARENT_SOURCE_MAP_V1.id})
    source_refs = _ports_with_contract(runner, state, _SOURCE_CONTRACTS)
    if not mapping_refs or not source_refs:
        # A pipeline may legitimately have neither -- one assembled by hand, or
        # one resumed from parents.  No identifier is not an error; the file is
        # simply written the way it was before identifiers existed.
        return {}

    origin: dict[str, str] = {}
    for ref in mapping_refs:
        for path in _dataset_files(runner, ref):
            for batch in iter_parquet_batches(
                path,
                columns=["source_record_id", "parent_id", "relation"],
                batch_size=_INDEX_BATCH_SIZE,
            ):
                columns = batch.to_pydict()
                for source_id, parent_id, relation in zip(
                    columns["source_record_id"],
                    columns["parent_id"],
                    columns["relation"],
                    strict=True,
                ):
                    if relation == "PRIMARY" and parent_id in wanted:
                        origin[source_id] = parent_id
    if not origin:
        return {}

    identifiers: dict[str, str] = {}
    for ref in source_refs:
        for path in _dataset_files(runner, ref):
            for batch in iter_parquet_batches(
                path,
                columns=["source_record_id", "source_candidate_id"],
                batch_size=_INDEX_BATCH_SIZE,
            ):
                columns = batch.to_pydict()
                for source_id, candidate_id in zip(
                    columns["source_record_id"],
                    columns["source_candidate_id"],
                    strict=True,
                ):
                    # An empty cell reads as "" from a CSV and as None from a
                    # worksheet; neither identifies anything.
                    if not candidate_id:
                        continue
                    parent_id = origin.get(source_id)
                    if parent_id is not None:
                        identifiers[parent_id] = candidate_id
    return identifiers


def materialize_shortlist(
    runner: LocalRunner,
    run_id: str,
    output: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> MaterializedShortlist:
    """Stream the final export-record port into one portable SMILES or SDF file."""

    state = runner.load_run(run_id)
    if state.status is not RunStatus.SUCCEEDED or state.output_ref is None:
        raise PipelineError(
            "only a successful run with a final output can be materialized",
            code="HANDOFF_RUN_NOT_SUCCEEDED",
            context={"run_id": run_id, "status": state.status.value},
        )
    manifest = runner.store.verify(state.output_ref.artifact_id)
    candidates = [
        item for item in manifest.outputs if item.contract_id == SHORTLIST_EXPORT_V1.id
    ]
    if len(candidates) != 1:
        raise PipelineError(
            "final stage does not expose exactly one shortlist export-record port",
            code="HANDOFF_EXPORT_PORT_MISSING",
            hint="End the pipeline with a MolCascade exporter plugin.",
            context={
                "run_id": run_id,
                "artifact_id": state.output_ref.artifact_id,
                "matching_ports": len(candidates),
            },
        )
    declared = candidates[0]
    suggested = declared.metadata.get("suggested_filename")
    if not isinstance(suggested, str) or not suggested:
        suggested = f"{run_id}-shortlist.smi"
    target = Path(suggested if output is None else output).expanduser()
    expected_format = declared.metadata.get("record_format")
    suffix = target.suffix.casefold()
    if expected_format == "SMILES_TSV" and suffix not in {".smi", ".smiles", ".tsv"}:
        raise PipelineError(
            "SMILES shortlist output must use .smi, .smiles, or .tsv",
            code="HANDOFF_OUTPUT_SUFFIX_INVALID",
            context={"record_format": expected_format, "path": str(target)},
        )
    if expected_format == "SDF_MOLBLOCK" and suffix not in {".sdf", ".sd"}:
        raise PipelineError(
            "SDF shortlist output must use .sdf or .sd",
            code="HANDOFF_OUTPUT_SUFFIX_INVALID",
            context={"record_format": expected_format, "path": str(target)},
        )
    dataset_ref = manifest.dataset_ref(declared.port)
    # ``_dataset_files`` resolves with ``verify=True``, which has already
    # validated manifest checksums, containment, regular-file status, symlink
    # components and the exact port-qualified file inventory.
    files = _dataset_files(runner, dataset_ref)
    # One projected pass to learn which parents were shortlisted, so the
    # identifier join can be bounded by the shortlist rather than the library.
    # Reading ``parent_id`` alone keeps this cheap even for SDF records.
    shortlisted: set[str] = set()
    for source in files:
        for batch in iter_parquet_batches(
            source, columns=["parent_id"], batch_size=_INDEX_BATCH_SIZE
        ):
            shortlisted.update(batch.column("parent_id").to_pylist())
    identifiers = _identifier_index(runner, state, shortlisted)
    # The column appears only when the library actually carried identifiers.
    # Otherwise every row would end in an empty field, changing the output of
    # every existing configuration for no gain.
    include_identifier = bool(identifiers)
    identified_count = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    row_count = 0
    observed_format: str | None = None
    export_spec_id: str | None = None
    plugin_config = manifest.metadata.get("plugin_config", {})
    include_header = (
        isinstance(plugin_config, dict)
        and plugin_config.get("include_header", True) is True
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if expected_format == "SMILES_TSV" and include_header:
                names = ["SMILES", "parent_id"]
                if include_identifier:
                    names.append("source_id")
                header = ("\t".join(names) + "\n").encode("utf-8")
                stream.write(header)
                digest.update(header)
            for source in files:
                for batch in iter_parquet_batches(source, batch_size=16_384):
                    SHORTLIST_EXPORT_V1.validate(batch)
                    for row in batch.to_pylist():
                        current_format = row["record_format"]
                        current_spec = row["export_spec_id"]
                        if observed_format is None:
                            observed_format = current_format
                            export_spec_id = current_spec
                        if current_format != observed_format or current_spec != export_spec_id:
                            raise PipelineError(
                                "shortlist export dataset mixes formats or specifications",
                                code="HANDOFF_EXPORT_DATASET_MIXED",
                            )
                        source_id = identifiers.get(row["parent_id"])
                        if source_id is not None:
                            identified_count += 1
                        if current_format == "SMILES_TSV":
                            content = row["structure_record"].rstrip("\r\n")
                            if include_identifier:
                                # An empty field rather than a short row: a
                                # ragged TSV misaligns every parser that reads
                                # it, and some parents legitimately come from a
                                # library row whose identifier cell was blank.
                                content += "\t" + _field_value(source_id or "")
                            content += "\n"
                        elif current_format == "SDF_MOLBLOCK":
                            properties = (
                                "\n>  <MOLCASCADE_PARENT_ID>\n"
                                + _property_value(row["parent_id"])
                                + "\n\n>  <MOLCASCADE_PARENT_SMILES>\n"
                                + _property_value(row["parent_smiles"])
                                + "\n\n"
                            )
                            if source_id is not None:
                                # SD properties are per-record and optional, so
                                # a record with no identifier simply omits this
                                # tag instead of carrying an empty one.
                                properties += (
                                    ">  <MOLCASCADE_SOURCE_ID>\n"
                                    + _property_value(source_id)
                                    + "\n\n"
                                )
                            content = (
                                row["structure_record"].rstrip("\r\n")
                                + properties
                                + "$$$$\n"
                            )
                        else:  # Contract validation should make this unreachable.
                            raise PipelineError(
                                "shortlist record format is unsupported",
                                code="HANDOFF_RECORD_FORMAT_UNSUPPORTED",
                                context={"record_format": str(current_format)},
                            )
                        encoded = content.encode("utf-8")
                        stream.write(encoded)
                        digest.update(encoded)
                        row_count += 1
            if row_count == 0 or observed_format is None or export_spec_id is None:
                raise PipelineError(
                    "shortlist export dataset is empty",
                    code="HANDOFF_EXPORT_EMPTY",
                )
            if observed_format != expected_format:
                raise PipelineError(
                    "shortlist manifest format differs from its records",
                    code="HANDOFF_EXPORT_FORMAT_MISMATCH",
                    context={
                        "manifest_format": str(expected_format),
                        "record_format": observed_format,
                    },
                )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            published = atomic_publish_file(temporary, target, overwrite=overwrite)
        except DestinationExistsError as error:
            raise PipelineError(
                f"shortlist output already exists: {target}",
                code="HANDOFF_OUTPUT_EXISTS",
                hint="Choose another path or explicitly enable overwrite.",
                context={"path": str(target)},
            ) from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return MaterializedShortlist(
        path=published,
        record_format=observed_format,
        row_count=row_count,
        sha256=digest.hexdigest(),
        source_artifact_id=state.output_ref.artifact_id,
        export_spec_id=export_spec_id,
        identified_count=identified_count,
    )


__all__ = ["MaterializedShortlist", "materialize_shortlist"]
