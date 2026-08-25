# Round 2: Multi-Port Runtime and Million-Scale Input

Status: Implementation and independent review complete; awaiting the full-repository regression after merge (2026-08-23)

## Goals for This Round

- Allow side ports from earlier stages to be referenced explicitly by any later stage;
- Include the exact stage, port, contract, and lineage in artifact, cache, and resume identities;
- Ingest CSV, XLSX, SDF, Parquet, and very large MOL2 directories into a unified raw contract in bounded batches;
- Fail closed on process crashes, source-file changes, symlinks/hardlinks, and corrupted checkpoints.

## Implemented

1. `ArtifactDatasetRef` is now a first-class entity; artifact manifest v2 records multiple named outputs.
2. The compiler supports explicit `request_port <- earlier_stage.output_port` connections. Without an explicit connection, it automatically selects only the nearest compatible primary port and never guesses a side port.
3. The runner preserves the full `(stage_id, port)` context; cache keys bind every input; resume revalidates bytes, manifests, producers, configurations, lineage, and output ports.
4. Before publication, every Parquet file is checked for the exact schema, globally unique primary keys, row counts, and a complete staging inventory; hardlink and symlink escapes are prohibited.
5. Publication and run locks use OS advisory locks and are released automatically if the lock-holding process crashes; manifests are checked against a 16 MiB limit before publication.
6. Delimited sources constrain both record counts and expanded byte counts; CR, LF, and CRLF are supported, and changes to the source snapshot cause rejection.
7. XLSX uses a read-only, first-visible-worksheet policy; MOL2 directories use an on-disk SQLite inventory, while `raw_molecule/v2` explicitly preserves `raw_format=MOL2` and the original text.
8. The standardizer accepts raw v1 and v2 and strictly invokes the corresponding RDKit parser; MOL2 input is not sent through the MolBlock parser.

## Multi-Agent Cross-Validation

- The runtime reviewer covered hostile exceptions, cross-process locks, side outputs, multiple inputs, cache identity, and resume corruption; no high- or medium-severity blockers remain.
- The source reviewer identified and drove fixes for accidental snapshot deletion, CR-only framing, a single oversized record, oversized Parquet read batches, mixed snapshots between the initial and final inventory, and `BaseException` cleanup.
- The artifact reviewer identified and drove fixes for stale O_EXCL locks left after crashes, post-publication oversized manifests poisoning the artifact, public-store hardlink bypasses, and reuse of cache keys across retries.

## Retained Boundaries

- `batch_size` and `max_buffered_bytes` are logical payload boundaries, not hard limits on operating-system RSS;
- Metadata IOPS may still be a bottleneck for millions of small MOL2 files; after the first import, the raw Parquet artifact should be reused;
- Sources do not use config-only caches across runs because user-path contents may change; a validated checkpoint from the same run can be resumed;
- Plugin versions must still be updated honestly; later rounds will bind backend, code, and asset fingerprints more completely into cache identity.

## Verification

- Independent runtime/compiler/artifact branch reviews: 273 tests passed;
- Extended source/contract/standardizer/registry/integration suite: 96 tests passed;
- The complete suite after merge is run by this round's unified exit gate.
