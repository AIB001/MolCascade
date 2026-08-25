# Large MOL2 ingestion: loose files, manifests, and packing

Date: 2026-08-23
Status: Follow-up to [0004](0004-large-input-ingestion.md), which chose to accept a
directory of MOL2 files as one input. This record covers what that costs at
multi-million-file scale and how a library would be packed to avoid it.

## Implemented loose-file path

`source.mol2_directory@0.1.0` treats a directory tree as an immutable input
snapshot. It enumerates without following symlinks, stores the manifest in
SQLite rather than a Python list, and streams files in binary UTF-8 path order.
Each source ID binds the file SHA-256 and its sorted occurrence index; the
absolute directory path is excluded, so moving an unchanged tree preserves IDs.

This design bounds Python row buffering and works for existing loose-file
collections. It cannot remove the filesystem cost of millions of `stat`, open,
read, and close operations. On NTFS, network filesystems, and object-backed
mounts, metadata operations can dominate structure parsing.

## Recommended packing for multi-million-file libraries

For repeated screening, pack immutable files once and retain a sidecar manifest:

- Use uncompressed tar shards when fast sequential streaming and exact byte
  offsets matter. A practical shard target is 10,000–100,000 molecules or a few
  GiB, chosen to keep retry scope manageable.
- Store one manifest row per member: shard ID/checksum, normalized POSIX member
  path, byte offset, byte length, member SHA-256, and original size. Use Parquet
  for large manifests and deterministic row ordering.
- Reject absolute paths, `..`, duplicate/case-colliding member names, symlinks,
  hard links, devices, sparse-file surprises, and members whose declared or
  observed size exceeds policy.
- Avoid a single gzip-compressed tar when random access or partial retry is
  required. Uncompressed tar is simplest; independently compressed shards or a
  reviewed seekable-zstd framing may be used when the manifest also binds the
  exact codec and version.
- Hash both every member and every shard. A shard checksum alone detects damage
  but does not provide stable per-molecule identity or efficient verification.
- Build archives in a controlled process with fixed member order, normalized
  metadata, and no executable extraction step. Consumers should stream members
  directly and never unpack untrusted paths onto the filesystem.

Packing is an ingestion optimization, not a chemistry conversion. Original
MOL2 text remains the `raw_structure`; RDKit parsing and registration happen in
the standardization stage, with parser/version provenance and structured
per-record rejection.
