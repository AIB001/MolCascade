# Round 3: Offline Drag-and-Drop Configuration and the CLI User Journey

Status: In progress (2026-08-23)

## User Entry Points

Compatibility entry points:

```bash
molcascade --generate config --output molcascade-config.html
molcascade --vs --config pipeline.json --molecules molecules.csv
```

Canonical entry points:

```bash
molcascade generate config --output molcascade-config.html
molcascade validate pipeline.json
molcascade run pipeline.json --molecules molecules.csv --workspace .molcascade
molcascade status <run-id> --events
molcascade doctor
```

## Current Implementation

- A single-file HTML document with no CDN or network requests, and a CSP hash that pins the embedded CSS and JavaScript;
- A hierarchical palette on the left, a pipeline in the center that supports drag-and-drop and button-based reordering, and a plugin/parameter inspector on the right;
- A default workflow, plugin availability/license/status information, a JSON preview, Blob downloads, and optional saving through the File System Access API;
- The browser performs only user-friendly preflight checks; the Pydantic configuration and compiler are the authoritative validators;
- `argparse` supports both compatibility flags and canonical subcommands without depending on Typer or Rich;
- `doctor` only probes capabilities; it does not install, download, or implicitly trust third-party backends;
- CLI source-path overrides and run status/audit information use structured JSON output;
- A new offline run report embeds only aggregate counts and provenance, never molecular rows, and can download a configuration for the next round.

## Cross-Checks for This Round

1. The HTML payload is safe against `</textarea>`, script/SVG/URL, and U+2028/U+2029 injection;
2. In addition to drag-and-drop, the UI retains click-to-add, move-up/move-down, and keyboard selection controls;
3. Every default or replacement workflow exported by the browser can be read back by the compiler;
4. The auto-routing logic for `--molecules` must not pass CSV-specific fields into XLSX, SDF, or MOL2 plugins;
5. `--json` remains machine-readable for parse, schema, and runtime failures;
6. Windows reserved device names, trailing dots, symlinked workspaces, and path overrides must be rejected before anything is written;
7. Operation is entirely local, with no docking, MD, or MMPBSA commands and no PRISM import.

## Exit Gate

- Targeted CLI/HTML tests and the full repository test suite are all green;
- Generate the HTML in a real environment, export JSON, and then run `molcascade validate`;
- Auto-routing for CSV, XLSX, SDF, Parquet, and MOL2 directories is covered by tests;
- The README provides a five-minute quick start and recommendations for large-input formats;
- The automatic run-report generation policy and its disable option are explicit.
