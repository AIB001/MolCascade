# Million-Scale Molecular Input: Ingestion Strategy and Implementation Boundaries

Date: 2026-08-23  
Status: Implementation basis for Round 2

## Conclusions

MolCascade does not confuse “supporting a file extension” with “being suitable for million-scale production.” Inputs fall into two categories:

- **Molecule lists**: CSV/TSV/SMI, XLSX, SDF, and existing MolCascade raw Parquet;
- **Structure-file collections**: a directory containing many MOL2 files.

Every source adapter first acquires an auditable local snapshot identity, then converts the input to `raw_molecule` Parquet in bounded record batches. Downstream stages never reread the user's mutable original files. Each record ID is derived from the source snapshot and the record's position/content, not from the current working directory. Moving an entire MOL2 directory does not change record IDs, whereas changing any file creates a new identity.

The recommended order is: **MolCascade raw Parquet > CSV/TSV/SMI > SDF > XLSX > millions of individual MOL2 files**. The last two are supported but are not equivalent high-throughput formats.

## Format Strategy

| Format | Implementation | Memory/concurrency strategy | Explicit boundaries |
|---|---|---|---|
| CSV/TSV/SMI | stdlib binary framing + CSV parser | Bound both row count and expanded byte count; support CR/LF/CRLF; preserve stable order | Fail fast on a single oversized row; do not rely on first-block type inference |
| XLSX/XLSM | openpyxl with `read_only=True`, `data_only=True` | Lazy row iterator; close the workbook explicitly; never reread Excel after conversion to raw Parquet | At most 1,048,576 rows per worksheet; formulas yield only cached values; `.xls` is unsupported |
| SDF | Byte-level record framing, followed by RDKit parsing | Single-pass streaming; a bad record gets its own decision | A multithreaded supplier may change ordering, so the source does not use one to define stable IDs by default |
| raw Parquet | PyArrow Dataset scanner | Fixed `batch_size`, column projection, and row-group scanning | Accept only an exact MolCascade contract match; do not guess that arbitrary Parquet is a molecule table |
| MOL2 directory | Disk-backed sorted inventory + one-file/one-record reads | Python memory never holds the complete file list or structure bodies; enforce file and record size limits | RDKit has limited support for non-Corina atom typing; parse failures must be recorded explicitly |

Apache Arrow documents that `open_csv()` can read incrementally, batch by batch, but the current incremental reader is single-threaded and freezes types after inference from the first block. MolCascade's initial delimited adapter therefore preserves the source text and explicitly specifies the SMILES and ID columns, preventing type inference from changing identity. A future, benchmarked replacement adapter may use the Arrow reader. [Arrow CSV documentation](https://arrow.apache.org/docs/python/csv.html)

openpyxl describes read-only worksheets as lazy-loading and approximately constant-memory, and requires workbooks to be closed explicitly. MolCascade follows this design, but does not promise a hard process-RSS limit for ZIP decompression, shared strings, or abnormally large cells. Production hard limits should still be enforced by the worker or container. [openpyxl optimized modes](https://openpyxl.readthedocs.io/en/stable/optimized.html)

XLSX is a user-friendly entry point, not the preferred interchange format for two million rows. Microsoft's current worksheet limit is 1,048,576 rows, while `.xls` supports only 65,536 rows. For more than one million rows, use partitioned CSV or Parquet instead of implicitly concatenating multiple worksheets. [Excel specifications](https://support.microsoft.com/en-US/Excel/excel-specifications-and-limits), [guidance for data exceeding the Excel grid](https://support.microsoft.com/en-US/Excel/what-to-do-if-a-data-set-is-too-large-for-the-excel-grid)

## Special Handling for MOL2

MOL2 is not a Mol/SDF block. `raw_molecule/v2` therefore adds `raw_format` and `raw_structure`. The standardizer invokes `MolFromMol2Block` according to the format and never passes MOL2 text to `MolFromMolBlock`. RDKit's documentation notes that its MOL2 parser primarily expects Corina atom typing, has limited support for Tripos dbtranslate typing, and may fail with other typing schemes. MolCascade therefore treats a failure as a source-record-level decision; it must never silently discard the record or fabricate a SMILES string. [RDKit MOL2 API](https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html)

With “millions of small files,” directory enumeration, metadata IOPS, antivirus software, and inode/MFT handling commonly become bottlenecks before chemical parsing does. The implementation therefore:

1. Rejects symlinks and out-of-bounds relative paths;
2. Stores normalized relative paths, sizes, and file states in a disk-backed SQLite/manifest inventory;
3. Sorts and reads files one by one, retaining each body only for the lifetime of the current record;
4. Revalidates the directory inventory before and after enumeration to prevent replacement during scanning from creating a mixed snapshot;
5. Converts the first import immediately into partitioned raw Parquet, so subsequent resume/cache operations never touch the millions of small files again.

If the generator's output format can be controlled, write SDF, SMILES, or Parquet batches instead of one MOL2 file per molecule. When 3D MOL2 must be retained, organize directories by generation batch and freeze the batch manifest alongside them.

## Determinism and Parallelism

RDKit provides `ForwardSDMolSupplier` for single-pass, file-like input as well as multithreaded suppliers, but its documentation also warns that multithreaded output order may differ from input order. MolCascade's source identity uses single-pass, order-stable framing by default. Expensive standardization may run in parallel in downstream workers, with outputs merged by stable ID. [RDKit molecule suppliers](https://rdkit.org/docs/GettingStartedInPython.html)

A Parquet scanner's `batch_size` is only a row-count boundary, not a strict byte or RSS limit. Arrow also states that read-ahead values are estimates and that a format may read a larger physical batch. MolCascade therefore also enforces a record-byte budget and retains separate worker RSS and temporary-disk quotas in production planning. [Arrow Parquet memory guidance](https://arrow.apache.org/docs/cpp/parquet.html)

## Production Recommendations

- Begin with a 10k pilot and record records/s, peak RSS, temporary-disk use, and the distribution of failure reasons;
- Then run a 500k load test to confirm the inventory and Parquet partitioning strategy;
- Before the full 2M run, reserve temporary disk for raw, parent, and side ports rather than estimating directly from input-file size;
- For CSV, specify encoding, delimiter, header, and SMILES column; for XLSX, specify the sheet and header row;
- For MOL2, specify the atom-typing source, maximum bytes per record, and policy for tolerating invalid structures;
- Record the source snapshot, plugin version, configuration, identity policy, and RDKit version in provenance.
