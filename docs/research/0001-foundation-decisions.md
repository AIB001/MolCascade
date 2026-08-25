# Foundation research decisions (2026-08-22)

## Decisions

1. Use a `src/` Python package and standard `pyproject.toml` metadata.
2. Use Pydantic v2 for strict control-plane schemas and JSON Schema generation.
3. Use Arrow/Parquet for typed, batch-oriented molecule data.
4. Use static registration for built-ins and Python distribution entry points for
   allow-listed third-party plugins.
5. Use an artifact-in/artifact-out public plugin protocol. A batch-map helper may be
   supplied later but is not the universal interface because deduplication, clustering,
   and selection require global operations.
6. Implement a single-writer local runner before adding Dask, Slurm, or Kubernetes.
7. Treat every generator as an external producer. Generator-specific adapters are
   deferred; the current scope begins with generic local SMILES/SDF/Parquet inputs.

## Deferred generator-adapter findings

- Generator-specific paths and status codes must be translated at an anti-corruption
  boundary rather than imported as internal Python objects.
- Generator QC is provenance only; every candidate still passes MolCascade identity and
  hard-gate stages.
- Generator-native scores are model-specific and cannot be compared across models without
  an explicit calibration plugin.

## Configuration hazards

PyYAML's default safe loader still accepts duplicate mapping keys, YAML 1.1-style boolean
tokens, and non-finite floats. MolCascade therefore uses a duplicate-key rejecting loader,
strict Pydantic models, and canonical finite JSON before hashing a revision.

## Primary references

- Python packaging entry points:
  https://packaging.python.org/en/latest/specifications/entry-points/
- Python plugin discovery:
  https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
- Arrow datasets:
  https://arrow.apache.org/docs/python/dataset.html
- Pydantic JSON Schema:
  https://docs.pydantic.dev/latest/concepts/json_schema/
- Atomic replacement:
  https://docs.python.org/3/library/os.html#os.replace
- RDKit RegistrationHash:
  https://www.rdkit.org/docs/source/rdkit.Chem.RegistrationHash.html
