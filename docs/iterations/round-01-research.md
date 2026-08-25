# Round 1: Screening Layers, Backends, and Interaction Boundaries

Status: Complete (2026-08-23)

## Inputs for This Round

- Read the original DOCX in full, including stages 0–7, data contracts, budgets, exceptions, and milestones.
- Two independent agents researched stages 0–4 and stages 5–7/HTML, respectively.
- The main thread reviewed the current configuration, contracts, plugins, artifacts, and chemistry implementation.

## Cross-Validation Findings

1. Multiple backends must share contracts, but identity equivalence across toolkits must not be assumed.
2. PAINS and general medicinal-chemistry alerts are warnings by default; only project-approved rules may cause hard rejection.
3. Clustering must avoid `N×N`; Murcko hashing, leader/sphere methods, and embedding-based clustering have different semantics.
4. Browser-exported configurations must return to the backend for strict validation; HTML has no authority to modify artifacts.
5. The existing artifact model has only one `kind`, yet plugins may produce multiple contract ports. This is a blocking defect for the runner and was scheduled for correction in Round 2.
6. PRISM is currently out of scope, and the associated experimental importer has been removed.

## Round 2 Entry Criteria

- Introduce typed artifact port references;
- Define explicit replacement boundaries for at least source, standardize, and gate;
- Have the runner validate contracts port by port before commit;
- Ensure `doctor` distinguishes `available`, `unavailable`, and `degraded` without automatically accessing the network or installing anything.
