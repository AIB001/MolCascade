# Round 7 — Digest-pinned assets and a second synthesizability opinion

Date: 2026-08-24

## Trigger

Round 6 left synthesizability resting on one number. RDKit's SA score is a fragment-frequency
proxy: it asks how unusual a molecule's substructures are against a PubChem frequency table.
That is a real signal and a narrow one. A natural product scores badly on it for reasons that
have nothing to do with whether a chemist can make the compound, and a heavily decorated but
entirely routine amide scores well. Filtering a library on that single axis discards the wrong
molecules in a predictable direction.

SCScore answers a different question — it was trained on twelve million Reaxys reactions under
the single assumption that a product is more complex than its reactants — so it scores reaction
complexity rather than fragment rarity. The two disagree often, and the disagreement is the
useful part.

Adding it required something the package did not have. SCScore is published as weights, not as
code, and MolCascade's offline guarantee says a screening run downloads nothing. Until this
round the architecture document asserted that model and rule assets "must be pinned by
cryptographic digest" without anything implementing that sentence.

## The asset store

Assets are declared in a catalogue that performs no I/O. Each entry names its files, their
SHA-256 digests, their sizes, an upstream URL, the licence, the paper to cite, and whether
loading the payload executes code supplied by someone else.

That last field is deliberately separate from the digest. A digest proves a file is the one the
author published; it does not make deserializing it safe. `PayloadTrust.DATA` means MolCascade's
own reader turns the bytes into numbers — a gzipped JSON array of floats cannot do anything
except be a bad array of floats. `PayloadTrust.EXECUTABLE` means the format runs code chosen by
whoever produced it, which Python pickles and TensorFlow SavedModel graphs both do. Nothing in
the package can conflate the two.

Download and use are separate commands with separate permissions:

| Command | Network | Purpose |
|---|---|---|
| `molcascade assets list` | no | what is declared |
| `molcascade assets status` | no | what is on disk and whether it verifies |
| `molcascade assets where` | no | resolve a reference to a real path |
| `molcascade assets fetch` | yes | the only code path that may download |

A screening run may cross only the first three. A run that needs an absent asset stops and
prints the `fetch` command that would supply it, rather than acquiring it mid-run — results then
depend on files whose digests are recorded, not on what a remote host served that afternoon.

URLs address the GitHub Git-data API rather than `raw.githubusercontent.com`, so a URL names an
exact object id instead of "whatever this branch points at today", and it survives a renamed
branch or a moved tag. Every digest in the catalogue was computed from bytes that were actually
downloaded and cross-checked against the object id the host announced independently; none was
copied from a README.

Configurations refer to assets by reference rather than absolute path:

```
asset:scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz
```

That resolves to the same content on a laptop and on a shared node, which an absolute path does
not. Resolution verifies the digest before handing any path to a backend. A verification stamp
records size, mtime and digest so an unchanged tree is not rehashed on every run, and the stamp
is evidence rather than authority — a file whose size or mtime moved is hashed again.

The asset root is `<repo>/vendor` in a source checkout and the user data directory from an
installed wheel; `MOLCASCADE_ASSET_ROOT` overrides both. `molcascade assets fetch` writes a
`CITATION.md` naming the paper, the licence and every file digest, and regenerates a top-level
`vendor/MANIFEST.md`.

## SCScore

`synthesis.scscore@0.1.0` consumes `parent/v1` and emits `synthesis_score/v1`. The published
model is a feed-forward network over a Morgan fingerprint distributed as a gzipped JSON list of
matrices, so the adapter adds no dependency: the forward pass is numpy, which is already
required, and no TensorFlow install is involved. Nothing in the weight file is executed.

Three things differ deliberately from the reference implementation, each because the reference
fails silently where MolCascade should stop:

- **Fingerprint width is read from the weights, not configured.** The reference takes `FP_len`
  as an argument and will build a 1024-bit fingerprint for a 2048-input model, which multiplies
  into a wrong-but-plausible score. The first weight matrix already states the input width.
- **An unscoreable molecule stops the stage instead of scoring zero.** The reference returns
  `0.0` for an empty fingerprint. Zero is outside the model's 1–5 range, and on a `HIGHER_HARDER`
  axis it reads as "trivially easy to make" — the opposite of "no answer".
- **Counts and bits are named, not inferred from the file name.** The reference switches
  featurization on whether the path contains `uint8`, so a renamed file is scored by the wrong
  featurization with no error at all.

The counts variant folds the *unfolded* sparse fingerprint with `k % bits` and accumulates in
`uint8`. That is not what RDKit returns for a folded count vector of the same width, so it is
reproduced exactly, wraparound past 255 included. Anything else would not be an SCScore.

Shape relations the reference assumes are checked rather than trusted, and the gzip reader
counts both compressed and decompressed bytes against explicit caps — a gzip member declares
nothing about its expanded size.

## Default cascade

The synthesizability tier now carries two arms when the weights are present, in `any` mode: a
molecule is dropped only when two scores built on different evidence agree it is hard. SA score
retains at or below 4.5, SCScore at or below 3.5, both under `HIGHER_HARDER`.

When the weights are absent the tier falls back to the single SA arm in `all` mode, and its note
prints the `molcascade assets fetch scscore` command that would add the second opinion. A
default that stopped on a missing file would be a worse first experience than a default with one
arm.

Neither score predicts whether a route exists. No threshold on either cleanly separates
molecules a retrosynthesis planner can solve from ones it cannot. This is coarse triage, and a
route search belongs below it once the population is small enough to afford one.

## Verification

The suite covers asset resolution, digest mismatch, missing files, stamp staleness, path escape,
symlink and non-regular-file rejection, the trust boundary, catalogue integrity, and manifest
rendering; and for the adapter, file-trust rules, the shape validation the reference omits, both
featurizations, the arithmetic, and the default-cascade fallback.

The forward pass is checked against a transcription of `standalone_model_numpy`'s own loop
rather than against a fixture of expected numbers. A fixture would only prove the code still
does what it did yesterday; comparing against the reference proves it does what SCScore does.
MolCascade evaluates a whole batch at once, so this is a genuine difference in code rather than
a restatement. The counts fold is checked bit-for-bit against RDKit's legacy call, because
getting it subtly wrong would still produce plausible 1–5 scores.

No test depends on the published weights. They were exercised separately, once, against all
three distributed variants: each parses to the published 5×300 architecture, reports the input
width its file name claims, and scores a complexity-ordered probe set inside the 1–5 range —
ethanol at 1.00 through paclitaxel at 4.17 on `1024bool`.

The tier's two shapes are pinned in `tests/cascade/test_defaults.py`, which supplies the answer
about the asset directly rather than reading the weights: one arm in `all` mode with the fetch
command in its note, two arms in `any` mode with the reference and digest pinned. That branch
changes what the default rejects, so leaving it uncovered would have been the wrong economy.

Suite at the close of this round: 644 passed, 2 skipped, up from 639 before the round's tests
were added.

The offline builder was regenerated; its palette had been stale by four adapters
(`synthesis.scscore`, `prediction.custom_model`, `chemistry.medchem_rules`,
`chemistry.medchem_alerts`), and the regenerated payload differs from the previous file only in
those additions.
