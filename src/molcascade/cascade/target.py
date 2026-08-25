"""Turn one receptor and one site definition into what each engine reads.

Three engines want three different things about the same protein.  Uni-Dock
reads a PDBQT and a box; GNINA reads the PDB and the same box; KarmaDock reads
the PDB and a ligand whose atoms locate the site.  Asking the operator for all
of that would be asking them to keep three descriptions of one pocket in
agreement, and the first time they drifted apart the tier would quietly stop
being a consensus.

So the operator supplies a receptor and *one* site definition, and this module
expands it: a reference ligand or a pocket PDB becomes an explicit box, and the
box is what reaches the stage config -- so a re-run needs the numbers, not the
ligand file.  The receptor's PDBQT is derived with meeko once, cached by the
receptor's digest, and can be replaced wholesale when meeko cannot cope.

What is *not* derived is as deliberate as what is.  Protonation states,
tautomers, cofactors, metals and missing loops decide the science.  The receptor
is read, hashed and validated here; it is never edited.  ``meeko`` has a flag
that deletes residues with missing atoms and this module does not pass it: a
silently truncated protein is exactly the kind of change that never shows up in
a score.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from molcascade.backends.preflight import TARGET_SETTING_FLAGS
from molcascade.cascade.models import BoxConfig, CascadeConfig, TargetConfig
from molcascade.errors import ConfigError
from molcascade.plugins.registry import PluginRegistry

#: Added to each side of the reference ligand's extent, the convention AutoDock
#: Vina's ``--autobox_ligand`` uses and the one every downstream engine's
#: documentation is written against.  Enough room for a ligand a little larger
#: than the reference without turning a pocket study into blind docking.
BOX_PADDING_ANGSTROM = 4.0

#: The same ceiling :class:`BoxConfig` enforces, checked here so an oversized
#: pocket file produces a sentence about the pocket file rather than a Pydantic
#: validation report about a field the operator never typed.
_MAX_BOX_EDGE_ANGSTROM = 200.0

#: A structure file, not a trajectory.  Matches the docking adapters' own limit.
_MAX_STRUCTURE_BYTES = 256 * 1024 * 1024

#: Receptor preparation is a one-off on a single structure.  Ten minutes is
#: generous for meeko and short enough that a hang is reported rather than slept
#: through.
_PREPARE_TIMEOUT_SECONDS = 600.0

_LIGAND_SUFFIXES = frozenset({".sdf", ".mol", ".mol2", ".pdb"})


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """A target with every derivation already performed and recorded.

    ``notes`` is the part that matters for reproducibility: it says which
    numbers were computed rather than typed, and from what.  A run whose box
    came from a ligand file and one whose box was typed are the same experiment
    only if the numbers match, and this is where a reader can check.
    """

    target: TargetConfig
    notes: tuple[str, ...] = ()


def parse_box(text: str) -> BoxConfig:
    """Read ``--box cx,cy,cz,sx,sy,sz``.

    Six numbers, no defaults.  Accepting three and inventing an edge length
    would be guessing at the size of someone else's binding site.
    """

    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 6 or any(not piece for piece in parts):
        raise ConfigError(
            "--box takes exactly six comma-separated numbers",
            code="DOCKING_BOX_MALFORMED",
            hint="--box cx,cy,cz,sx,sy,sz -- centre first, then the edge lengths in angstrom.",
            context={"value": text},
        )
    try:
        numbers = [float(piece) for piece in parts]
    except ValueError as error:
        raise ConfigError(
            "--box contains something that is not a number",
            code="DOCKING_BOX_MALFORMED",
            hint="--box cx,cy,cz,sx,sy,sz -- centre first, then the edge lengths in angstrom.",
            context={"value": text},
        ) from error
    return _box(numbers, source="--box")


def _box(numbers: Sequence[float], *, source: str) -> BoxConfig:
    try:
        return BoxConfig(
            center_x=numbers[0],
            center_y=numbers[1],
            center_z=numbers[2],
            size_x=numbers[3],
            size_y=numbers[4],
            size_z=numbers[5],
        )
    except ValueError as error:
        raise ConfigError(
            f"the search box from {source} is not usable",
            code="DOCKING_BOX_INVALID",
            hint=(
                "Edge lengths must be positive and no larger than "
                f"{_MAX_BOX_EDGE_ANGSTROM:g} A. A box that big is a blind-docking "
                "run rather than a pocket."
            ),
            context={"source": source, "detail": str(error)},
        ) from error


def merge_target(
    base: TargetConfig | None,
    *,
    receptor: Path | None = None,
    reference_ligand: Path | None = None,
    pocket: Path | None = None,
    box: str | None = None,
    receptor_pdbqt: Path | None = None,
) -> TargetConfig | None:
    """Lay the run-time flags over whatever target the cascade already carried.

    A site flag *replaces* the cascade's site rather than joining it.  Passing
    ``--box`` to a cascade written against a reference ligand is an operator
    saying "this pocket, these numbers"; treating it as a second definition and
    refusing would make the override useless exactly when it is wanted.
    """

    site_flags = [
        ("--reference-ligand", reference_ligand),
        ("--pocket", pocket),
        ("--box", box),
    ]
    given = [name for name, value in site_flags if value is not None]
    if len(given) > 1:
        raise ConfigError(
            f"the binding site was named more than once ({', '.join(given)})",
            code="DOCKING_TARGET_SITE_AMBIGUOUS",
            hint=(
                "Two site definitions that disagree have no correct resolution, so "
                "exactly one is accepted. Pick the one you trust."
            ),
            context={"flags": given},
        )
    nothing_given = (
        receptor is None and receptor_pdbqt is None and not given
    )
    if base is None and nothing_given:
        return None

    receptor_path = str(receptor.expanduser().resolve()) if receptor is not None else None
    if receptor_path is None and base is not None:
        receptor_path = base.receptor_path
    if receptor_path is None:
        raise ConfigError(
            "a binding site was given but no receptor to find it in",
            code="DOCKING_TARGET_RECEPTOR_REQUIRED",
            hint="Pass '--receptor RECEPTOR.pdb' as well.",
            context={"flags": given},
        )

    fields: dict[str, object] = {
        "name": base.name if base is not None else Path(receptor_path).stem,
        "receptor_path": receptor_path,
        "receptor_sha256": base.receptor_sha256 if base is not None else None,
        "receptor_pdbqt_path": (
            str(receptor_pdbqt.expanduser().resolve())
            if receptor_pdbqt is not None
            else (base.receptor_pdbqt_path if base is not None else None)
        ),
        "pocket_pdb_path": base.pocket_pdb_path if base is not None else None,
    }
    if receptor is not None and base is not None and receptor_path != base.receptor_path:
        # A different structure is a different target, and a digest pinned
        # against the old bytes would stop the run with a mismatch the operator
        # already resolved by naming a new file.
        fields["receptor_sha256"] = None

    if given:
        fields["box"] = parse_box(box) if box is not None else None
        fields["reference_ligand_path"] = (
            str(reference_ligand.expanduser().resolve()) if reference_ligand is not None else None
        )
        fields["pocket_path"] = str(pocket.expanduser().resolve()) if pocket is not None else None
    elif base is not None:
        fields["box"] = base.box
        fields["reference_ligand_path"] = base.reference_ligand_path
        fields["pocket_path"] = base.pocket_path
    else:
        raise ConfigError(
            "a receptor was given but no binding site in it",
            code="DOCKING_TARGET_SITE_REQUIRED",
            hint=(
                "Name the site exactly once: '--reference-ligand LIGAND.sdf', "
                "'--pocket POCKET.pdb', or '--box cx,cy,cz,sx,sy,sz'."
            ),
            context={"receptor_path": receptor_path},
        )
    return TargetConfig.model_validate(fields)


def docking_requirements(
    cascade: CascadeConfig,
    *,
    registry: PluginRegistry,
) -> frozenset[str]:
    """Which target settings this cascade's engines actually require.

    Derivation is not free -- preparing a receptor with meeko takes seconds and
    needs meeko installed -- so a cascade with only KarmaDock in it must not be
    made to produce a PDBQT nothing will read, and a cascade with no docking at
    all must not be made to produce anything.
    """

    needed: set[str] = set()
    for backend in _cascade_backends(cascade):
        try:
            plugin = registry.entry(backend).plugin
        except Exception:
            # An unknown plugin is the compiler's error to raise, with a better
            # message than anything this function could invent.
            continue
        fields = getattr(getattr(plugin, "config_model", None), "model_fields", None)
        if not fields or "receptor_path" not in fields:
            continue
        needed.update(
            name
            for name, field_info in fields.items()
            if name in TARGET_SETTING_FLAGS and field_info.is_required()
        )
    return frozenset(needed)


def _cascade_backends(cascade: CascadeConfig) -> Iterator[str]:
    yield cascade.ingest.backend
    if cascade.standardize is not None and cascade.standardize.enabled:
        yield cascade.standardize.backend
    for tier in cascade.tiers:
        if not tier.enabled:
            continue
        for criterion in tier.active_criteria:
            yield criterion.backend
            if criterion.gate is not None:
                yield criterion.gate.backend
    for step in cascade.finalize.steps:
        if step.enabled:
            yield step.backend


def resolve_target(
    target: TargetConfig | None,
    *,
    requirements: frozenset[str],
    workspace: Path,
) -> ResolvedTarget | None:
    """Perform every derivation the cascade's engines need, once, up front.

    Once, because deriving per shard would mean discovering a receptor
    preparation failure one worker at a time; up front, because a run that
    reaches the docking tier before finding out that meeko cannot read the
    structure has already spent the whole funnel.
    """

    if target is None:
        return None
    notes: list[str] = []
    fields: dict[str, object] = {
        "name": target.name,
        "receptor_path": target.receptor_path,
        "receptor_sha256": target.receptor_sha256,
        "box": target.box,
        "reference_ligand_path": target.reference_ligand_path,
        "pocket_path": target.pocket_path,
        "receptor_pdbqt_path": target.receptor_pdbqt_path,
        "pocket_pdb_path": target.pocket_pdb_path,
    }

    site_file = target.reference_ligand_path or target.pocket_path
    if "center_x" in requirements and target.box is None:
        assert site_file is not None  # TargetConfig guarantees a site in some form
        derived = box_from_structure(Path(site_file))
        fields["box"] = derived
        notes.append(
            f"box derived from {Path(site_file).name}: centre "
            f"({derived.center_x:.3f}, {derived.center_y:.3f}, {derived.center_z:.3f}) "
            f"size ({derived.size_x:.3f}, {derived.size_y:.3f}, {derived.size_z:.3f}) "
            f"with {BOX_PADDING_ANGSTROM:g} A padding"
        )
    elif target.box is not None and site_file is not None:
        # Both were written by hand.  The engines that read six numbers and the
        # one that reads a ligand would otherwise search different pockets and
        # the tier would report their disagreement as a consensus.
        _require_site_agreement(target.box, Path(site_file))
        notes.append(f"box checked against {Path(site_file).name}: the site is inside it")

    if "receptor_pdbqt_path" in requirements and target.receptor_pdbqt_path is None:
        prepared, note = prepare_receptor_pdbqt(Path(target.receptor_path), workspace=workspace)
        fields["receptor_pdbqt_path"] = str(prepared)
        notes.append(note)

    return ResolvedTarget(target=TargetConfig.model_validate(fields), notes=tuple(notes))


def _require_site_agreement(box: BoxConfig, site: Path) -> None:
    """Refuse a hand-written box that does not contain the hand-written site.

    Containment rather than equality on purpose: shrinking the box around a
    reference ligand to speed a search up, or widening it to allow a larger
    scaffold, are both ordinary and both leave the engines looking at one
    pocket.  A centre outside the box is the case that does not -- KarmaDock
    would dock around the ligand while Uni-Dock searched somewhere else, and
    the tier would call the resulting disagreement a consensus.
    """

    derived = box_from_structure(site)
    centres = (
        ("x", derived.center_x, box.center_x, box.size_x),
        ("y", derived.center_y, box.center_y, box.size_y),
        ("z", derived.center_z, box.center_z, box.size_z),
    )
    outside = [axis for axis, site_c, box_c, size in centres if abs(site_c - box_c) > size / 2.0]
    if outside:
        raise ConfigError(
            f"the search box does not contain the site in {site.name}",
            code="DOCKING_SITE_DISAGREEMENT",
            hint=(
                "The box and the site file are both used -- the box by Uni-Dock and "
                "GNINA, the file by KarmaDock -- so a box the file's atoms sit "
                "outside of means the engines are docking into different places. "
                "Drop the box and let it be derived, or point the file at the same "
                "pocket."
            ),
            context={
                "site_path": str(site),
                "axes_outside": outside,
                "site_center": [derived.center_x, derived.center_y, derived.center_z],
                "box_center": [box.center_x, box.center_y, box.center_z],
                "box_size": [box.size_x, box.size_y, box.size_z],
            },
        )


def box_from_structure(path: Path) -> BoxConfig:
    """The smallest box containing a structure's atoms, plus padding on each side.

    Deliberately geometric and nothing more.  A reference ligand is being used
    here only for where its atoms are; nothing is read about its chemistry, so a
    file this project could not parse as a molecule still works as a ruler.
    """

    coordinates = list(_coordinates(path))
    if not coordinates:
        raise ConfigError(
            f"no atom coordinates were found in {path.name}",
            code="DOCKING_SITE_STRUCTURE_EMPTY",
            hint=(
                "The binding site is located from this file's atoms. Accepted "
                "formats are .sdf, .mol, .mol2 and .pdb; an empty or unrecognised "
                "one gives no site rather than a wrong one."
            ),
            context={"path": str(path)},
        )
    lows = [min(axis) for axis in zip(*coordinates, strict=True)]
    highs = [max(axis) for axis in zip(*coordinates, strict=True)]
    centre = [(low + high) / 2.0 for low, high in zip(lows, highs, strict=True)]
    sizes = [
        (high - low) + 2.0 * BOX_PADDING_ANGSTROM
        for low, high in zip(lows, highs, strict=True)
    ]
    if any(size > _MAX_BOX_EDGE_ANGSTROM for size in sizes):
        raise ConfigError(
            f"the atoms in {path.name} span more than a binding site",
            code="DOCKING_SITE_TOO_LARGE",
            hint=(
                "This looks like a whole structure rather than a pocket or a bound "
                "ligand. Pass the ligand actually in the site, or give the six "
                "numbers with '--box'."
            ),
            context={
                "path": str(path),
                "size_x": sizes[0],
                "size_y": sizes[1],
                "size_z": sizes[2],
            },
        )
    return _box([*centre, *sizes], source=path.name)


def _coordinates(path: Path) -> Iterable[tuple[float, float, float]]:
    suffix = path.suffix.lower()
    if suffix not in _LIGAND_SUFFIXES:
        raise ConfigError(
            f"{path.name} is not a structure format the site can be read from",
            code="DOCKING_SITE_FORMAT_UNSUPPORTED",
            hint="Accepted: .sdf, .mol, .mol2 and .pdb.",
            context={"path": str(path), "suffix": suffix},
        )
    text = _read_text(path)
    if suffix == ".pdb":
        return _pdb_coordinates(text)
    if suffix == ".mol2":
        return _mol2_coordinates(text, path=path)
    return _molfile_coordinates(text, path=path)


def _read_text(path: Path) -> str:
    try:
        info = path.stat()
    except OSError as error:
        raise ConfigError(
            f"the binding-site file could not be read: {path}",
            code="DOCKING_SITE_UNREADABLE",
            hint="The path is read at start-up so a run cannot discover it three tiers deep.",
            context={"path": str(path), "error_type": type(error).__name__},
        ) from error
    if info.st_size > _MAX_STRUCTURE_BYTES:
        raise ConfigError(
            f"the binding-site file is larger than a structure file: {path}",
            code="DOCKING_SITE_UNREADABLE",
            hint="A reference ligand or a pocket, not a trajectory.",
            context={"path": str(path), "size_bytes": info.st_size},
        )
    return path.read_text(encoding="utf-8", errors="replace")


def _pdb_coordinates(text: str) -> Iterable[tuple[float, float, float]]:
    """Read the fixed columns, not whitespace-separated fields.

    PDB is a column format and its coordinate fields run together as soon as one
    of them needs four digits before the decimal point, which happens in any
    structure placed far from the origin.  Splitting on whitespace reads those
    as one number and silently loses atoms.
    """

    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            yield (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue


def _mol2_coordinates(text: str, *, path: Path) -> Iterable[tuple[float, float, float]]:
    lines = text.splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "@<TRIPOS>ATOM"
        )
    except StopIteration:
        raise ConfigError(
            f"{path.name} has no @<TRIPOS>ATOM record",
            code="DOCKING_SITE_FORMAT_UNSUPPORTED",
            hint="A mol2 file without an atom block carries no coordinates to locate a site with.",
            context={"path": str(path)},
        ) from None
    found: list[tuple[float, float, float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("@<TRIPOS>"):
            break
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            found.append((float(fields[2]), float(fields[3]), float(fields[4])))
        except ValueError:
            continue
    return found


def _molfile_coordinates(text: str, *, path: Path) -> Iterable[tuple[float, float, float]]:
    """Read a V2000 connection table's atom block.

    V3000 is refused rather than half-read.  Its atom block is a different
    format entirely, and a parser that skipped the lines it did not recognise
    would report an empty structure as a site of size zero.
    """

    lines = text.splitlines()
    if len(lines) < 4:
        raise ConfigError(
            f"{path.name} is too short to be a molfile",
            code="DOCKING_SITE_FORMAT_UNSUPPORTED",
            hint="Accepted: .sdf, .mol, .mol2 and .pdb.",
            context={"path": str(path)},
        )
    counts = lines[3]
    if "V3000" in counts.upper():
        raise ConfigError(
            f"{path.name} is a V3000 molfile, which this reader does not parse",
            code="DOCKING_SITE_FORMAT_UNSUPPORTED",
            hint=(
                "Convert it to V2000 or mol2, or give the six numbers with '--box'. "
                "Half-reading it would report a site of size zero."
            ),
            context={"path": str(path)},
        )
    try:
        atom_count = int(counts[0:3])
    except ValueError:
        raise ConfigError(
            f"{path.name} has no readable counts line",
            code="DOCKING_SITE_FORMAT_UNSUPPORTED",
            hint="Accepted: .sdf, .mol, .mol2 and .pdb.",
            context={"path": str(path)},
        ) from None
    found: list[tuple[float, float, float]] = []
    for line in lines[4 : 4 + max(atom_count, 0)]:
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            found.append((float(fields[0]), float(fields[1]), float(fields[2])))
        except ValueError:
            continue
    return found


def prepare_receptor_pdbqt(receptor: Path, *, workspace: Path) -> tuple[Path, str]:
    """Derive Uni-Dock's PDBQT from the operator's PDB, cached by its digest.

    Cached by digest rather than by path so that re-running the same cascade
    does not re-prepare, and so that editing the receptor cannot leave a stale
    PDBQT behind wearing the same name.

    This is the fragile step in the whole tier -- protonation, non-standard
    residues and missing atoms are all places meeko stops -- so its failure is
    reported with meeko's own words and the documented way past it.  Meeko's
    ``--allow_bad_res``, which deletes incomplete residues, is deliberately not
    passed: quietly docking against a protein with holes in it is worse than
    stopping.
    """

    from molcascade.plugins.builtin.docking.common import structure_digest

    _, digest = structure_digest(
        str(receptor),
        code="DOCKING_RECEPTOR_UNREADABLE",
        hint=(
            "This is the receptor for the docking tier, read once here so a run "
            "cannot discover it is unreadable three tiers deep."
        ),
    )
    destination = workspace / "targets" / digest / "receptor.pdbqt"
    if destination.exists():
        return destination, f"receptor PDBQT reused from the workspace cache for sha256:{digest}"

    command = _meeko_command()
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [*command, "--read_pdb", str(receptor), "-p", str(destination)],
        capture_output=True,
        timeout=_PREPARE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout).decode("utf-8", "replace").strip()
        raise ConfigError(
            f"meeko could not prepare {receptor.name} as a PDBQT receptor",
            code="DOCKING_RECEPTOR_PREPARATION_FAILED",
            hint=(
                "Uni-Dock reads PDBQT and this is where a PDB becomes one. "
                "Protonation, non-standard residues and missing atoms are where it "
                "stops. Prepare the receptor yourself and pass "
                "'--receptor-pdbqt PREPARED.pdbqt', or fix the structure -- "
                "MolCascade will not delete residues to make it parse."
            ),
            context={
                "receptor_path": str(receptor),
                "returncode": completed.returncode,
                "meeko_output": detail[-4000:],
            },
        )
    return destination, f"receptor PDBQT derived with meeko from sha256:{digest}"


def _meeko_command() -> list[str]:
    """How to invoke meeko's receptor preparation on this installation.

    The console script when it is on ``PATH``, and the module otherwise -- a
    conda environment activated only by ``PYTHONPATH`` has the package without
    the script, and failing there would look like meeko being absent.
    """

    script = shutil.which("mk_prepare_receptor.py")
    if script is not None:
        return [script]
    if _meeko_importable():
        return [sys.executable, "-m", "meeko.cli.mk_prepare_receptor"]
    raise ConfigError(
        "meeko is not installed, so a receptor PDBQT cannot be derived",
        code="DOCKING_RECEPTOR_PREPARATION_UNAVAILABLE",
        hint=(
            'pip install "molcascade[docking]" -- meeko is what turns a PDB into '
            "the PDBQT Uni-Dock reads. Alternatively prepare it yourself and pass "
            "'--receptor-pdbqt PREPARED.pdbqt'."
        ),
    )


def _meeko_importable() -> bool:
    from importlib.util import find_spec

    try:
        return find_spec("meeko") is not None
    except (ImportError, ValueError):
        return False


__all__ = [
    "BOX_PADDING_ANGSTROM",
    "ResolvedTarget",
    "box_from_structure",
    "docking_requirements",
    "merge_target",
    "parse_box",
    "prepare_receptor_pdbqt",
    "resolve_target",
]
