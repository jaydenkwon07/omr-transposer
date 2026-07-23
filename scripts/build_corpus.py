"""Materialize a named corpus from music21's core corpus into ``data/corpora/<name>/``.

Why this exists: ``corpus_id`` is one third of the ``(corpus_id, seed, config_hash)``
reproducibility triple, but a corpus directory assembled by hand records nothing about where
its files came from. This script makes the corpus itself reproducible — it writes a
``CORPUS.json`` alongside the scores recording the music21 version, the selection recipe, and
a sha256 per file, so "did the model get worse, or the data?" stays answerable.

The corpus is chosen for *unit* diversity, not file count. ``datagen.corpus`` decomposes each
score into units: a plain ``Part`` becomes a single-staff unit, a contiguous ``PartStaff`` run
becomes a grand-staff unit. So open-score vocal/quartet writing is the single-staff source and
closed-score keyboard writing is the grand-staff source.

Usage:
    uv run python scripts/build_corpus.py --name mixed_v1
    uv run python scripts/build_corpus.py --name mixed_v1 --limit 5   # quick smoke build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import converter, corpus, stream

# Collections to draw from, with a per-collection cap. Caps keep any one composer from
# dominating: Bach alone is 410 of the corpus's ~654 MusicXML files, which would reproduce
# exactly the monoculture this corpus exists to avoid.
#
# `single` / `grand` is what the collection actually yields under _unit_groups, measured;
# it is documentation, not a filter.
RECIPE: tuple[tuple[str, int], ...] = (
    ("beethoven", 22),        # single-staff, widest key-signature spread (9)
    ("mozart", 16),           # single-staff + a little grand
    ("haydn", 9),             # single-staff
    ("trecento", 40),         # single-staff, widest meter spread (12 time signatures)
    ("monteverdi", 30),       # single-staff, early-music notation
    ("schumann_robert", 7),   # mixed
    ("schumann_clara", 5),    # grand-staff heavy
    ("joplin", 1),            # grand-staff, ragtime
    ("scarlatti", 4),         # grand-staff keyboard, if present
    ("cpebach", 1),           # grand-staff
    ("handel", 1),
    ("schubert", 1),
    ("verdi", 1),
    ("weber", 1),
    ("beach", 1),
    ("liliuokalani", 1),
    ("johnson_j_r", 1),
    ("joplin", 1),
    ("corelli", 1),
    ("ciconia", 1),
    ("luca", 1),
    ("lusitano", 1),
    ("webern", 1),
    ("schoenberg", 2),
)

_MUSICXML = (".mxl", ".xml", ".musicxml")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Entry:
    path: str        # filename written into the corpus dir
    source: str      # path within the music21 core corpus
    collection: str
    sha256: str


def _collection_of(path: Path) -> str:
    parts = path.parts
    i = parts.index("corpus") if "corpus" in parts else 0
    return parts[i + 1] if i + 1 < len(parts) else "?"


def _safe_name(collection: str, path: Path) -> str:
    """A filename unique across the whole corpus.

    The collection alone is not enough: works share movement filenames (``movement1.mxl``
    appears under many opus directories), so keying on the stem silently overwrites earlier
    files and leaves CORPUS.json describing scores that are no longer on disk. Keep every
    path segment below the collection.
    """
    return f"{collection}__{_logical_id(path)}.musicxml"


def _logical_id(path: Path) -> str:
    """The work's identity within its collection, independent of source format.

    Strips only the true final suffix (``.replace('.xml', '')`` would also corrupt a name
    that merely contains the substring), so ``foo.xml`` and ``foo.mxl`` — the same piece
    shipped twice — collapse to one id and can be deduplicated.
    """
    parts = path.parts
    i = parts.index("corpus") if "corpus" in parts else 0
    tail = list(parts[i + 2 :]) if i + 2 <= len(parts) else [path.name]
    if tail:
        tail[-1] = Path(tail[-1]).stem
    return _UNSAFE.sub("_", "_".join(tail)).strip("_")


def _fold_in(
    src_dir: Path,
    out_dir: Path,
    written: dict[str, str],
    taken: Counter[str],
    limit: int | None,
) -> list[Entry]:
    """Copy an existing local corpus directory in, recording provenance for it.

    music21's core MusicXML is overwhelmingly open-score (part-per-staff), so it is a poor
    grand-staff source — the measured yield is ~18 grand-staff units against ~372
    single-staff. Closed-score material has to come from somewhere else. Folding it in here,
    rather than pointing the generator at a second directory, keeps one corpus root and one
    CORPUS.json describing everything in it.
    """
    label = src_dir.name
    files = sorted(
        p for p in src_dir.rglob("*") if p.suffix.lower() in _MUSICXML and p.is_file()
    )
    if limit is not None:
        files = files[:limit]
    out: list[Entry] = []
    for path in files:
        name = f"{label}__{_UNSAFE.sub('_', path.stem)}.musicxml"
        if name in written:
            raise RuntimeError(f"filename collision {name!r} ({path} vs {written[name]})")
        written[name] = path.as_posix()
        blob = path.read_bytes()
        (out_dir / name).write_bytes(blob)
        out.append(
            Entry(
                path=name,
                source=path.as_posix(),
                collection=label,
                sha256=hashlib.sha256(blob).hexdigest(),
            )
        )
        taken[label] += 1
    return out


def build(
    out_dir: Path,
    limit_per_collection: int | None,
    include_dirs: tuple[Path, ...] = (),
    include_limit: int | None = None,
) -> int:
    caps: dict[str, int] = {}
    for name, cap in RECIPE:
        caps[name] = max(caps.get(name, 0), cap)
    if limit_per_collection is not None:
        caps = {k: min(v, limit_per_collection) for k, v in caps.items()}

    # Sorted for a deterministic selection: the same recipe builds the same corpus.
    # Sort key puts .mxl before .xml for the same work so dedup keeps the compressed
    # canonical copy; the path itself breaks remaining ties.
    candidates = sorted(
        (p for p in corpus.getCorePaths() if p.suffix in _MUSICXML),
        key=lambda p: (_collection_of(p), _logical_id(p), _MUSICXML.index(p.suffix)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    taken: Counter[str] = Counter()
    entries: list[Entry] = []
    skipped: Counter[str] = Counter()
    written: dict[str, str] = {}

    for path in candidates:
        coll = _collection_of(path)
        allowed = caps.get(coll)
        if allowed is None or taken[coll] >= allowed:
            continue
        # The same work ships in more than one format (foo.mxl and foo.xml). Taking both
        # would put an identical score in the corpus twice, skewing unit sampling toward it.
        name = _safe_name(coll, path)
        if name in written:
            skipped[f"{coll}:duplicate-format"] += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = converter.parse(path)
        except Exception as exc:  # noqa: BLE001 - corpus files are third-party
            skipped[f"{coll}:parse"] += 1
            print(f"  skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not isinstance(score, stream.Score) or not score.parts:
            skipped[f"{coll}:not-a-score"] += 1
            continue

        # Write MusicXML, not a copy of the source file: the corpus mixes .mxl (zipped) and
        # .xml, and the loader should see one uniform, text-diffable format.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xml = music21.musicxml.m21ToXml.GeneralObjectExporter().parse(score)
        except Exception as exc:  # noqa: BLE001
            skipped[f"{coll}:export"] += 1
            print(f"  skip {path.name}: export {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Registered only now, after a successful parse and export: a work whose .mxl fails
        # to convert should still be reachable via its .xml twin rather than being lost to
        # the dedup above.
        written[name] = path.as_posix()
        (out_dir / name).write_bytes(xml)
        entries.append(
            Entry(
                path=name,
                source=path.as_posix(),
                collection=coll,
                sha256=hashlib.sha256(xml).hexdigest(),
            )
        )
        taken[coll] += 1

    for extra in include_dirs:
        entries.extend(_fold_in(extra, out_dir, written, taken, include_limit))

    provenance = {
        "generator": "scripts/build_corpus.py",
        "music21_version": music21.__version__,
        "recipe": [{"collection": c, "cap": caps[c]} for c in sorted(caps)],
        "limit_per_collection": limit_per_collection,
        "include_dirs": [str(d) for d in include_dirs],
        "include_limit": include_limit,
        "count": len(entries),
        "by_collection": dict(sorted(taken.items())),
        "skipped": dict(sorted(skipped.items())),
        "files": [e.__dict__ for e in entries],
    }
    (out_dir / "CORPUS.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nwrote {len(entries)} scores to {out_dir}")
    for coll, n in sorted(taken.items()):
        print(f"  {coll:22} {n}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="corpus name under data/corpora/")
    ap.add_argument("--out", type=Path, default=None, help="override output directory")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap every collection at this many files (smoke builds)",
    )
    ap.add_argument(
        "--include-dir",
        type=Path,
        action="append",
        default=[],
        dest="include_dirs",
        help="fold an existing local corpus directory in (grand-staff material)",
    )
    ap.add_argument(
        "--include-limit",
        type=int,
        default=None,
        help="cap files taken from each --include-dir",
    )
    args = ap.parse_args()
    out = args.out or Path("data/corpora") / args.name
    build(out, args.limit, tuple(args.include_dirs), args.include_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
