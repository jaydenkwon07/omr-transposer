"""Corpus loading and unit extraction.

A source score (e.g. an OpenScore Lieder song, voice + piano ≈ 3 staves) is decomposed
into *units* that respect the single-staff / grand-staff ceiling (CLAUDE.md non-goal, see
ADR 0006): each plain vocal ``Part`` becomes a single-staff unit, and each contiguous run
of ``PartStaff`` (the piano) becomes a grand-staff unit. A unit is a fresh ``Score`` so
downstream rendering and labelling never see the rest of the page.

Survives the Project 0 traps: ``.notes`` is always filtered by ``note.Note`` class, and a
unit whose pitches carry inferred spelling (``spellingIsInferred``) is warned about — an
ambiguous label before augmentation.
"""
from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music21 import converter, layout, note, stream

from omrt.datagen.types import CorpusId

__all__ = [
    "Corpus",
    "CorpusItem",
    "load_corpus",
    "InferredSpellingWarning",
    "SINGLE_STAFF",
    "GRAND_STAFF",
]

SINGLE_STAFF = "single_staff"
GRAND_STAFF = "grand_staff"

_SUFFIXES = (".mxl", ".musicxml", ".xml")


class InferredSpellingWarning(UserWarning):
    """A unit has a pitch with ``spellingIsInferred`` — an ambiguous label."""


@dataclass(frozen=True)
class CorpusItem:
    corpus_id: CorpusId  # e.g. "Composer/Song/lc123.mxl#s0"
    source: Path
    unit: str  # SINGLE_STAFF | GRAND_STAFF
    staves: int
    score: stream.Score  # a fresh score holding only this unit


class Corpus:
    """A directory of MusicXML, addressable by source file, decomposed into units."""

    def __init__(self, root: Path, sources: list[Path], max_staves: int) -> None:
        self.root = root
        self._sources = sources
        self.max_staves = max_staves

    @property
    def sources(self) -> list[Path]:
        return self._sources

    def __len__(self) -> int:
        return len(self._sources)

    def units_in(self, source: Path) -> list[CorpusItem]:
        """Parse one source and return its eligible units. A broken file warns and
        yields nothing rather than aborting a whole generation run."""
        try:
            parsed = converter.parse(source)
        except Exception as exc:  # noqa: BLE001 - corpus files are third-party, skip broken
            warnings.warn(f"skipping unparseable {source}: {type(exc).__name__}: {exc}")
            return []
        score = parsed if isinstance(parsed, stream.Score) else _wrap(parsed)

        rel = source.relative_to(self.root).as_posix()
        items: list[CorpusItem] = []
        for tag, unit_kind, group in _unit_groups(score, self.max_staves):
            unit_score = _build_unit(group)
            if not unit_score.recurse().getElementsByClass(note.Note):
                continue  # empty staff, nothing to engrave or label
            _warn_on_inferred_spelling(unit_score, f"{rel}#{tag}")
            items.append(
                CorpusItem(
                    corpus_id=f"{rel}#{tag}",
                    source=source,
                    unit=unit_kind,
                    staves=len(group),
                    score=unit_score,
                )
            )
        return items

    def items(self) -> Iterator[CorpusItem]:
        """Every unit in the corpus. A full parse pass — use for coverage/manifests on
        small sets, not inside the hot generation loop."""
        for source in self._sources:
            yield from self.units_in(source)


def load_corpus(root: Path, *, max_staves: int = 2) -> Corpus:
    if not root.is_dir():
        raise FileNotFoundError(f"corpus dir does not exist: {root}")
    sources = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _SUFFIXES and p.is_file()
    )
    return Corpus(root, sources, max_staves)


def _wrap(part: stream.Stream[Any]) -> stream.Score:
    score = stream.Score()
    score.insert(0, part)
    return score


def _unit_groups(
    score: stream.Score, max_staves: int
) -> list[tuple[str, str, list[stream.Part]]]:
    """Group a score's parts into units. Contiguous ``PartStaff`` runs form grand-staff
    units; each plain ``Part`` is its own single-staff unit. Order is preserved so tags
    are stable across runs."""
    groups: list[tuple[str, str, list[stream.Part]]] = []
    parts = list(score.parts)
    i = 0
    n_single = 0
    n_grand = 0
    while i < len(parts):
        part = parts[i]
        if isinstance(part, stream.PartStaff):
            run: list[stream.Part] = [part]
            j = i + 1
            while j < len(parts) and isinstance(parts[j], stream.PartStaff):
                run.append(parts[j])
                j += 1
            i = j
            if len(run) == 1:
                groups.append((f"s{n_single}", SINGLE_STAFF, run))
                n_single += 1
            elif len(run) <= max_staves:
                groups.append((f"g{n_grand}", GRAND_STAFF, run))
                n_grand += 1
            # a run longer than max_staves is not a supported unit; drop it
        else:
            groups.append((f"s{n_single}", SINGLE_STAFF, [part]))
            n_single += 1
            i += 1
    return groups


def _build_unit(group: list[stream.Part]) -> stream.Score:
    """A fresh Score holding deep copies of the group's parts. A grand-staff group gets a
    braced StaffGroup so an engraver renders the two staves as one keyboard system."""
    unit = stream.Score()
    parts = [copy.deepcopy(p) for p in group]
    for part in parts:
        unit.insert(0, part)
    if len(parts) > 1:
        unit.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    return unit


def _warn_on_inferred_spelling(unit: stream.Score, unit_id: str) -> None:
    for n in unit.recurse().getElementsByClass(note.Note):
        if n.pitch.spellingIsInferred:
            warnings.warn(
                f"{unit_id}: pitch {n.nameWithOctave} has inferred spelling; "
                "label is ambiguous before augmentation",
                InferredSpellingWarning,
                stacklevel=2,
            )
            return
