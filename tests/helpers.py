"""Shared test helpers (not a test module)."""

from __future__ import annotations

from music21 import converter, key, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter


def make_score(pitches: list[str], key_name: str = "C", mode: str = "major") -> str:
    """Build a one-part single-staff MusicXML string from note names."""
    score = stream.Score()
    part = stream.Part()
    part.insert(0, key.Key(key_name, mode))
    for name in pitches:
        part.append(note.Note(name, quarterLength=1))
    score.insert(0, part)
    return GeneralObjectExporter(score).parse().decode("utf-8")


def pitches_of(musicxml: str) -> list[str]:
    """Return note names with octave, in document order."""
    parsed = converter.parse(musicxml)
    return [n.pitch.nameWithOctave for n in parsed.recurse().notes]


def key_sharps(musicxml: str) -> int | None:
    """Return the first key signature's signed sharp count, or None."""
    parsed = converter.parse(musicxml)
    sigs = list(parsed.recurse().getElementsByClass(key.KeySignature))
    return sigs[0].sharps if sigs else None
