"""Adversarial score fixtures for the symbolic layer.

Each fixture isolates ONE notational feature that breaks naive transposition.
All return MusicXML strings, matching the str -> str boundary of omrt.symbolic.
"""
from __future__ import annotations

import pytest
from music21 import (chord, clef, harmony, instrument, interval, key, meter,
                     note, pitch, spanner, stream)


def _to_xml(score: stream.Score) -> str:
    from music21.musicxml.m21ToXml import GeneralObjectExporter
    return GeneralObjectExporter().parse(score).decode("utf-8")


def _score(part: stream.Part) -> stream.Score:
    sc = stream.Score()
    sc.insert(0, part)
    return sc


def _measure(notes, ks=key.Key("C"), ts="4/4") -> stream.Measure:
    m = stream.Measure(number=1)
    m.insert(0, meter.TimeSignature(ts))
    m.insert(0, ks)
    m.insert(0, clef.TrebleClef())
    for n in notes:
        m.append(note.Note(n) if isinstance(n, str) else n)
    return m


@pytest.fixture
def c_major_scale() -> str:
    """Baseline. C D E F in C major. Up an A4 -> F# G# A# B in F# major."""
    p = stream.Part()
    p.append(_measure(["C4", "D4", "E4", "F4"]))
    return _to_xml(_score(p))


@pytest.fixture
def concert_score() -> str:
    """Concert pitch, no <transpose> element. Parts parse with
    atSoundingPitch == True."""
    p = stream.Part()
    p.append(_measure(["C4", "D4", "E4", "F4"]))
    return _to_xml(_score(p))


@pytest.fixture
def written_bb_trumpet() -> str:
    """ALREADY at written pitch for a Bb trumpet, carrying a <transpose>
    element. Notes read C D E F; they SOUND Bb C D Eb.
    Parts parse with atSoundingPitch == False."""
    p = stream.Part()
    p.append(_measure(["C4", "D4", "E4", "F4"]))
    tr = instrument.Trumpet()
    tr.transposition = interval.Interval("M-2")
    p.insert(0, tr)
    sc = _score(p)
    sc.atSoundingPitch = False
    return _to_xml(sc)


@pytest.fixture
def lead_sheet() -> str:
    """Melody plus a Cmaj7 chord symbol."""
    p = stream.Part()
    m = _measure(["C4", "E4", "G4", "B4"])
    m.insert(0, harmony.ChordSymbol("Cmaj7"))
    p.append(m)
    return _to_xml(_score(p))


@pytest.fixture
def with_ottava() -> str:
    """8va spanner. music21 stores these notes at WRITTEN octave and shifts
    them on toSoundingPitch()."""
    p = stream.Part()
    ns = [note.Note("C5"), note.Note("D5"), note.Note("E5"), note.Note("F5")]
    m = _measure(ns)
    ott = spanner.Ottava()
    ott.addSpannedElements(ns)
    p.insert(0, ott)
    p.append(m)
    return _to_xml(_score(p))


@pytest.fixture
def modulating() -> str:
    """C major, then G major halfway through. transpose_to_key reads ONE key."""
    p = stream.Part()
    p.append(_measure(["C4", "D4", "E4", "F4"]))
    m2 = stream.Measure(number=2)
    m2.insert(0, key.Key("G"))
    for n in ["G4", "A4", "B4", "C5"]:
        m2.append(note.Note(n))
    p.append(m2)
    return _to_xml(_score(p))


@pytest.fixture
def inferred_spelling() -> str:
    """Pitches built from MIDI numbers. spellingIsInferred is True, and
    music21 documents that this changes transposition results."""
    p = stream.Part()
    ns = []
    for ps in (60, 61, 62, 64):
        n = note.Note()
        n.pitch.ps = ps
        ns.append(n)
    p.append(_measure(ns))
    return _to_xml(_score(p))
