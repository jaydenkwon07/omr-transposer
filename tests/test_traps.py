"""Traps. Each test encodes a music21 behavior verified against 10.5.

Run: pytest tests/test_traps.py -v
"""
from __future__ import annotations

import pytest
from music21 import (converter, harmony, instrument, interval, key, note,
                     pitch, spanner)

from omrt.symbolic import (transpose_by_interval, transpose_for_instrument,
                           transpose_to_key)


def notes(xml: str) -> list[str]:
    """Note names only. NOTE: .notes includes ChordSymbol objects, which have
    no .nameWithOctave. Always filter by class."""
    sc = converter.parse(xml)
    return [n.nameWithOctave for n in sc.recurse().getElementsByClass(note.Note)]


def keysigs(xml: str) -> list[int]:
    sc = converter.parse(xml)
    return [k.sharps for k in sc.recurse().getElementsByClass(key.KeySignature)]


# --- TRAP 1: diatonic vs chromatic ------------------------------------------
# music21: score.transpose(6) on C D E F yields F#4 G#4 B-4 B4.
#          score.transpose(Interval('A4'))      yields F#4 G#4 A#4 B4.
# A B-flat in a six-sharp key signature is the classic symptom of transposing
# by semitone count instead of by a directed diatonic interval.

def test_interval_transpose_spells_diatonically(c_major_scale):
    assert notes(transpose_by_interval(c_major_scale, "A4")) == \
        ["F#4", "G#4", "A#4", "B4"]


def test_to_key_spells_diatonically(c_major_scale):
    out = transpose_to_key(c_major_scale, "F# major")
    assert notes(out) == ["F#4", "G#4", "A#4", "B4"]
    assert keysigs(out) == [6]
    assert "B-" not in "".join(notes(out))


# --- TRAP 2: octave direction is undefined ----------------------------------
# Interval(Note('C4'), Note('B4')) is M7 UP (11 semitones), not m2 down.
# Naive tonic-to-tonic interval construction moves the whole piece up an octave.

def test_to_key_takes_the_short_way(c_major_scale):
    out = notes(transpose_to_key(c_major_scale, "B major"))
    assert out == ["B3", "C#4", "D#4", "E4"], \
        "C major -> B major should descend a minor 2nd, not ascend a major 7th"


# --- TRAP 3: instrument mode must normalize first ----------------------------
# A part carrying <transpose> parses with Part.atSoundingPitch == False.
# Applying the same instrument to it again must be a no-op, not a second shift.

def test_concert_to_bb_trumpet_goes_up_a_major_second(concert_score):
    assert notes(transpose_for_instrument(concert_score, "bb-trumpet")) == \
        ["D4", "E4", "F#4", "G4"]


def test_already_written_bb_part_is_idempotent(written_bb_trumpet):
    """THE bug. Feeding an existing Bb part back in must not shift it again."""
    assert notes(transpose_for_instrument(written_bb_trumpet, "bb-trumpet")) == \
        ["C4", "D4", "E4", "F4"]


def test_instrument_survives_export_reparse(concert_score):
    """music21's MusicXML exporter itself calls toWrittenPitch(inPlace=True).
    Only the atSoundingPitch flag prevents a double transposition."""
    once = transpose_for_instrument(concert_score, "bb-trumpet")
    reparsed = converter.parse(once)
    assert [n.nameWithOctave for n in
            reparsed.recurse().getElementsByClass(note.Note)] == \
        ["D4", "E4", "F#4", "G4"]


def test_unknown_instrument_raises_with_valid_names():
    with pytest.raises(ValueError, match="bb-trumpet"):
        transpose_for_instrument("<score/>", "kazoo")


# --- TRAP 4: instrument defaults are inconsistent ---------------------------
# Verified in music21 10.5:
#   Trumpet()       -> M-2   (Bb, not C!)
#   Clarinet()      -> M-2
#   AltoSaxophone() -> M-6
#   Horn()          -> P-5
#   Flute()         -> None
# So bb-trumpet works by accident; c-trumpet would silently be a Bb trumpet.
# Never rely on the default. The registry must set transposition explicitly.

def test_registry_sets_transposition_explicitly():
    from omrt.symbolic import _INSTRUMENTS
    assert instrument.Flute().transposition is None
    for name, expected in [("bb-trumpet", "M-2"), ("eb-alto-sax", "M-6"),
                           ("f-horn", "P-5")]:
        inst = _INSTRUMENTS[name]()
        assert inst.transposition == interval.Interval(expected), name


# --- TRAP 5: round trip ------------------------------------------------------

@pytest.mark.parametrize("iv", ["m2", "M3", "P5", "A4", "m7"])
def test_round_trip(c_major_scale, iv):
    up = transpose_by_interval(c_major_scale, iv)
    back = transpose_by_interval(up, f"-{iv}")
    assert notes(back) == notes(c_major_scale)


def test_melodic_intervals_preserved(c_major_scale):
    def steps(xml):
        ps = [n.pitch.ps for n in
              converter.parse(xml).recurse().getElementsByClass(note.Note)]
        return [b - a for a, b in zip(ps, ps[1:])]
    assert steps(transpose_by_interval(c_major_scale, "A4")) == steps(c_major_scale)


# --- TRAP 6: accidental display status ---------------------------------------
# transpose() wipes Accidental.displayStatus to None. Pitches stay correct;
# the PAGE renders wrong. And makeAccidentals() must be called on the PART:
#   Score.makeAccidentals()  -> F#=True  G#=True  A#=True   (all redundant!)
#   Part.makeAccidentals()   -> F#=False G#=False A#=False  F##=True  (correct)

def test_key_signature_accidentals_are_not_redrawn(c_major_scale):
    out = converter.parse(transpose_by_interval(c_major_scale, "A4"))
    shown = {n.nameWithOctave: (n.pitch.accidental.displayStatus
                                if n.pitch.accidental else None)
             for n in out.recurse().getElementsByClass(note.Note)}
    assert shown["F#4"] is False, "F# is in the F# major key signature"
    assert shown["A#4"] is False


# --- TRAP 7: chord symbols ---------------------------------------------------
# Verified: music21 DOES regenerate .figure. Cmaj7 -> F#maj7 under A4.
# Assert it, so a future refactor that rebuilds ChordSymbols by hand can't
# silently break it.

def test_chord_symbol_figure_transposes(lead_sheet):
    out = converter.parse(transpose_by_interval(lead_sheet, "A4"))
    figs = [cs.figure for cs in
            out.recurse().getElementsByClass(harmony.ChordSymbol)]
    assert figs == ["F#maj7"]


# --- TRAP 8: inferred spelling ------------------------------------------------
# A Pitch built from a MIDI number has spellingIsInferred == True and
# transposes differently: Pitch(ps=61) + A4 -> G, but Pitch('C#4') + A4 -> F##.

def test_inferred_spelling_is_detected(inferred_spelling):
    sc = converter.parse(inferred_spelling)
    for n in sc.recurse().getElementsByClass(note.Note):
        assert n.pitch.spellingIsInferred is False, \
            "MusicXML <step>/<alter> should give explicit spelling on parse"


def test_inferred_pitch_transposes_differently():
    p = pitch.Pitch(); p.ps = 61
    assert p.spellingIsInferred is True
    assert p.transpose("A4").name == "G"
    assert pitch.Pitch("C#4").transpose("A4").name == "F##"


# --- TRAP 9: scope --------------------------------------------------------
# ADR 0004 decided both: reject rather than guess. A theoretical target key
# (>7 accidentals) and a modulating input each raise ValueError; the caller who
# wants either must ask for a practical enharmonic or use interval mode instead.

def test_target_key_beyond_seven_accidentals(c_major_scale):
    with pytest.raises(ValueError, match="theoretical"):
        transpose_to_key(c_major_scale, "G# major")


def test_modulating_score(modulating):
    with pytest.raises(ValueError, match="modulates"):
        transpose_to_key(modulating, "D major")


def test_ottava_survives(with_ottava):
    out = converter.parse(transpose_by_interval(with_ottava, "M2"))
    assert len(list(out.recurse().getElementsByClass(spanner.Ottava))) == 1
