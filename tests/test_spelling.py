"""Enharmonic spelling normalization (ADR 0002)."""

from __future__ import annotations

from music21 import key, note, stream

from omrt.symbolic.spelling import normalize_spelling


def _single(name: str, key_context: key.Key | None) -> str:
    s = stream.Stream()
    s.append(note.Note(name))
    normalize_spelling(s, key_context)
    return s.recurse().notes[0].pitch.nameWithOctave


def test_keeps_diatonic_double_sharp() -> None:
    # F## is the leading tone of G# major — correct, must be left alone.
    assert _single("F##4", key.Key("G#")) == "F##4"


def test_respells_spurious_double_sharp() -> None:
    # E## is not diatonic to C major; simplest spelling is F#.
    assert _single("E##4", key.Key("C")) == "F#4"


def test_respell_can_cross_octave_without_changing_pitch() -> None:
    # B##4 (midi 73) has no simple spelling in its own octave; C#5 preserves pitch.
    result = _single("B##4", key.Key("C"))
    assert result == "C#5"


def test_none_context_leaves_spelling_untouched() -> None:
    assert _single("E##4", None) == "E##4"


def test_single_accidentals_never_touched() -> None:
    assert _single("F#4", key.Key("C")) == "F#4"
    assert _single("B-4", key.Key("C")) == "B-4"
