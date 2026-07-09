"""Interval-mode transposition: round-trip and interval-preservation invariants."""

from __future__ import annotations

from helpers import make_score, pitches_of
from hypothesis import given, settings
from hypothesis import strategies as st
from music21 import interval, pitch

from omrt.symbolic.transpose import transpose_by_interval

# Ascending interval names music21 parses; each also used descending as "-name".
INTERVALS = ["m2", "M2", "m3", "M3", "P4", "A4", "d5", "P5", "m6", "M6", "m7", "M7", "P8"]

_NAMES = [
    f"{letter}{acc}{octave}"
    for letter in "CDEFGAB"
    for acc in ("", "#", "-")
    for octave in (3, 4, 5)
]


@st.composite
def _pitch_lists(draw: st.DrawFn, min_size: int = 1) -> list[str]:
    return draw(st.lists(st.sampled_from(_NAMES), min_size=min_size, max_size=6))


@settings(max_examples=40, deadline=None)
@given(pitches=_pitch_lists(), name=st.sampled_from(INTERVALS))
def test_roundtrip_interval_then_reverse_restores_pitches(
    pitches: list[str], name: str
) -> None:
    xml = make_score(pitches)
    once = transpose_by_interval(xml, name)
    back = transpose_by_interval(once, f"-{name}")
    assert pitches_of(back) == pitches_of(xml)


@settings(max_examples=40, deadline=None)
@given(pitches=_pitch_lists(min_size=2), name=st.sampled_from(INTERVALS))
def test_melodic_intervals_preserved_exactly(pitches: list[str], name: str) -> None:
    original = pitches_of(make_score(pitches))
    transposed = pitches_of(transpose_by_interval(make_score(pitches), name))
    assert len(original) == len(transposed)

    for (a, b), (c, d) in zip(
        zip(original, original[1:]), zip(transposed, transposed[1:])
    ):
        before = interval.Interval(noteStart=pitch.Pitch(a), noteEnd=pitch.Pitch(b))
        after = interval.Interval(noteStart=pitch.Pitch(c), noteEnd=pitch.Pitch(d))
        # Same spelled interval AND same chromatic size: spelling is preserved.
        assert before.name == after.name
        assert before.semitones == after.semitones
