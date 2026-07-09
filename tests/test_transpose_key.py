"""To-key transposition: correct spelling, matching key signature, no spurious doubles."""

from __future__ import annotations

import warnings

from helpers import key_sharps, make_score, pitches_of
from hypothesis import given, settings
from hypothesis import strategies as st
from music21 import key

from omrt.symbolic.keys import parse_key
from omrt.symbolic.transpose import transpose_to_key

C_MAJOR_SCALE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]

TARGET_KEYS = [
    "C major", "G major", "D major", "A major", "E major", "B major",
    "F# major", "F major", "B- major", "E- major", "A- major",
    "a minor", "e minor", "d minor", "g minor",
]


def test_c_major_to_fsharp_major_yields_esharp_not_f() -> None:
    out = transpose_to_key(make_score(C_MAJOR_SCALE), "F# major")
    assert pitches_of(out) == ["F#4", "G#4", "A#4", "B4", "C#5", "D#5", "E#5"]


@settings(max_examples=len(TARGET_KEYS), deadline=None)
@given(target=st.sampled_from(TARGET_KEYS))
def test_output_key_signature_matches_target(target: str) -> None:
    out = transpose_to_key(make_score(C_MAJOR_SCALE), target)
    assert key_sharps(out) == parse_key(target).sharps


@settings(max_examples=len(TARGET_KEYS), deadline=None)
@given(target=st.sampled_from(TARGET_KEYS))
def test_no_spurious_double_accidentals(target: str) -> None:
    """Any double accidental in the output must be diatonic to the target key;
    otherwise a simpler spelling existed and should have been used."""
    tgt = parse_key(target)
    diatonic = {p.name for p in tgt.pitches}
    out = transpose_to_key(make_score(C_MAJOR_SCALE), target)

    from music21 import converter

    for n in converter.parse(out).recurse().notes:
        for p in n.pitches:
            if p.accidental is not None and abs(p.accidental.alter) == 2:
                assert p.name in diatonic, f"spurious {p.name} in {target}"


def test_theoretical_key_is_accepted_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = transpose_to_key(make_score(C_MAJOR_SCALE), "G# major")
    assert key_sharps(out) == 8
    assert any("theoretical" in str(w.message) for w in caught)
