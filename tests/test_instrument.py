"""Instrument-mode transposition: written/sounding pitch, exactly one shift."""

from __future__ import annotations

import warnings

import pytest
from helpers import make_score
from music21 import converter, interval, key, pitch

from omrt.symbolic.instruments import resolve_instrument
from omrt.symbolic.transpose import transpose_for_instrument

CONCERT_SCALE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]


def test_bb_trumpet_appears_major_second_above_concert() -> None:
    concert = CONCERT_SCALE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = transpose_for_instrument(make_score(concert), "bb-trumpet")

    # Re-parse the exported MusicXML: music21's exporter also calls toWrittenPitch(),
    # so a wrong atSoundingPitch flag would double-transpose. Assert exactly one M2.
    written = [n.pitch for n in converter.parse(out).recurse().notes]
    assert len(written) == len(concert)
    for concert_name, w in zip(concert, written):
        iv = interval.Interval(noteStart=pitch.Pitch(concert_name), noteEnd=w)
        assert iv.name == "M2"
        assert iv.semitones == 2  # not 4 — that would be a double transposition


def test_bb_trumpet_key_signature_gains_two_sharps() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = transpose_for_instrument(make_score(CONCERT_SCALE, "C"), "bb-trumpet")
    sigs = list(converter.parse(out).recurse().getElementsByClass(key.KeySignature))
    assert sigs[0].sharps == 2  # concert C major -> written D major


def test_unknown_atsoundingpitch_warns_not_raises() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transpose_for_instrument(make_score(CONCERT_SCALE), "bb-trumpet")
    assert any("atSoundingPitch" in str(w.message) for w in caught)


def test_unknown_instrument_raises_with_valid_names() -> None:
    with pytest.raises(ValueError, match="bb-trumpet"):
        resolve_instrument("sousaphone")
