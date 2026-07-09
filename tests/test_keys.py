"""Key parsing and detection precedence (ADR 0003)."""

from __future__ import annotations

import warnings

import pytest
from helpers import make_score
from music21 import converter, key, note, stream

from omrt.symbolic.keys import KeyDetectionWarning, detect_key, parse_key


@pytest.mark.parametrize(
    "text, tonic, mode, sharps",
    [
        ("C major", "C", "major", 0),
        ("F# major", "F#", "major", 6),
        ("bb minor", "B-", "minor", -5),
        ("E- major", "E-", "major", -3),
        ("a minor", "A", "minor", 0),
        ("G", "G", "major", 1),
    ],
)
def test_parse_key(text: str, tonic: str, mode: str, sharps: int) -> None:
    k = parse_key(text)
    assert k.tonic.name == tonic
    assert k.mode == mode
    assert k.sharps == sharps


@pytest.mark.parametrize("bad", ["", "H major", "C dorian"])
def test_parse_key_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_key(bad)


def test_detect_explicit_key_no_warning() -> None:
    xml = make_score(["C4", "E4", "G4"], "D", "major")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        detected = detect_key(converter.parse(xml))
    assert detected.tonic.name == "D"
    assert detected.mode == "major"


def test_detect_plain_keysignature_assumes_major_and_warns() -> None:
    part = stream.Part()
    part.insert(0, key.KeySignature(2))  # two sharps, no mode
    part.append(note.Note("D4"))
    score = stream.Score()
    score.insert(0, part)
    with pytest.warns(KeyDetectionWarning, match="assuming major"):
        detected = detect_key(score)
    assert detected.sharps == 2
    assert detected.mode == "major"


def test_detect_no_keysignature_falls_back_to_analysis_and_warns() -> None:
    part = stream.Part()
    for n in ["C4", "E4", "G4", "C5"]:  # clearly C major, no key signature element
        part.append(note.Note(n))
    score = stream.Score()
    score.insert(0, part)
    with pytest.warns(KeyDetectionWarning, match="heuristic"):
        detected = detect_key(score)
    assert isinstance(detected, key.Key)
