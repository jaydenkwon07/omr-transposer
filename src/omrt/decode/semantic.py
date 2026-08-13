"""PrIMuS SEMANTIC token stream -> MusicXML. The inverse of eval/symbols.py.

Total and best-effort: unknown or malformed tokens are skipped and decode never
raises, so a half-trained CRNN's garbage prediction scores badly via SER instead
of crashing the training-eval loop. Imports nothing from omrt.eval — the forward
encoder (to_symbols) is reused only in the round-trip tests.
"""
from __future__ import annotations

import re

from music21 import duration, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.datagen.types import MusicXMLStr

__all__ = ["Token", "decode", "parse_semantic"]

Token = str

# PrIMuS duration name -> music21 type name. Inverse of eval/symbols._DURATION_NAMES;
# a test asserts they are exact inverses so the two directions cannot drift.
_DURATION_TYPES = {
    "double_whole": "breve",
    "sixteenth": "16th",
    "thirty_second": "32nd",
    "sixty_fourth": "64th",
}

_PITCH_RE = re.compile(r"^([A-G])(b+|#+)?(-?\d+)$")


def parse_semantic(line: str) -> list[Token]:
    """Split a raw PrIMuS `.semantic` line into tokens. Blank input -> []."""
    return line.split()


def decode(tokens: list[Token]) -> MusicXMLStr:
    """Seam 3. Total & best-effort: always returns well-formed MusicXML."""
    return _export(_build_score(tokens))


def _duration(body: str) -> tuple[str, duration.Duration] | None:
    """Parse `<pitch?>_<durname>[.][_fermata]`. Returns (pitch_str, Duration).

    Fermata handling is added in Task 5; here the `_fermata` segment, if present,
    is simply not part of the duration name.

    Handles both note format `<pitch>_<durname>[.]` and rest format `<durname>[.]`.
    """
    idx = body.find("_")
    if idx < 0:
        # Rest format: no pitch, just duration name (and optional dots)
        pitch_str = ""
        dur_field = body
    else:
        # Note format: pitch_<durname>[.]
        pitch_str = body[:idx]
        dur_field = body[idx + 1 :]
    dots = len(dur_field) - len(dur_field.rstrip("."))
    name = dur_field[: len(dur_field) - dots]
    type_name = _DURATION_TYPES.get(name, name)
    try:
        dur = duration.Duration(type=type_name)
    except Exception:  # noqa: BLE001 — unknown duration name -> skip token
        return None
    dur.dots = dots
    return pitch_str, dur


def _note(body: str) -> note.Note | None:
    parsed = _duration(body)
    if parsed is None:
        return None
    pitch_str, dur = parsed
    m = _PITCH_RE.match(pitch_str)
    if m is None:
        return None
    step, accidental, octave = m.group(1), m.group(2) or "", m.group(3)
    # PrIMuS spells a flat 'b'; music21 spells it '-'. '#' is shared.
    alter = accidental.replace("b", "-")
    n = note.Note(f"{step}{alter}{octave}")
    n.duration = dur
    return n


def _rest(body: str) -> note.Rest | None:
    parsed = _duration(body)
    if parsed is None:
        return None
    _pitch_str, dur = parsed
    r = note.Rest()
    r.duration = dur
    return r


def _event(tok: Token) -> note.GeneralNote | None:
    prefix, _, body = tok.partition("-")
    if prefix == "note":
        return _note(body)
    if prefix == "rest":
        return _rest(body)
    return None


def _build_score(tokens: list[Token]) -> stream.Score:
    part: stream.Part = stream.Part()  # type: ignore[no-untyped-call]
    measures: list[stream.Measure] = []
    current = stream.Measure()
    for tok in tokens:
        if tok == "barline":
            if current.elements:
                measures.append(current)
            current = stream.Measure()
            continue
        event = _event(tok)
        if event is not None:
            current.append(event)  # type: ignore[no-untyped-call]
    if current.elements:  # trailing partial measure, only if it has content
        measures.append(current)
    for number, measure in enumerate(measures, start=1):
        measure.number = number
        part.append(measure)  # type: ignore[no-untyped-call]
    score: stream.Score = stream.Score()
    score.insert(0, part)
    return score


def _export(score: stream.Score) -> MusicXMLStr:
    # makeNotation=False stops music21 from synthesizing a default treble clef and
    # 4/4 meter into an attribute-less incipit, which to_symbols would otherwise
    # read back as phantom clef-G2 / timeSignature-4/4 tokens. See spec section 5.
    # For empty scores (no measures), return minimal valid MusicXML.
    has_measures = any(part.getElementsByClass(stream.Measure) for part in score.parts)
    if not has_measures:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n'
            '<score-partwise version="3.1">\n'
            '  <part-list>\n'
            '    <score-part id="P1"/>\n'
            '  </part-list>\n'
            '  <part id="P1"/>\n'
            '</score-partwise>\n'
        )
    exporter = GeneralObjectExporter(score)
    exporter.makeNotation = False
    return exporter.parse().decode("utf-8")
