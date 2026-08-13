"""PrIMuS SEMANTIC token stream -> MusicXML. The inverse of eval/symbols.py.

Total and best-effort: unknown or malformed tokens are skipped and decode never
raises, so a half-trained CRNN's garbage prediction scores badly via SER instead
of crashing the training-eval loop. Imports nothing from omrt.eval — the forward
encoder (to_symbols) is reused only in the round-trip tests.
"""
from __future__ import annotations

import re

from music21 import base, clef, duration, expressions, key, meter, note, spanner, stream
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

# PrIMuS names the *signature* by its major tonic. Inverse of
# eval/symbols._MAJOR_TONIC_BY_SHARPS; the dict-inverse test asserts they match.
_SHARPS_BY_MAJOR_TONIC = {
    "Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1,
    "C": 0,
    "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
}
# PrIMuS symbol -> music21 TimeSignature.symbol. Inverse of eval/symbols._TIME_SYMBOLS.
_TIME_SYMBOLS = {"C": "common", "C/": "cut"}

_NOTE_RE = re.compile(r"^([A-G])(b+|#+)?(-?\d+)_(.+)$")


def parse_semantic(line: str) -> list[Token]:
    """Split a raw PrIMuS `.semantic` line into tokens. Blank input -> []."""
    return line.split()


def decode(tokens: list[Token]) -> MusicXMLStr:
    """Seam 3. Total & best-effort: always returns well-formed MusicXML."""
    return _export(_build_score(tokens))


def _duration(field: str) -> tuple[duration.Duration, bool] | None:
    """Parse a duration FIELD: `<name>[.]` with an optional trailing `_fermata`.
    `name` may itself contain an underscore (`thirty_second`), so never split on
    interior underscores. Returns (Duration, has_fermata)."""
    fermata = field.endswith("_fermata")
    if fermata:
        field = field[: -len("_fermata")]
    dots = len(field) - len(field.rstrip("."))
    name = field[: len(field) - dots]
    type_name = _DURATION_TYPES.get(name, name)
    try:
        dur = duration.Duration(type=type_name)
    except Exception:  # noqa: BLE001 — unknown duration name -> skip token
        return None
    dur.dots = dots
    return dur, fermata


def _attribute(prefix: str, body: str) -> base.Music21Object | None:
    """Parse clef, keySignature, and timeSignature attribute tokens."""
    if prefix == "clef":
        m = re.match(r"^([A-G])(-?\d+)$", body)
        if m is None:
            return None
        c = clef.Clef()
        c.sign, c.line = m.group(1), int(m.group(2))
        return c
    if prefix == "keySignature":
        tonic = body[:-1] if body.endswith("M") else body
        sharps = _SHARPS_BY_MAJOR_TONIC.get(tonic)
        return None if sharps is None else key.KeySignature(sharps)
    if prefix == "timeSignature":
        symbol = _TIME_SYMBOLS.get(body)
        if symbol is not None:
            ts = meter.TimeSignature()
            # Set numerator/denominator first, then symbol (order matters in music21).
            if symbol == "common":
                ts.numerator = 4
                ts.denominator = 4
            else:  # "cut"
                ts.numerator = 2
                ts.denominator = 2
            ts.symbol = symbol
            return ts
        else:
            ts = meter.TimeSignature()
            try:
                ts.ratioString = body
            except Exception:  # noqa: BLE001 — malformed ratio -> skip token
                return None
            return ts
    return None


def _note(body: str) -> note.Note | None:
    m = _NOTE_RE.match(body)
    if m is None:
        return None
    step, accidental, octave, field = m.group(1), m.group(2) or "", m.group(3), m.group(4)
    parsed = _duration(field)
    if parsed is None:
        return None
    dur, fermata = parsed
    # PrIMuS spells a flat 'b'; music21 spells it '-'. '#' is shared.
    alter = accidental.replace("b", "-")
    n = note.Note(f"{step}{alter}{octave}")
    n.duration = dur
    if fermata:
        n.expressions.append(expressions.Fermata())  # type: ignore[no-untyped-call]
    return n


def _rest(body: str) -> note.Rest | None:
    # A rest body is the duration field only (no pitch).
    parsed = _duration(body)
    if parsed is None:
        return None
    dur, fermata = parsed
    r = note.Rest()
    r.duration = dur
    if fermata:
        r.expressions.append(expressions.Fermata())  # type: ignore[no-untyped-call]
    return r


def _multirest(body: str) -> tuple[list[note.Rest], spanner.Spanner | None]:
    """`multirest-N` -> N whole-rest events, all in the *same* measure, wrapped by
    one `MultiMeasureRest` spanner.

    music21's own MusicXML round trip forces this shape. The brief's original design
    put a single `Rest` in the measure with `numRests` set directly on the spanner:
    that exports fine (`GeneralObjectExporter` reads `numRests` straight off the
    spanner), but on *reimport* `xmlToM21.PartParser.applyMultiMeasureRest` doesn't
    trust the `<measure-style>` count as given — it decrements a counter seeded from
    that count once per `<note><rest/></note>` it actually parses, and only inserts
    the spanner once the counter hits zero. A single rest reimports as a bare
    `rest-whole`, never becoming a spanner at all. Giving the spanner N real Rest
    events (still confined to one `stream.Measure`, so still one PrIMuS "barline"
    worth of measure) produces N `<note>` elements under one `<measure-style>` tag,
    which is also the same shape music21's own MusicXML fixtures use for a
    multi-measure rest. `eval/symbols._multirest_index` already anticipates exactly
    this: it emits one token for the spanner's first rest and silently absorbs the
    rest.
    """
    try:
        count = int(body)
    except ValueError:
        return [], None
    if count < 1:
        return [], None
    rests = [note.Rest(type="whole") for _ in range(count)]
    mmr = spanner.MultiMeasureRest(*rests)  # type: ignore[no-untyped-call]
    mmr.numRests = count
    return rests, mmr


def _event(tok: Token) -> tuple[list[base.Music21Object], spanner.Spanner | None]:
    prefix, _, body = tok.partition("-")
    attr = _attribute(prefix, body)
    if attr is not None:
        return [attr], None
    if prefix == "note":
        n = _note(body)
        return ([n] if n is not None else []), None
    if prefix == "gracenote":
        n = _note(body)
        return ([n.getGrace()] if n is not None else []), None  # type: ignore[no-untyped-call]
    if prefix == "rest":
        r = _rest(body)
        return ([r] if r is not None else []), None
    if prefix == "multirest":
        rests, mmr = _multirest(body)
        events: list[base.Music21Object] = list(rests)
        return events, mmr
    return [], None


def _build_score(tokens: list[Token]) -> stream.Score:
    part: stream.Part = stream.Part()  # type: ignore[no-untyped-call]
    spanners: list[spanner.Spanner] = []
    measures: list[stream.Measure] = []
    current = stream.Measure()
    for tok in tokens:
        if tok == "barline":
            if current.elements:
                measures.append(current)
            current = stream.Measure()
            continue
        events, span = _event(tok)
        for event in events:
            current.append(event)  # type: ignore[no-untyped-call]
        if span is not None:
            spanners.append(span)
    if current.elements:  # trailing partial measure, only if it has content
        measures.append(current)
    for number, measure in enumerate(measures, start=1):
        measure.number = number
        part.append(measure)  # type: ignore[no-untyped-call]
    for span in spanners:
        part.insert(0, span)
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
