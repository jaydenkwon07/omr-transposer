"""MusicXML → a flat symbol stream that mirrors the PrIMuS SEMANTIC vocabulary.

Why mirror someone else's spellings instead of inventing our own: the point of a
semantic-mirroring granularity is that symbol error rate measured here tracks the
~2% the 2018 CRNN paper reports on PrIMuS. Renaming tokens would weaken that claim
for no benefit, and it would turn the calibration diff against the shipped
``.semantic`` files into a naming-difference hunt instead of a semantic one.

Grammar::

    clef-<shape><line>                  clef-G2, clef-C1
    keySignature-<MajorTonic>M          from the sharps/flats count, NOT detect_key
    timeSignature-C | C/ | <n>/<d>      preserves TimeSignature.symbol
    note-<step><alter?><oct>_<dur>[.]
    gracenote-<step><alter?><oct>_<dur>[.]
    rest-<dur>[.]
    multirest-<n>
    tie                                 bare, follows the note it ties from
    barline

Scope is single-staff monophonic, per the Project 2 non-goal.
"""
from __future__ import annotations

from music21 import (bar, base, chord, clef, converter, key, meter, note, pitch,
                     spanner, stream)

from omrt.datagen.types import MusicXMLStr

__all__ = ["to_symbols", "symbols_from_score", "parse_score"]

# music21 duration names on the left, PrIMuS's on the right. Anything absent passes
# through unchanged, so an unmapped duration shows up as itself rather than silently
# becoming a wrong-but-plausible token.
_DURATION_NAMES = {
    "breve": "double_whole",
    "longa": "quadruple_whole",
    "16th": "sixteenth",
    "32nd": "thirty_second",
    "64th": "sixty_fourth",
}

# Signed sharp count -> the major tonic that names the signature. PrIMuS names the
# *signature*, never the key: three flats is EbM even in C minor.
_MAJOR_TONIC_BY_SHARPS = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C",
    1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}

_TIME_SYMBOLS = {"common": "C", "cut": "C/"}


def to_symbols(musicxml: MusicXMLStr) -> list[str]:
    """Linearize a MusicXML score. Blank or unparseable input returns ``[]``."""
    score = parse_score(musicxml)
    return [] if score is None else symbols_from_score(score)


def parse_score(musicxml: MusicXMLStr) -> stream.Score | None:
    """Parse MusicXML to a Score, or ``None`` if it is blank or does not parse.

    A single-part fragment comes back as a Part, so wrap it: everything downstream
    may then assume a Score. Mirrors ``omrt.symbolic.transpose._parse``, kept separate
    because seam 4 does not import the transposition layer.
    """
    if not musicxml.strip():
        return None
    try:
        parsed = converter.parseData(musicxml, format="musicxml")
    except Exception:  # noqa: BLE001 — any parse failure is "no symbols", by contract
        return None
    if isinstance(parsed, stream.Score):
        return parsed
    if isinstance(parsed, stream.Stream):
        score = stream.Score()
        score.insert(0, parsed)
        return score
    return None


def symbols_from_score(score: stream.Score) -> list[str]:
    """Linearize an already-parsed score.

    Split out from :func:`to_symbols` so ``evaluate`` can normalize spelling on the
    parsed score without a round trip back through MusicXML text.
    """
    multirests, absorbed = _multirest_index(score)

    parts: list[stream.Stream[base.Music21Object]] = list(score.getElementsByClass(stream.Part))
    symbols: list[str] = []
    for part in parts or [score]:
        state: dict[str, str] = {}
        measures: list[stream.Stream[base.Music21Object]] = list(
            part.getElementsByClass(stream.Measure)
        )
        # A fragment with no measures is still a legal score; treat it as one block,
        # and emit no barline for it since none was notated.
        for block in measures or [part]:
            symbols.extend(_block_symbols(block, state, multirests, absorbed))
            if measures:
                symbols.append("barline")
    return symbols


def _multirest_index(
    score: stream.Score,
) -> tuple[dict[int, str], set[int]]:
    """Map the first rest of each multi-measure rest to its token; absorb the others.

    Keyed by ``id()`` because music21 rests are not hashable by value and two rests of
    the same duration are legitimately distinct events.
    """
    first: dict[int, str] = {}
    absorbed: set[int] = set()
    for span in score.recurse().getElementsByClass(spanner.MultiMeasureRest):
        # spannerStorage rather than getSpannedElements(): same elements, and it is
        # the typed accessor.
        rests = list(span.spannerStorage.getElementsByClass(note.Rest))
        if not rests:
            continue
        first[id(rests[0])] = f"multirest-{span.numRests}"
        absorbed.update(id(r) for r in rests[1:])
    return first, absorbed


def _block_symbols(
    block: stream.Stream[base.Music21Object],
    state: dict[str, str],
    multirests: dict[int, str],
    absorbed: set[int],
) -> list[str]:
    """Symbols for one measure. ``state`` carries the last-emitted clef/key/time."""
    symbols: list[str] = []
    for element in block.recurse():
        if isinstance(element, bar.Barline):
            continue  # barlines are emitted per measure, not per element
        token = _attribute_token(element, state)
        if token is not None:
            symbols.append(token)
            continue
        if isinstance(element, note.GeneralNote):
            symbols.extend(_event_symbols(element, multirests, absorbed))
    return symbols


def _attribute_token(element: object, state: dict[str, str]) -> str | None:
    """A clef/key/time token, but only when it differs from what is already in force."""
    if isinstance(element, clef.Clef):
        slot, token = "clef", f"clef-{element.sign}{element.line}"
    elif isinstance(element, key.KeySignature):
        tonic = _MAJOR_TONIC_BY_SHARPS.get(element.sharps)
        if tonic is None:
            return None
        slot, token = "key", f"keySignature-{tonic}M"
    elif isinstance(element, meter.TimeSignature):
        slot = "time"
        token = f"timeSignature-{_TIME_SYMBOLS.get(element.symbol or '', element.ratioString)}"
    else:
        return None

    if state.get(slot) == token:
        return None
    state[slot] = token
    return token


def _event_symbols(
    element: note.GeneralNote,
    multirests: dict[int, str],
    absorbed: set[int],
) -> list[str]:
    if isinstance(element, note.Rest):
        if id(element) in absorbed:
            return []
        multirest = multirests.get(id(element))
        return [multirest] if multirest else [f"rest-{_duration(element)}"]

    if isinstance(element, chord.Chord):
        # Out of scope for Project 2 (CTC cannot express a chord), but the metric must
        # stay a total function: spell each pitch as its own note, lowest first.
        pitches = sorted(element.pitches, key=lambda p: p.ps)
    elif isinstance(element, note.Note):
        pitches = [element.pitch]
    else:
        return []

    prefix = "gracenote" if element.duration.isGrace else "note"
    duration = _duration(element)
    symbols = [f"{prefix}-{_pitch_name(p)}{p.octave}_{duration}" for p in pitches]
    if element.tie is not None and element.tie.type in ("start", "continue"):
        symbols.append("tie")
    return symbols


def _pitch_name(p: pitch.Pitch) -> str:
    # music21 spells a flat as '-'; PrIMuS spells it 'b'.
    return p.name.replace("-", "b")


def _duration(element: note.GeneralNote) -> str:
    duration = element.duration
    name = _DURATION_NAMES.get(duration.type, duration.type)
    dots = "." * duration.dots
    fermata = "_fermata" if _has_fermata(element) else ""
    return f"{name}{dots}{fermata}"


def _has_fermata(element: note.GeneralNote) -> bool:
    return any(type(e).__name__ == "Fermata" for e in element.expressions)
