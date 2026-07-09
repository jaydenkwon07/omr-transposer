"""Transposition operations over MusicXML strings.

Every function here takes a MusicXML string and returns a MusicXML string — no file
I/O, no rendering. This is the part of Project 0 that survives to production unchanged.

Three modes:

* :func:`transpose_to_key` — analyze the source key, compute the directed interval to
  the target tonic, and transpose by it so spelling comes out right (C major → F#
  major yields E#, not F).
* :func:`transpose_by_interval` — transpose by a named interval; descending intervals
  are written with the sign before the number (``m-3``) or with a leading ``-``.
* :func:`transpose_for_instrument` — go concert → written pitch through music21's
  ``toSoundingPitch``/``toWrittenPitch`` so a Bb trumpet part appears a M2 above concert.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import Any

from music21 import converter, instrument, interval, key, pitch, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.symbolic.instruments import resolve_instrument
from omrt.symbolic.keys import detect_key, parse_key
from omrt.symbolic.spelling import normalize_spelling

__all__ = [
    "transpose_to_key",
    "transpose_by_interval",
    "transpose_for_instrument",
    "parse_interval",
]


def _parse(musicxml: str) -> stream.Score:
    parsed = converter.parseData(musicxml, format="musicxml")
    if not isinstance(parsed, stream.Score):
        # A single-part fragment parses as a Part/Score depending on the source;
        # wrap so downstream code can always assume a Score.
        score = stream.Score()
        score.insert(0, parsed)
        return score
    return parsed


def _serialize(score: stream.Score) -> str:
    return GeneralObjectExporter(score).parse().decode("utf-8")


def parse_interval(name: str) -> interval.Interval:
    """Parse an interval name into a music21 ``Interval``.

    Accepts ascending forms (``m3``, ``P5``, ``A4``) and two descending spellings:
    the music21-native sign-before-number (``m-3``) and a leading minus (``-m3``).
    """
    text = name.strip()
    if not text:
        raise ValueError("empty interval name")
    descending = text.startswith("-")
    core = text[1:] if descending else text
    try:
        iv = interval.Interval(core)
    except Exception as exc:
        raise ValueError(f"cannot parse interval {name!r}: {exc}") from exc
    if descending:
        # reverse() is unannotated in music21, hence both suppressions.
        return iv.reverse()  # type: ignore[no-untyped-call, no-any-return]
    return iv


def transpose_by_interval(musicxml: str, interval_name: str) -> str:
    """Transpose every pitch by ``interval_name``. Spelling is left as music21
    produces it (interval transposition is already canonical)."""
    iv = parse_interval(interval_name)
    score = _parse(musicxml)
    transposed = score.transpose(iv, inPlace=False)
    assert transposed is not None
    return _serialize(transposed)


def transpose_to_key(musicxml: str, target_key: str) -> str:
    """Transpose from the score's detected key to ``target_key``.

    The interval is computed between tonic *pitches* (not pitch classes) so diatonic
    spelling is preserved: C major → F# major sends B to E#. The resulting key
    signature matches the target, and spurious double accidentals are normalized.
    """
    target = parse_key(target_key)
    _warn_if_theoretical(target)
    score = _parse(musicxml)
    source = detect_key(score)

    iv = _shortest_transposition(source, target)
    transposed = score.transpose(iv, inPlace=False)
    assert transposed is not None

    _force_key_signature(transposed, target)
    normalize_spelling(transposed, target)
    return _serialize(transposed)


def transpose_for_instrument(musicxml: str, instrument_name: str) -> str:
    """Render a concert-pitch score as it should appear for a transposing instrument.

    Normalizes to sounding pitch first (handling an unknown ``atSoundingPitch`` flag
    explicitly), installs the requested instrument, then converts to written pitch.
    Leaving the score at written pitch (``atSoundingPitch is False``) is what stops
    music21's exporter from transposing a second time.
    """
    inst = resolve_instrument(instrument_name)
    score = _parse(musicxml)

    _normalize_to_sounding(score)

    targets: Sequence[stream.Stream[Any]] = list(score.parts) or [score]
    for part in targets:
        for existing in list(part.getElementsByClass(instrument.Instrument)):
            part.remove(existing)
        part.insert(0, copy.deepcopy(inst))

    score.toWrittenPitch(inPlace=True)
    return _serialize(score)


def _shortest_transposition(source: key.Key, target: key.Key) -> interval.Interval:
    """Directed, correctly-spelled interval from the source tonic to the target tonic
    that minimizes register shift.

    The interval must be spelled (m2, A4, ...) so diatonic spelling survives — C major
    to F# major sends B to E#. But tonic-to-tonic in a fixed octave can move the whole
    piece up a major 7th (C -> B) when a descending minor 2nd is meant. So we try the
    target tonic in nearby octaves and keep the smallest absolute interval; ties resolve
    upward, which keeps C -> F# an ascending augmented 4th.
    """
    assert source.tonic is not None and target.tonic is not None
    start = pitch.Pitch(source.tonic.name)
    start.octave = 4
    best: interval.Interval | None = None
    for octave in (4, 3, 5):  # 4 first so a tritone tie stays ascending
        end = pitch.Pitch(target.tonic.name)
        end.octave = octave
        candidate = interval.Interval(noteStart=start, noteEnd=end)
        if best is None or abs(candidate.semitones) < abs(best.semitones):
            best = candidate
    assert best is not None
    return best


def _normalize_to_sounding(score: stream.Score) -> None:
    """Ensure the score is at sounding (concert) pitch, declaring the assumption
    explicitly when the flag is unknown rather than letting a StreamException escape."""
    if score.atSoundingPitch == "unknown":
        warnings.warn(
            "atSoundingPitch is unknown; assuming the input is concert (sounding) pitch",
            stacklevel=2,
        )
        score.atSoundingPitch = True
    score.toSoundingPitch(inPlace=True)


def _force_key_signature(score: stream.Score, target: key.Key) -> None:
    """Guarantee the notated key signature matches ``target``.

    Transposing an explicit key signature already lands on the target for a
    non-modulating piece; but a score detected via analysis has no key signature to
    transpose, so insert one at the start of each part.
    """
    existing = list(score.recurse().getElementsByClass(key.KeySignature))
    desired = key.KeySignature(target.sharps)
    if existing:
        for sig in existing:
            sig.sharps = target.sharps
        return
    targets: Sequence[stream.Stream[Any]] = list(score.parts) or [score]
    for part in targets:
        part.insert(0, copy.deepcopy(desired))


def _warn_if_theoretical(target: key.Key) -> None:
    """Theoretical keys (>7 accidentals, e.g. G# major) are accepted as requested,
    but we point at the practical enharmonic. See ADR 0002."""
    if abs(target.sharps) > 7 and target.tonic is not None:
        enharmonic = target.tonic.getEnharmonic()
        practical = enharmonic.name.replace("-", "b") if enharmonic is not None else "its enharmonic"
        warnings.warn(
            f"{target} is a theoretical key ({target.sharps} sharps/flats); "
            f"consider {practical} {target.mode} for a practical key signature",
            stacklevel=3,
        )
