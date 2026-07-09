"""Enharmonic spelling normalization.

Produces a canonical spelling by removing *spurious* double accidentals. This module
has no dependency on :mod:`omrt.symbolic.transpose` on purpose: Project 3 reuses the
same normal form as a label canonicalizer for model training data.

Precise definition (see docs/decisions/0002-enharmonic-spelling.md): a pitch carrying a
double accidental (``alter == +2 or -2``) is *spurious* iff it is **not** diatonic to
the supplied key context. Diatonic double accidentals are correct and kept — e.g. F##
is the leading tone of G# major. With no key context, spelling is left untouched:
interval transposition already yields a canonical spelling and there is no tonal frame
in which to judge a "simpler" alternative.
"""

from __future__ import annotations

from typing import Any, TypeVar

from music21 import key, stream

__all__ = ["normalize_spelling"]

_StreamT = TypeVar("_StreamT", bound="stream.Stream[Any]")


def normalize_spelling(score: _StreamT, key_context: key.Key | None) -> _StreamT:
    """Respell spurious double accidentals in place; return the same score.

    ``key_context`` is the tonal frame the pitches live in (typically the target key
    after a to-key transposition). If ``None``, nothing is changed.
    """
    if key_context is None:
        return score

    diatonic = {p.name for p in key_context.pitches}

    for element in score.recurse().notes:
        for p in element.pitches:  # a Note has one pitch; a Chord has several
            acc = p.accidental
            if acc is None or abs(acc.alter) != 2:
                continue
            if p.name in diatonic:
                continue  # e.g. F## in G# major — correct, leave it
            simpler = p.simplifyEnharmonic(mostCommon=True)
            # Copy octave too: simplify can cross an octave boundary (B##4 -> C#5)
            # and dropping that would change the sounding pitch.
            p.step = simpler.step
            p.octave = simpler.octave
            p.accidental = simpler.accidental

    return score
