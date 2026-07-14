"""Label production: a unit score -> canonical MusicXML.

The label is MusicXML, never tokens (that is per-model, downstream). It passes through
``spelling.normalize_spelling`` so identical-sounding pixels always carry one canonical
enharmonic spelling — Transcoda's one-to-many lesson applied early.

Critical: ``generate`` engraves *and* labels from the string this returns, so the rendered
glyphs and the label are the same normalized score. Normalizing the label but rendering the
original would let a respelled double accidental drift the image away from its label.
"""
from __future__ import annotations

import copy

from music21 import stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.datagen.types import MusicXMLStr
from omrt.symbolic.keys import detect_key
from omrt.symbolic.spelling import normalize_spelling

__all__ = ["canonical_musicxml"]


def canonical_musicxml(score: stream.Score) -> MusicXMLStr:
    """Return the unit's canonical MusicXML: spelling normalized to the unit's own key.

    Operates on a copy so the caller's score is untouched (it may be reused for stats).
    """
    work = copy.deepcopy(score)
    key_context = detect_key(work)
    normalize_spelling(work, key_context)
    return GeneralObjectExporter(work).parse().decode("utf-8")
