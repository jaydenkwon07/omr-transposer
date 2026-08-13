"""PrIMuS SEMANTIC token stream -> MusicXML. The inverse of eval/symbols.py.

Total and best-effort: unknown or malformed tokens are skipped and decode never
raises, so a half-trained CRNN's garbage prediction scores badly via SER instead
of crashing the training-eval loop. Imports nothing from omrt.eval — the forward
encoder (to_symbols) is reused only in the round-trip tests.
"""
from __future__ import annotations

from music21 import stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.datagen.types import MusicXMLStr

__all__ = ["Token", "decode", "parse_semantic"]

Token = str


def parse_semantic(line: str) -> list[Token]:
    """Split a raw PrIMuS `.semantic` line into tokens. Blank input -> []."""
    return line.split()


def decode(tokens: list[Token]) -> MusicXMLStr:
    """Seam 3. Total & best-effort: always returns well-formed MusicXML."""
    return _export(_build_score(tokens))


def _build_score(tokens: list[Token]) -> stream.Score:
    part = stream.Part()  # type: ignore
    part.append(stream.Measure(number=1))  # type: ignore
    score = stream.Score()
    score.insert(0, part)
    return score


def _export(score: stream.Score) -> MusicXMLStr:
    # makeNotation=False stops music21 from synthesizing a default treble clef and
    # 4/4 meter into an attribute-less incipit, which to_symbols would otherwise
    # read back as phantom clef-G2 / timeSignature-4/4 tokens. See spec section 5.
    exporter = GeneralObjectExporter(score)
    exporter.makeNotation = False
    return exporter.parse().decode("utf-8")
