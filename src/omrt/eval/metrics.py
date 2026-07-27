"""Seam 4: ``evaluate(predicted, truth) -> Metrics``.

Symbol error rate over the PrIMuS-mirroring symbol stream. Project 2 scope is SER
only; ``Metrics`` is shaped to grow OMR-NED and TEDn fields in Project 3.
"""
from __future__ import annotations

from dataclasses import dataclass

from music21 import key, stream

from omrt.datagen.types import MusicXMLStr
from omrt.eval.editdistance import levenshtein
from omrt.eval.symbols import parse_score, symbols_from_score
from omrt.symbolic.spelling import normalize_spelling

__all__ = ["Metrics", "evaluate"]


@dataclass(frozen=True)
class Metrics:
    """One comparison. ``substitutions + insertions + deletions == edits``."""

    ser: float
    edits: int
    ref_len: int
    substitutions: int
    insertions: int
    deletions: int


def evaluate(predicted: MusicXMLStr, truth: MusicXMLStr) -> Metrics:
    """Symbol error rate of ``predicted`` against ``truth``.

    ``ser = edits / len(symbols(truth))``. Raw ``edits`` is symmetric in its two
    arguments; ``ser`` is not, because the normalizing length is the truth's.

    Degenerate reference: an empty truth scores ``0.0`` when the prediction is also
    empty and ``1.0`` otherwise, which makes ``evaluate(x, x) == 0.0`` and
    ``evaluate("", x) == 1.0`` exact rather than approximate.
    """
    predicted_symbols = _normalized_symbols(predicted)
    truth_symbols = _normalized_symbols(truth)

    ops = levenshtein(predicted_symbols, truth_symbols)
    ref_len = len(truth_symbols)
    if ref_len == 0:
        ser = 0.0 if ops.distance == 0 else 1.0
    else:
        ser = ops.distance / ref_len

    return Metrics(
        ser=ser,
        edits=ops.distance,
        ref_len=ref_len,
        substitutions=ops.substitutions,
        insertions=ops.insertions,
        deletions=ops.deletions,
    )


def _normalized_symbols(musicxml: MusicXMLStr) -> list[str]:
    """Parse, canonicalize the spelling, linearize.

    The spelling pass runs on both sides, so the model is never punished for a
    spelling difference that renders as identical pixels. The key context comes from
    the score's own signature rather than :func:`omrt.symbolic.keys.detect_key`: this
    runs per sample during training, and detect_key's no-signature branch falls back
    to heuristic ``analyze('key')``, which is both slow and a warning storm here.
    Scores without a signature simply go un-normalized, which is what
    :func:`normalize_spelling` already does with a ``None`` context.
    """
    score = parse_score(musicxml)
    if score is None:
        return []

    normalize_spelling(score, _key_context(score))
    return symbols_from_score(score)


def _key_context(score: stream.Score) -> key.Key | None:
    keys = list(score.recurse().getElementsByClass(key.Key))
    if keys:
        return keys[0]
    sigs = list(score.recurse().getElementsByClass(key.KeySignature))
    if sigs:
        as_key = sigs[0].asKey("major")
        return as_key if isinstance(as_key, key.Key) else None
    return None
