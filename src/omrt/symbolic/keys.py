"""Key detection and parsing.

Detection reads the score's explicit ``KeySignature`` (and mode, when the element
is a :class:`music21.key.Key`) first. ``.analyze()`` is a heuristic fallback and
emits a warning when used. See docs/decisions/0003-key-detection.md.
"""

from __future__ import annotations

import warnings
from typing import cast

from music21 import key, stream

__all__ = ["detect_key", "parse_key", "KeyDetectionWarning"]


class KeyDetectionWarning(UserWarning):
    """Raised when key detection falls back to heuristics or assumes a mode."""


def parse_key(text: str) -> key.Key:
    """Parse a key name such as ``'F# major'``, ``'bb minor'``, ``'C'``.

    A bare tonic defaults to major. Accepts music21 tonic spellings
    (``F#``, ``B-``/``Bb``). Raises ``ValueError`` on anything unparseable.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("empty key name")
    parts = raw.split()
    token = parts[0]
    if not token or token[0].upper() not in "ABCDEFG":
        raise ValueError(f"cannot parse key {text!r}: bad tonic {token!r}")
    # First char is the note letter; the rest are accidentals, where 'b' means
    # flat (music21 spells flats with '-'). So 'bb' -> 'B-', 'F#' -> 'F#'.
    tonic = token[0].upper() + token[1:].replace("b", "-")
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode not in {"major", "minor"}:
        raise ValueError(f"unsupported mode {mode!r}; use 'major' or 'minor'")
    try:
        return key.Key(tonic, mode)
    except Exception as exc:  # music21 raises assorted types on bad tonics
        raise ValueError(f"cannot parse key {text!r}: {exc}") from exc


def detect_key(score: stream.Score) -> key.Key:
    """Determine the score's key, preferring explicit notation over analysis.

    1. An explicit :class:`music21.key.Key` (carries tonic *and* mode) wins outright.
    2. A plain :class:`music21.key.KeySignature` gives the sharp/flat count but no
       mode; we assume major and warn.
    3. With no key signature at all, fall back to ``score.analyze('key')`` and warn.
    """
    keys = list(score.recurse().getElementsByClass(key.Key))
    if keys:
        return keys[0]

    sigs = list(score.recurse().getElementsByClass(key.KeySignature))
    if sigs:
        warnings.warn(
            "score has a key signature but no explicit mode; assuming major",
            KeyDetectionWarning,
            stacklevel=2,
        )
        return cast(key.Key, sigs[0].asKey("major"))

    warnings.warn(
        "score has no key signature; falling back to heuristic key analysis",
        KeyDetectionWarning,
        stacklevel=2,
    )
    analyzed = score.analyze("key")
    assert isinstance(analyzed, key.Key)
    return analyzed
