"""Project 0: the symbolic layer.

MusicXML in, transposed MusicXML out (music21), rendered to PDF (Verovio). Public
surface: the three transposition operations and the renderer interface.
"""

from omrt.symbolic.render import Renderer, VerovioRenderer
from omrt.symbolic.transpose import (
    transpose_by_interval,
    transpose_for_instrument,
    transpose_to_key,
)

__all__ = [
    "transpose_to_key",
    "transpose_by_interval",
    "transpose_for_instrument",
    "Renderer",
    "VerovioRenderer",
]
