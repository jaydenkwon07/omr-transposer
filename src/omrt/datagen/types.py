"""Shared type aliases for the data generator.

One image representation, documented here, from the engraver outward: a 2-D grayscale
``uint8`` ndarray, ``0`` = ink, ``255`` = paper. Single channel because sheet music is
single channel and the model consumes grayscale; committing to one channel order also
sidesteps the RGB-vs-BGR footgun between PIL (RGB) and cv2 (BGR).
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# A grayscale page: shape (H, W), dtype uint8, 255 = paper / 0 = ink.
Image: TypeAlias = npt.NDArray[np.uint8]

# The label is always MusicXML. Never a token vocabulary — that is per-model, downstream.
MusicXMLStr: TypeAlias = str

# Stable identifier for one generation unit, e.g. "Composer/Song/lc123.mxl#voice0".
CorpusId: TypeAlias = str

# Pixel rectangle (top, left, bottom, right), half-open on bottom/right.
BBox: TypeAlias = "tuple[int, int, int, int]"
