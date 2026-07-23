"""The raster engraver protocol.

Distinct from Project 0's ``Renderer`` on purpose: ``Renderer.to_pdf`` serves the shipped
product and returns a document (bytes); an ``Engraver`` serves the data pipeline and
returns an in-memory grayscale image, because the very next thing the pipeline does is
augment in pixel space. Returning PNG bytes here would just mean decoding them again.

``name`` lives on the protocol so the manifest requirement ("record which engraver made
each sample") is enforced by the type: an engraver cannot exist without identifying itself.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from omrt.datagen.types import Image, MusicXMLStr


class EngraverError(RuntimeError):
    """Engraving failed (bad input, missing backend binary, or a converter error)."""


@runtime_checkable
class Engraver(Protocol):
    #: Stable identifier recorded in every sample's metadata.
    name: str

    def available(self) -> bool:
        """Whether this backend can run on the current machine (binaries present, etc.)."""
        ...

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        """Engrave ``musicxml`` to a single grayscale page (255=paper, 0=ink).

        Multi-page scores are flattened to one tall page so the label and image stay 1:1.
        """
        ...


def read_gray_over_white(path: str) -> Image:
    """Decode a PNG page to grayscale (255=paper, 0=ink), compositing any alpha over white.

    Engravers export with a transparent background (MuseScore RGBA; the same trap CLAUDE.md
    flags for Verovio), so grayscaling directly would rely on transparent pixels happening
    to carry white RGB. Compositing over white first makes paper paper regardless of what
    sits under alpha=0. Opaque RGB or already-gray PNGs pass straight through."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise EngraverError(f"could not decode PNG {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        raw = (bgr * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return np.asarray(raw, dtype=np.uint8)


def stack_vertically(pages: list[Image]) -> Image:
    """Concatenate per-page rasters into one tall page, right-padding narrower pages with
    paper so widths match. Keeps image and label 1:1 for multi-page scores."""
    if not pages:
        raise EngraverError("no pages to stack")
    if len(pages) == 1:
        return pages[0]
    width = max(p.shape[1] for p in pages)
    padded = [
        cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=255)
        for p in pages
    ]
    return np.vstack(padded).astype(np.uint8)


def trim_white_margins(image: Image, *, pad: int) -> Image:
    """Crop surrounding paper to the ink bounding box, then re-pad with ``pad`` px of paper.

    Full-page engravers (MuseScore, LilyPond) frame the music inside wide printer margins,
    unlike Verovio's tight SVG. Trimming to the ink box makes the three engravers'
    framing comparable and keeps the ink fraction meaningful; augmentation reintroduces
    crop/margin variance downstream in a labelled, mask-safe way. A blank page is returned
    unchanged (no ink means no bounding box)."""
    ink = np.argwhere(image < 250)
    if ink.size == 0:
        return image
    top, left = ink.min(axis=0)
    bottom, right = ink.max(axis=0)
    h, w = image.shape
    top = max(0, int(top) - pad)
    left = max(0, int(left) - pad)
    bottom = min(h, int(bottom) + 1 + pad)
    right = min(w, int(right) + 1 + pad)
    return np.ascontiguousarray(image[top:bottom, left:right])
