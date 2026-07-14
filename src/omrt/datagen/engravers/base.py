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
