"""LilyPond engraver — STUB.

Wired behind the protocol so the pipeline treats it uniformly, but not yet implemented:
``available()`` returns False, so ``build_engravers`` drops it and it never gets picked.
Fill this in once Verovio is working end-to-end (music21 ``.write('lilypond')`` or the
``lilypond`` binary -> PNG -> grayscale). See Project 1 TODO.
"""
from __future__ import annotations

from omrt.datagen.engravers.base import EngraverError
from omrt.datagen.types import Image, MusicXMLStr


class LilyPondEngraver:
    name = "lilypond"

    def available(self) -> bool:
        return False

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        raise EngraverError("LilyPondEngraver is a stub; not yet implemented")
