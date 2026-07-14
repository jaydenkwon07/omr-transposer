"""MuseScore CLI engraver — STUB.

Wired behind the protocol but not yet implemented: ``available()`` returns False (it will
later probe for the ``mscore`` binary on PATH), so ``build_engravers`` drops it. Fill in
with ``mscore -o out.png`` once Verovio is working end-to-end. See Project 1 TODO.
"""
from __future__ import annotations

from omrt.datagen.engravers.base import EngraverError
from omrt.datagen.types import Image, MusicXMLStr


class MuseScoreEngraver:
    name = "musescore"

    def available(self) -> bool:
        return False

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        raise EngraverError("MuseScoreEngraver is a stub; not yet implemented")
