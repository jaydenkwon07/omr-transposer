"""Engraver registry and seeded selection.

Three engravers behind one protocol. The engraver is the highest-value variance axis
(three engravers beat a thousand augmentations of one), so selection is a first-class,
seeded step: ``pick_engraver`` draws from the *live* engravers using the same
``np.random.Generator`` that drives augmentation, so the whole sample is reproducible.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from omrt.datagen.engravers.base import Engraver, EngraverError
from omrt.datagen.engravers.lilypond import LilyPondEngraver
from omrt.datagen.engravers.musescore import MuseScoreEngraver
from omrt.datagen.engravers.verovio import VerovioEngraver

__all__ = [
    "Engraver",
    "EngraverError",
    "VerovioEngraver",
    "LilyPondEngraver",
    "MuseScoreEngraver",
    "ENGRAVERS",
    "build_engravers",
    "pick_engraver",
]

ENGRAVERS: dict[str, type[Engraver]] = {
    "verovio": VerovioEngraver,
    "lilypond": LilyPondEngraver,
    "musescore": MuseScoreEngraver,
}


def build_engravers(names: Sequence[str]) -> list[Engraver]:
    """Instantiate the named engravers, keeping only the ones that can actually run.

    An unknown name is an error (a typo should not silently shrink coverage); an
    unavailable-but-known engraver (a stub, a missing binary) is dropped quietly.
    """
    live: list[Engraver] = []
    for name in names:
        try:
            cls = ENGRAVERS[name]
        except KeyError:
            raise EngraverError(
                f"unknown engraver {name!r}; known: {sorted(ENGRAVERS)}"
            ) from None
        engraver = cls()
        if engraver.available():
            live.append(engraver)
    if not live:
        raise EngraverError(f"no engraver among {list(names)} is available")
    return live


def pick_engraver(engravers: Sequence[Engraver], rng: np.random.Generator) -> Engraver:
    """Uniformly choose one live engraver using the supplied generator."""
    if not engravers:
        raise EngraverError("no engravers to pick from")
    return engravers[int(rng.integers(len(engravers)))]
