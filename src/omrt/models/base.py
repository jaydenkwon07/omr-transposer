from __future__ import annotations

from typing import Protocol, runtime_checkable

from omrt.datagen.types import Image
from omrt.decode import Token


@runtime_checkable
class Model(Protocol):
    """Seam 2: an image in, a semantic token sequence out."""

    def predict(self, image: Image) -> list[Token]: ...
