"""Verovio raster engraver: MusicXML -> SVG -> (rsvg-convert) -> grayscale page.

The white background is verified-critical (CLAUDE.md): Verovio SVG is transparent, and a
transparent background rasterizes to solid black in grayscale. We force it with
``rsvg-convert -b white``.

CLAUDE.md names cairosvg for this step; we use ``rsvg-convert`` instead — cairosvg needs a
loadable native ``libcairo`` that is absent here, the same loader-path problem that made
Project 0 choose rsvg-convert in ADR 0001. The critical constraint (white background) is
preserved; the tool differs. See ADR 0005.
"""
from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np

from omrt.datagen.engravers.base import EngraverError, stack_vertically
from omrt.datagen.types import Image, MusicXMLStr
from omrt.symbolic.render import RenderError, render_verovio_svgs

_RSVG = "rsvg-convert"
_BASE_DPI = 96.0  # Verovio SVG reference DPI; zoom relative to it to hit the target dpi.


class VerovioEngraver:
    name = "verovio"

    def available(self) -> bool:
        return shutil.which(_RSVG) is not None

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        try:
            svgs = render_verovio_svgs(musicxml)
        except RenderError as exc:
            raise EngraverError(str(exc)) from exc
        pages = [self._svg_to_gray(svg, dpi) for svg in svgs]
        return stack_vertically(pages)

    def _svg_to_gray(self, svg: str, dpi: int) -> Image:
        exe = shutil.which(_RSVG)
        if exe is None:
            raise EngraverError(
                f"{_RSVG!r} not found on PATH; install it with `brew install librsvg`"
            )
        zoom = dpi / _BASE_DPI
        try:
            result = subprocess.run(
                [exe, "-f", "png", "-b", "white", "-z", f"{zoom:.6f}"],
                input=svg.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", "replace").strip()
            raise EngraverError(f"{_RSVG} failed: {detail}") from exc
        buffer = np.frombuffer(result.stdout, dtype=np.uint8)
        gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise EngraverError(f"{_RSVG} produced a PNG that OpenCV could not decode")
        return np.asarray(gray, dtype=np.uint8)
