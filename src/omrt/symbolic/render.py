"""Rendering MusicXML to PDF, behind a small interface.

The interface is deliberately narrow so a LilyPond backend can replace Verovio later
without touching callers. See docs/decisions/0001-renderer.md for why the SVG→PDF step
shells out to ``rsvg-convert`` instead of using cairosvg.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from typing import Protocol, runtime_checkable

import verovio
from pypdf import PdfReader, PdfWriter

__all__ = ["Renderer", "VerovioRenderer", "RenderError"]


class RenderError(RuntimeError):
    """Rendering failed (bad input, missing system tool, or a converter error)."""


@runtime_checkable
class Renderer(Protocol):
    """Turns a MusicXML string into PDF bytes."""

    def to_pdf(self, musicxml: str) -> bytes: ...


class VerovioRenderer:
    """Verovio engraves MusicXML to per-page SVG; ``rsvg-convert`` turns each SVG into
    a one-page PDF; pypdf concatenates them into a single document."""

    _RSVG = "rsvg-convert"

    def to_pdf(self, musicxml: str) -> bytes:
        svgs = self._render_svgs(musicxml)
        page_pdfs = [self._svg_to_pdf(svg) for svg in svgs]
        return self._merge(page_pdfs)

    def _render_svgs(self, musicxml: str) -> list[str]:
        toolkit = verovio.toolkit()
        if not toolkit.loadData(musicxml):
            raise RenderError("verovio could not parse the MusicXML input")
        page_count = toolkit.getPageCount()
        if page_count < 1:
            raise RenderError("verovio produced no pages")
        return [toolkit.renderToSVG(i) for i in range(1, page_count + 1)]

    def _svg_to_pdf(self, svg: str) -> bytes:
        exe = shutil.which(self._RSVG)
        if exe is None:
            raise RenderError(
                f"{self._RSVG!r} not found on PATH; install it with "
                "`brew install librsvg` (macOS) or your platform's librsvg package"
            )
        try:
            result = subprocess.run(
                [exe, "-f", "pdf"],
                input=svg.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", "replace").strip()
            raise RenderError(f"{self._RSVG} failed: {detail}") from exc
        return result.stdout

    @staticmethod
    def _merge(page_pdfs: list[bytes]) -> bytes:
        writer = PdfWriter()
        for pdf in page_pdfs:
            reader = PdfReader(io.BytesIO(pdf))
            for page in reader.pages:
                writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()
