"""VerovioRenderer integration: MusicXML actually becomes a PDF.

Requires the `rsvg-convert` binary (brew install librsvg); skipped if absent.
"""

from __future__ import annotations

import shutil

import pytest
from helpers import make_score

from omrt.symbolic.render import Renderer, RenderError, VerovioRenderer

pytestmark = pytest.mark.skipif(
    shutil.which("rsvg-convert") is None, reason="rsvg-convert not installed"
)


def test_renderer_satisfies_protocol() -> None:
    assert isinstance(VerovioRenderer(), Renderer)


def test_produces_pdf_bytes() -> None:
    pdf = VerovioRenderer().to_pdf(make_score(["C4", "D4", "E4", "F4"]))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_bad_input_raises_render_error() -> None:
    with pytest.raises(RenderError):
        VerovioRenderer().to_pdf("<not-musicxml/>")
