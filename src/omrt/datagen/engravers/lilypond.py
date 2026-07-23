"""LilyPond engraver: MusicXML -> (musicxml2ly) -> .ly -> (lilypond --png) -> grayscale.

The third leg of the engraver-diversity thesis. Two binaries, both shipped with LilyPond:
``musicxml2ly`` converts MusicXML to LilyPond source, then ``lilypond`` renders it to PNG.
Both must be present for the engraver to be ``available``.

LilyPond frames music inside full page margins (like MuseScore, unlike Verovio's tight SVG),
so we ``trim_white_margins`` to the ink box for comparable framing. Output is byte-identical
across runs, honoring the (corpus_id, seed, config_hash) reproducibility contract.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from omrt.datagen.engravers.base import (
    EngraverError,
    read_gray_over_white,
    stack_vertically,
    trim_white_margins,
)
from omrt.datagen.types import Image, MusicXMLStr

_MUSICXML2LY = "musicxml2ly"
_LILYPOND = "lilypond"


class LilyPondEngraver:
    name = "lilypond"

    def available(self) -> bool:
        return shutil.which(_MUSICXML2LY) is not None and shutil.which(_LILYPOND) is not None

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        m2ly = shutil.which(_MUSICXML2LY)
        ly = shutil.which(_LILYPOND)
        if m2ly is None or ly is None:
            raise EngraverError(
                "LilyPond not found; install it (provides `musicxml2ly` and `lilypond`)"
            )
        with tempfile.TemporaryDirectory(prefix="omrt-lily-") as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "in.musicxml"
            src.write_text(musicxml, encoding="utf-8")
            source = tmpdir / "score.ly"
            self._run([m2ly, str(src), "-o", str(source)], tmpdir, "musicxml2ly")
            if not source.exists():
                raise EngraverError("musicxml2ly produced no LilyPond source")
            # Suppress LilyPond's "LilyPond vX.Y.Z" footer tagline: it is an engraver
            # watermark absent from the label — a leak that lets a model identify the
            # engraver from its footprint instead of learning engraving-robustness.
            with source.open("a", encoding="utf-8") as fh:
                fh.write("\n\\paper { tagline = ##f }\n")

            stem = tmpdir / "page"
            self._run(
                [ly, f"-dresolution={dpi}", "--png", "-o", str(stem), str(source)],
                tmpdir,
                "lilypond",
            )
            grays = [read_gray_over_white(str(p)) for p in _pages(tmpdir, stem.name)]

        pad = max(8, dpi // 12)
        return trim_white_margins(stack_vertically(grays), pad=pad)

    def _run(self, cmd: list[str], cwd: Path, label: str) -> None:
        try:
            subprocess.run(cmd, cwd=cwd, capture_output=True, check=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise EngraverError(f"{label} timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", "replace").strip()
            raise EngraverError(f"{label} failed: {detail}") from exc


def _pages(tmpdir: Path, stem: str) -> list[Path]:
    """LilyPond writes ``<stem>.png`` for a single page and ``<stem>-page1.png``,
    ``<stem>-page2.png``, ... for a multi-page score. Prefer the numbered pages; fall back
    to the single-page name."""
    numbered = sorted(
        tmpdir.glob(f"{stem}-page*.png"),
        key=lambda p: int(p.stem.rsplit("page", 1)[1]),
    )
    if numbered:
        return numbered
    single = tmpdir / f"{stem}.png"
    if single.exists():
        return [single]
    raise EngraverError("lilypond produced no PNG pages")
