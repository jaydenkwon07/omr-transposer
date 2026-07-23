"""MuseScore 4 CLI engraver: MusicXML -> (mscore) -> per-page PNG -> grayscale.

MuseScore is the third leg of the engraver-diversity thesis (three engravers beat a
thousand augmentations of one, CLAUDE.md). The CLI is headless: ``mscore -r <dpi> -o
out.png in.musicxml`` writes one PNG per page as ``out-1.png``, ``out-2.png``, ... — the
page suffix is present even for a single-page score.

Unlike Verovio's tight SVG, MuseScore frames the music inside full printer-page margins, so
we ``trim_white_margins`` back to the ink box for comparable framing across engravers.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from omrt.datagen.engravers.base import (
    EngraverError,
    read_gray_over_white,
    stack_vertically,
    trim_white_margins,
)
from omrt.datagen.types import Image, MusicXMLStr

#: Binary names to try on PATH, then the default macOS app-bundle location.
_CANDIDATES = ("mscore", "MuseScore4", "MuseScore")
_MACOS_APP = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"

#: A render that leaves no PNG at all is retried a few times before giving up, in case the
#: shutdown abort (see _run) fired early enough to lose the write. Success is byte-identical
#: across runs, so retrying never threatens the (corpus_id, seed, config_hash) contract.
_MAX_TRIES = 4
_RETRY_SLEEP_S = 0.4


def _find_binary() -> str | None:
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found is not None:
            return found
    if os.path.exists(_MACOS_APP):
        return _MACOS_APP
    return None


class MuseScoreEngraver:
    name = "musescore"

    def available(self) -> bool:
        return _find_binary() is not None

    def to_image(self, musicxml: MusicXMLStr, *, dpi: int) -> Image:
        exe = _find_binary()
        if exe is None:
            raise EngraverError(
                "MuseScore binary not found; install MuseScore 4 or put `mscore` on PATH"
            )
        with tempfile.TemporaryDirectory(prefix="omrt-mscore-") as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "in.musicxml"
            src.write_text(musicxml, encoding="utf-8")
            pages = self._run(exe, src, tmpdir / "out.png", dpi)
            grays = [read_gray_over_white(str(p)) for p in pages]

        pad = max(8, dpi // 12)
        return trim_white_margins(stack_vertically(grays), pad=pad)

    def _run(self, exe: str, src: Path, out: Path, dpi: int) -> list[Path]:
        """Invoke mscore and return the rendered page PNGs, ordered by page number.

        Success is judged by *output, not exit code*: MuseScore 4 frequently aborts during
        shutdown teardown (SIGABRT, "mutex lock failed") after the PNG has already been
        written and flushed — a spurious failure with a valid, byte-identical result on
        disk. So we ignore the return code and look for the pages. Only a genuinely empty
        render is an error, retried a few times in case that empty run was itself the
        transient abort firing before the write."""
        cmd = [exe, "-r", str(dpi), "-o", str(out), str(src)]
        last = ""
        for attempt in range(_MAX_TRIES):
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise EngraverError("mscore timed out") from exc
            # MuseScore writes out-1.png, out-2.png, ...; page order is the numeric suffix.
            pages = sorted(
                out.parent.glob("out-*.png"),
                key=lambda p: int(p.stem.rsplit("-", 1)[1]),
            )
            if pages:
                return pages
            last = proc.stderr.decode("utf-8", "replace").strip()
            time.sleep(_RETRY_SLEEP_S * (attempt + 1))
        raise EngraverError(f"mscore produced no PNG pages after {_MAX_TRIES} tries: {last}")
