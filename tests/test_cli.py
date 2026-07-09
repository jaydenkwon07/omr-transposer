"""CLI wiring, including the argparse leading-hyphen fix for descending intervals."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_score

from omrt.symbolic.cli import main

SCALE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]


class FakeRenderer:
    """Captures the MusicXML it is asked to render; returns sentinel PDF bytes."""

    def __init__(self) -> None:
        self.seen: str | None = None

    def to_pdf(self, musicxml: str) -> bytes:
        self.seen = musicxml
        return b"%PDF-fake"


@pytest.fixture
def score_file(tmp_path: Path) -> Path:
    p = tmp_path / "in.musicxml"
    p.write_text(make_score(SCALE), encoding="utf-8")
    return p


def _run(args: list[str]) -> tuple[int, FakeRenderer, FakeRenderer]:
    renderer = FakeRenderer()
    code = main(args, renderer=renderer)
    return code, renderer, renderer


def test_to_key_mode(score_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    code, renderer, _ = _run(
        ["transpose", str(score_file), "--to-key", "F# major", "--out", str(out)]
    )
    assert code == 0
    assert out.read_bytes() == b"%PDF-fake"
    assert renderer.seen is not None


def test_descending_interval_with_leading_hyphen(
    score_file: Path, tmp_path: Path
) -> None:
    """The whole point of the argparse fix: `--by-interval -m3` must not be
    misread as an option."""
    out = tmp_path / "out.pdf"
    code, _, _ = _run(
        ["transpose", str(score_file), "--by-interval", "-m3", "--out", str(out)]
    )
    assert code == 0
    assert out.exists()


def test_descending_interval_native_form(score_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    code, _, _ = _run(
        ["transpose", str(score_file), "--by-interval", "m-3", "--out", str(out)]
    )
    assert code == 0


def test_instrument_mode(score_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    code, _, _ = _run(
        ["transpose", str(score_file), "--for-instrument", "bb-trumpet", "--out", str(out)]
    )
    assert code == 0


def test_missing_input_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    code = main(
        ["transpose", str(tmp_path / "nope.musicxml"), "--by-interval", "P5", "--out", str(out)],
        renderer=FakeRenderer(),
    )
    assert code == 1
    assert not out.exists()


def test_bad_interval_returns_error(score_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    code = main(
        ["transpose", str(score_file), "--by-interval", "zzz", "--out", str(out)],
        renderer=FakeRenderer(),
    )
    assert code == 1


def test_modes_are_mutually_exclusive(score_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    with pytest.raises(SystemExit):
        main(
            ["transpose", str(score_file), "--to-key", "G major",
             "--by-interval", "P5", "--out", str(out)],
            renderer=FakeRenderer(),
        )
