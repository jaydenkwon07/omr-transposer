"""End-to-end data generator tests: corpus extraction, reproducibility, label fidelity,
image sanity, engraver coverage, and the manifest deliverable.

Rendering tests need ``rsvg-convert``; they skip if it is absent. The corpus is a handful
of tiny scores written to a tmp dir so the suite is hermetic and fast (no dependency on the
gitignored Lieder download).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from music21 import converter, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.datagen import (GenConfig, build_engravers, generate, load_corpus,
                          pick_engraver, write_dataset)
from omrt.datagen.corpus import (GRAND_STAFF, SINGLE_STAFF,
                                 InferredSpellingWarning, _unit_groups,
                                 _warn_on_inferred_spelling)
from omrt.datagen.engravers import VerovioEngraver

_HAVE_RSVG = VerovioEngraver().available()
needs_rsvg = pytest.mark.skipif(not _HAVE_RSVG, reason="rsvg-convert not installed")


def _single_staff_xml(tonic: str, ts: str, pitches: list[str]) -> str:
    from music21 import clef, key, meter
    p = stream.Part()
    m = stream.Measure(number=1)
    m.insert(0, clef.TrebleClef())
    m.insert(0, key.Key(tonic))
    m.insert(0, meter.TimeSignature(ts))
    for n in pitches:
        m.append(note.Note(n))
    p.append(m)
    sc = stream.Score()
    sc.insert(0, p)
    return GeneralObjectExporter(sc).parse().decode("utf-8")


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    (d / "a.musicxml").write_text(_single_staff_xml("C", "4/4", ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]))
    (d / "b.musicxml").write_text(_single_staff_xml("G", "3/4", ["G4", "A4", "B4", "C5", "D5", "E5"]))
    (d / "c.musicxml").write_text(_single_staff_xml("F", "6/8", ["F4", "G4", "A4", "B-4", "C5", "D5"]))
    return d


# --- corpus / unit extraction ------------------------------------------------

def test_unit_extraction_splits_voice_and_piano() -> None:
    """ADR 0006: a vocal Part is a single-staff unit; a PartStaff run is grand-staff."""
    sc = stream.Score()
    voice = stream.Part()
    voice.append(note.Note("C5"))
    rh = stream.PartStaff()
    rh.append(note.Note("E4"))
    lh = stream.PartStaff()
    lh.append(note.Note("C3"))
    for part in (voice, rh, lh):
        sc.insert(0, part)
    kinds = sorted((kind, len(parts)) for _, kind, parts in _unit_groups(sc, max_staves=2))
    assert kinds == sorted([(SINGLE_STAFF, 1), (GRAND_STAFF, 2)])


def test_inferred_spelling_warns() -> None:
    sc = stream.Score()
    p = stream.Part()
    n = note.Note()
    n.pitch.ps = 61  # built from MIDI -> spellingIsInferred is True
    p.append(n)
    sc.insert(0, p)
    with pytest.warns(InferredSpellingWarning):
        _warn_on_inferred_spelling(sc, "unit#s0")


def test_broken_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "bad.musicxml").write_text("this is not musicxml")
    corpus = load_corpus(tmp_path)
    with pytest.warns(UserWarning):
        assert list(corpus.items()) == []


# --- reproducibility ---------------------------------------------------------

@needs_rsvg
def test_generate_is_byte_identical_for_a_seed(corpus_dir: Path) -> None:
    cfg = GenConfig(corpus_dir=corpus_dir, seed=5, dpi=80)
    first = [img for img, _ in generate(3, cfg)]
    second = [img for img, _ in generate(3, cfg)]
    assert len(first) == 3
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


# --- label fidelity ----------------------------------------------------------

@needs_rsvg
def test_label_round_trips_without_loss(corpus_dir: Path) -> None:
    cfg = GenConfig(corpus_dir=corpus_dir, seed=1, dpi=80, geometric=False, photometric=False)
    labels = [label for _, label in generate(3, cfg)]
    assert labels
    for label in labels:
        sc = converter.parse(label)
        n1 = len(sc.recurse().getElementsByClass(note.Note))
        assert n1 > 0
        reparsed = converter.parse(GeneralObjectExporter(sc).parse().decode("utf-8"))
        n2 = len(reparsed.recurse().getElementsByClass(note.Note))
        assert n1 == n2, "note count changed across a serialize/parse round trip"


# --- image sanity ------------------------------------------------------------

@needs_rsvg
def test_no_black_or_white_images(corpus_dir: Path) -> None:
    cfg = GenConfig(corpus_dir=corpus_dir, seed=2, dpi=80)
    images = [img for img, _ in generate(4, cfg)]
    assert images
    for img in images:
        dark = float((img < 100).mean())  # true ink, below the lighting-gradient floor
        assert 0.001 < dark < 0.6, f"suspicious ink fraction {dark}"
        assert img.max() > 200, "no bright paper — image may be black"
        assert img.min() < 60, "no dark ink — image may be blank"


# --- engraver coverage -------------------------------------------------------

class _FakeEngraver:
    def __init__(self, name: str) -> None:
        self.name = name

    def available(self) -> bool:
        return True

    def to_image(self, musicxml: str, *, dpi: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def test_engraver_selection_covers_all_available() -> None:
    """Over 300 draws every live engraver appears — the coverage guarantee, tested on the
    selection logic so it does not depend on all three backends being installed."""
    engravers = [_FakeEngraver("verovio"), _FakeEngraver("lilypond"), _FakeEngraver("musescore")]
    rng = np.random.default_rng(0)
    seen = {pick_engraver(engravers, rng).name for _ in range(300)}
    assert seen == {"verovio", "lilypond", "musescore"}


@pytest.mark.xfail(reason="lilypond/musescore engravers are stubs until wired (Project 1 TODO)")
def test_all_three_real_engravers_available() -> None:
    live = {e.name for e in build_engravers(("verovio", "lilypond", "musescore"))}
    assert live == {"verovio", "lilypond", "musescore"}


# --- manifest ----------------------------------------------------------------

@needs_rsvg
def test_manifest_records_engraver_and_histograms(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cfg = GenConfig(corpus_dir=corpus_dir, seed=3, dpi=80)
    manifest = json.loads(write_dataset(3, cfg, out).read_text())
    assert manifest["count"] == 3
    assert set(manifest["aggregate"]["engraver"]) <= {"verovio", "lilypond", "musescore"}
    for s in manifest["samples"]:
        assert s["engraver"]
        assert {"unit", "staves", "key_sharps", "time_sig", "note_density"} <= set(s["histograms"])
        assert (out / s["image_path"]).exists()
        assert (out / s["label_path"]).exists()
