"""Tests for the corpus builder (``scripts/build_corpus.py``).

The corpus directory is one third of the ``(corpus_id, seed, config_hash)`` reproducibility
triple, so the property that matters is that ``CORPUS.json`` describes exactly what is on
disk. The first build violated it: distinct works share movement filenames across the
music21 corpus (``.../opus18no1/movement1.mxl`` and ``.../opus59no1/movement1.mxl``), so
keying the output name on the stem overwrote earlier files while the manifest still listed
them — 140 entries describing 116 files.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("build_corpus", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_corpus"] = module
    spec.loader.exec_module(module)
    return module


build_corpus = _load()


def test_same_stem_in_different_works_gets_distinct_names() -> None:
    """The exact collision that corrupted the first build."""
    a = Path("/x/music21/corpus/beethoven/opus18no1/movement1.mxl")
    b = Path("/x/music21/corpus/beethoven/opus59no1/movement1.mxl")
    assert build_corpus._safe_name("beethoven", a) != build_corpus._safe_name(  # type: ignore[attr-defined]
        "beethoven", b
    )


def test_names_are_filesystem_safe_and_musicxml() -> None:
    name = build_corpus._safe_name(  # type: ignore[attr-defined]
        "trecento", Path("/x/music21/corpus/trecento/Fava Ave/no 3.xml")
    )
    assert name.endswith(".musicxml")
    assert build_corpus._UNSAFE.search(name.removesuffix(".musicxml")) is None  # type: ignore[attr-defined]


def test_same_work_in_two_formats_collapses_to_one_id() -> None:
    """``foo.mxl`` and ``foo.xml`` are the same piece shipped twice. They must share a
    logical id so the builder deduplicates them instead of putting the score in the corpus
    twice and skewing unit sampling toward it."""
    base = Path("/x/music21/corpus/trecento/PMFC_13_01-Kyrie")
    assert build_corpus._logical_id(base.with_suffix(".mxl")) == (  # type: ignore[attr-defined]
        build_corpus._logical_id(base.with_suffix(".xml"))  # type: ignore[attr-defined]
    )


def test_logical_id_strips_only_the_real_suffix() -> None:
    """``.replace('.xml', '')`` would corrupt a name that merely contains the substring."""
    p = Path("/x/music21/corpus/demos/prelude.xml.variant.mxl")
    assert build_corpus._logical_id(p) == "prelude.xml.variant"  # type: ignore[attr-defined]


def test_core_corpus_names_are_unique_after_dedup() -> None:
    """Corpus-wide, against the real music21 corpus: once same-work duplicates collapse,
    every remaining score gets its own filename. This is what CORPUS.json depends on."""
    from music21 import corpus

    paths = [p for p in corpus.getCorePaths() if p.suffix in build_corpus._MUSICXML]  # type: ignore[attr-defined]
    seen: dict[str, Path] = {}
    for p in paths:
        coll = build_corpus._collection_of(p)  # type: ignore[attr-defined]
        name = build_corpus._safe_name(coll, p)  # type: ignore[attr-defined]
        prior = seen.get(name)
        if prior is not None:
            # Allowed only when it is genuinely the same work in another format.
            assert prior.with_suffix("") == p.with_suffix(""), (
                f"collision between different works: {p} vs {prior}"
            )
            continue
        seen[name] = p


def test_fold_in_refuses_to_overwrite(tmp_path: Path) -> None:
    """The guard, not just the naming: a collision must raise rather than silently drop a
    score and leave the manifest lying about it."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.musicxml").write_text("<score/>", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    written = {"src__a.musicxml": "somewhere/else.musicxml"}
    with pytest.raises(RuntimeError, match="collision"):
        build_corpus._fold_in(src, out, written, build_corpus.Counter(), None)  # type: ignore[attr-defined]
