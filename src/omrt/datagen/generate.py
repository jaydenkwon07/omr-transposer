"""Seam 1: ``generate(n, config) -> Iterator[tuple[Image, MusicXMLStr]]``.

Signature verbatim from CLAUDE.md. Reproducibility is the load-bearing property: every
sample is a deterministic function of ``(corpus_id, seed, config_hash)``. Source/unit
selection is driven by one selection generator (seeded from ``seed`` + ``config_hash``);
each sample then derives its own generator from its triple, and that single generator is
threaded through *both* engraver choice and augmentation — no transform ever touches global
``np.random``. So the same unit under the same config always yields the same image.
"""
from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from music21 import key, meter, note, stream

from omrt.datagen.augment import AugmentParams, augment
from omrt.datagen.config import GenConfig
from omrt.datagen.corpus import CorpusItem, load_corpus
from omrt.datagen.engravers import EngraverError, build_engravers, pick_engraver
from omrt.datagen.labels import canonical_musicxml
from omrt.datagen.types import CorpusId, Image, MusicXMLStr

__all__ = ["generate", "generate_with_meta", "SampleMeta"]


@dataclass(frozen=True)
class SampleMeta:
    corpus_id: CorpusId
    seed: int
    config_hash: str
    engraver: str
    augment: AugmentParams
    histograms: dict[str, Any] = field(default_factory=dict)
    image_path: str = ""
    label_path: str = ""


def generate(n: int, config: GenConfig) -> Iterator[tuple[Image, MusicXMLStr]]:
    for image, label, _meta in generate_with_meta(n, config):
        yield image, label


def generate_with_meta(
    n: int, config: GenConfig
) -> Iterator[tuple[Image, MusicXMLStr, SampleMeta]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    corpus = load_corpus(config.corpus_dir, max_staves=config.max_staves)
    if not corpus.sources:
        raise FileNotFoundError(f"no MusicXML found under {config.corpus_dir}")
    engravers = build_engravers(config.engravers)
    chash = config.config_hash()

    sel = np.random.default_rng(_derive_seed("\x00selection", config.seed, chash))
    produced = 0
    attempts = 0
    budget = n * 50 + 100  # bound the search for usable units before giving up
    while produced < n:
        attempts += 1
        if attempts > budget:
            raise RuntimeError(
                f"produced only {produced}/{n} samples before exhausting the unit budget; "
                "corpus may be too small or too many units are unrenderable"
            )
        source = corpus.sources[int(sel.integers(len(corpus.sources)))]
        units = corpus.units_in(source)
        if not units:
            continue
        item = units[int(sel.integers(len(units)))]

        srng = np.random.default_rng(_derive_seed(item.corpus_id, config.seed, chash))
        engraver = pick_engraver(engravers, srng)
        xml = canonical_musicxml(item.score)
        try:
            base = engraver.to_image(xml, dpi=config.dpi)
        except EngraverError as exc:
            warnings.warn(f"skipping {item.corpus_id}: {engraver.name} failed: {exc}")
            continue

        image, params = augment(base, srng, config)
        meta = SampleMeta(
            corpus_id=item.corpus_id,
            seed=config.seed,
            config_hash=chash,
            engraver=engraver.name,
            augment=params,
            histograms=_histograms(item),
        )
        produced += 1
        yield image, xml, meta


def _derive_seed(corpus_id: str, seed: int, config_hash: str) -> int:
    digest = hashlib.blake2b(
        f"{corpus_id}|{seed}|{config_hash}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _histograms(item: CorpusItem) -> dict[str, Any]:
    sc = item.score
    sigs = list(sc.recurse().getElementsByClass(key.KeySignature))
    times = list(sc.recurse().getElementsByClass(meter.TimeSignature))
    notes = list(sc.recurse().getElementsByClass(note.Note))
    measures = list(sc.recurse().getElementsByClass(stream.Measure))
    return {
        "unit": item.unit,
        "staves": item.staves,
        "key_sharps": sigs[0].sharps if sigs else None,
        "time_sig": times[0].ratioString if times else None,
        "note_count": len(notes),
        "measures": len(measures),
        "note_density": round(len(notes) / max(len(measures), 1), 3),
    }
