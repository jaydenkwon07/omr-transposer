"""Project 1: the synthetic data generator.

MusicXML corpus in, ``(image, MusicXML label)`` pairs out. Honors seam 1:
``generate(n, config) -> Iterator[tuple[Image, MusicXMLStr]]``. No model, no torch — the
label is MusicXML, never tokens.
"""
from omrt.datagen.config import GenConfig
from omrt.datagen.corpus import Corpus, CorpusItem, load_corpus
from omrt.datagen.dataset import write_dataset
from omrt.datagen.engravers import Engraver, build_engravers, pick_engraver
from omrt.datagen.generate import SampleMeta, generate, generate_with_meta
from omrt.datagen.types import CorpusId, Image, MusicXMLStr

__all__ = [
    "generate",
    "generate_with_meta",
    "GenConfig",
    "SampleMeta",
    "Image",
    "MusicXMLStr",
    "CorpusId",
    "Corpus",
    "CorpusItem",
    "load_corpus",
    "write_dataset",
    "Engraver",
    "build_engravers",
    "pick_engraver",
]
