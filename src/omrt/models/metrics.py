from __future__ import annotations

from omrt.decode import decode
from omrt.eval.editdistance import levenshtein
from omrt.eval.metrics import evaluate
from omrt.models.dataset import iter_samples
from omrt.models.predict import CRNNModel


def dataset_token_ser(model: CRNNModel, root: str, ids: list[str]) -> float:
    """Symbol Error Rate in token space (paper's SER): total edit ops / total gold length."""
    ops = length = 0
    for image, gold in iter_samples(root, ids):
        pred = model.predict(image)
        ops += levenshtein(gold, pred).distance
        length += len(gold)
    return ops / length if length else 0.0


def dataset_musicxml_ser(model: CRNNModel, root: str, ids: list[str]) -> float:
    """Canonical seam-4 cross-check: SER in MusicXML space via decode(pred) vs decode(gold).
    Guaranteed ≈ token SER by the 0.0045% decode ceiling (ADR 0011)."""
    total = 0.0
    n = 0
    for image, gold in iter_samples(root, ids):
        pred = model.predict(image)
        total += evaluate(decode(pred), decode(gold)).ser
        n += 1
    return total / n if n else 0.0
