from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omrt.models import CRNN, CRNNModel, Vocabulary
from omrt.models.dataset import iter_samples, list_incipit_ids
from omrt.models.metrics import dataset_token_ser

_SAMPLE = "data/primus_sample/package_aa"


class _PerfectModel(CRNNModel):
    """Test double: predicts each image's gold tokens by lookup. Verifies the metric's
    aggregation, not the network."""

    def __init__(self, gold: dict[int, list[str]]) -> None:  # noqa: super not called on purpose
        self._gold = gold

    def predict(self, image):  # type: ignore[override]
        return self._gold[int(image.sum())]


@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_token_ser_is_zero_for_perfect_predictions():
    ids = list_incipit_ids(_SAMPLE)[:5]
    gold = {int(img.sum()): toks for img, toks in iter_samples(_SAMPLE, ids)}
    model = _PerfectModel(gold)
    assert dataset_token_ser(model, _SAMPLE, ids) == 0.0
