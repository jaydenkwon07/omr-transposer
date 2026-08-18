from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omrt.models import CRNN, CRNNModel, Vocabulary
from omrt.models.dataset import iter_samples, list_incipit_ids
from omrt.models.metrics import dataset_token_ser
from omrt.models.train import TrainConfig, select_device, train

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


def test_select_device_falls_back_to_cpu():
    assert select_device("cpu").type == "cpu"


def test_select_device_never_auto_selects_mps():
    assert select_device(None).type in ("cuda", "cpu")


def test_train_raises_on_empty_train_ids(tmp_path):
    cfg = TrainConfig(
        root=_SAMPLE,
        out_dir=str(tmp_path),
        device="cpu",
        train_ids=[],
        val_ids=[],
    )
    with pytest.raises(ValueError):
        train(cfg)


@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_train_smoke_runs_one_step_and_checkpoints(tmp_path):
    cfg = TrainConfig(
        root=_SAMPLE,
        out_dir=str(tmp_path),
        device="cpu",
        max_steps=1,
        batch_size=4,
        val_every=1,
        train_ids=list_incipit_ids(_SAMPLE)[:8],
        val_ids=list_incipit_ids(_SAMPLE)[8:12],
    )
    ckpt = train(cfg)
    assert (tmp_path / "best.pt").exists()
    reloaded = CRNNModel.load(str(tmp_path / "best.pt"), torch.device("cpu"))
    assert reloaded.vocab.size == ckpt["vocab_size"]
