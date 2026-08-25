from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

import torch
from torch.utils.data import DataLoader

from omrt.models import CRNN, CRNNModel, Vocabulary
from omrt.models.dataset import (
    PrimusDataset,
    collate_fn,
    iter_samples,
    list_incipit_ids,
)
from omrt.models.train import select_device

_SAMPLE = "data/primus_sample/package_aa"
_RUN = os.environ.get("RUN_CANARY") == "1"
_MAX_EPOCHS = int(os.environ.get("CANARY_EPOCHS", "2000"))


@pytest.mark.skipif(
    not _RUN,
    reason="slow overfit canary — needs many hundreds of Adadelta epochs; opt in with RUN_CANARY=1 (run on GPU)",
)
@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_model_can_overfit_eight_examples():
    torch.manual_seed(0)
    ids = list_incipit_ids(_SAMPLE)[:8]
    vocab = Vocabulary.build(t for _, t in iter_samples(_SAMPLE, ids))
    ds = PrimusDataset(_SAMPLE, ids, vocab)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    device = select_device(os.environ.get("CANARY_DEVICE"))
    model = CRNN(vocab_size=vocab.size).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)

    model.train()
    last = float("inf")
    for _ in range(_MAX_EPOCHS):
        for images, targets, input_lengths, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            opt.zero_grad()
            out = model(images)
            loss = loss_fn(out, targets, input_lengths, target_lengths)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        if last < 0.05:
            break
    assert last < 0.1, f"model failed to overfit 8 examples (loss={last:.4f}) — architecture bug"

    wrapped = CRNNModel(model, vocab, torch.device("cpu"))
    exact = sum(
        1 for image, tokens in iter_samples(_SAMPLE, ids) if wrapped.predict(image) == tokens
    )
    assert exact == 8, f"overfit model decoded {exact}/8 exactly"
