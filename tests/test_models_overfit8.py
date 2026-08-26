from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

import torch
from torch.utils.data import DataLoader

from omrt.datagen.types import Image
from omrt.eval.editdistance import levenshtein
from omrt.models import CRNN, CRNNModel, Vocabulary
from omrt.models.ctc import ctc_greedy_decode
from omrt.models.dataset import (
    PrimusDataset,
    collate_fn,
    iter_samples,
    list_incipit_ids,
    preprocess,
)
from omrt.models.metrics import dataset_token_ser
from omrt.models.train import _GRAD_CLIP_NORM, select_device

_SAMPLE = "data/primus_sample/package_aa"
_RUN = os.environ.get("RUN_CANARY") == "1"
_MAX_EPOCHS = int(os.environ.get("CANARY_EPOCHS", "2500"))

# Overfit acceptance: <=2% token error on the 8 memorized examples. NOT exact-match: greedy CTC
# cannot recover an adjacent-duplicate token (two identical eighths need a blank exactly between
# them), so exact 8/8 is flaky even for a correct model. Token-SER sidesteps that greedy edge
# while still catching a broken model cold — a blank-collapsed net scores ~1.0, fifty times over
# this bar. Reaching 0.02 also depends on the trailing whitespace margin in preprocess (ADR 0013):
# without it, greedy has no blank frame to commit the final label and SER floors near 0.076 on
# tail-token deletions. See ADR 0012 (clipping/gate) and 0013 (margin).
_SER_TARGET = 0.02


def _token_ser_on_device(
    model: CRNN,
    device: torch.device,
    samples: list[tuple[Image, list[str]]],
    vocab: Vocabulary,
) -> float:
    """Token-SER over ``samples`` via the real predict path (unpadded single image, eval mode),
    without moving the model off ``device`` — unlike ``CRNNModel``, which relocates it. Restores
    train mode so training can resume."""
    model.eval()
    ops = length = 0
    with torch.no_grad():
        for image, gold in samples:
            x = preprocess(image).unsqueeze(0).to(device)
            log_probs = model(x)
            ids = ctc_greedy_decode(log_probs.cpu(), torch.tensor([log_probs.shape[0]]))[0]
            pred = vocab.decode(ids)
            ops += levenshtein(gold, pred).distance
            length += len(gold)
    model.train()
    return ops / length if length else 0.0


@pytest.mark.skipif(
    not _RUN,
    reason="slow overfit canary — needs ~1.5k Adadelta epochs; opt in with RUN_CANARY=1 (run on GPU)",
)
@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_model_can_overfit_eight_examples() -> None:
    torch.manual_seed(0)
    ids = list_incipit_ids(_SAMPLE)[:8]
    samples = list(iter_samples(_SAMPLE, ids))
    vocab = Vocabulary.build(t for _, t in samples)
    ds = PrimusDataset(_SAMPLE, ids, vocab)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    device = select_device(os.environ.get("CANARY_DEVICE"))
    model = CRNN(vocab_size=vocab.size).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    # Shadow loss with NO masking. zero_infinity rewrites a diverged (inf) loss to ~0, which
    # would let a dead, blank-collapsed model sail past any loss gate. We assert it stays finite.
    raw_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)

    model.train()
    last = float("inf")
    raw = float("inf")
    ser = 1.0
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
            # RNN+CTC gradients explode without this; unclipped, an Adadelta(lr=1.0) step can
            # blow the LSTM up to inf, which zero_infinity then masks to 0 — see train.py.
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            opt.step()
            last = float(loss.detach())
            raw = float(raw_loss_fn(out, targets, input_lengths, target_lengths).detach())
        # Greedy decode lags the loss (CTC over-emits), so gate on decode, not a loss value.
        # Only bother once the loss is low — decoding all 8 every epoch is the expensive part.
        if last < 0.1:
            ser = _token_ser_on_device(model, device, samples, vocab)
            if ser <= _SER_TARGET:
                break

    # A low `last` must be genuine, not a zero_infinity-masked divergence.
    assert raw == raw and raw != float("inf"), (
        f"raw CTC loss non-finite ({raw}) — training diverged; zero_infinity would hide this as ~0"
    )
    assert last < 0.1, f"model failed to overfit 8 examples (loss={last:.4f}) — architecture bug"

    # Final gate through the real seam-2 path (CRNNModel.predict on unpadded single images).
    wrapped = CRNNModel(model, vocab, torch.device("cpu"))
    ser = dataset_token_ser(wrapped, _SAMPLE, ids)
    assert ser <= _SER_TARGET, f"overfit model token-SER {ser:.4f} > {_SER_TARGET} — not learning"
