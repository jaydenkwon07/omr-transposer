"""Overfit-8 diagnostic: separate BatchNorm train/eval gap from optimization plateau.

Run from the repo root on Colab (GPU):  RUN it as a cell, or `python canary_diag.py`.
Logs, every 100 epochs: train-mode loss (batch stats), eval-mode loss (running stats,
same batch), and eval-mode token-SER via the real predict path.

Reading:
  - train loss LOW but eval loss HIGH  => BatchNorm running-stat gap (the eval/predict
    path never sees what train mode fits). Fix is about BN, not training longer.
  - both losses descend together but slowly => plain undertraining / clip too tight.
  - both plateau high => optimization is stuck.
"""
from __future__ import annotations

import difflib
import os

import torch
from torch.utils.data import DataLoader

from omrt.eval.editdistance import levenshtein
from omrt.models import CRNN, Vocabulary
from omrt.models.ctc import ctc_greedy_decode
from omrt.models.dataset import (
    PrimusDataset,
    collate_fn,
    iter_samples,
    list_incipit_ids,
    preprocess,
)
from omrt.models.train import _GRAD_CLIP_NORM, select_device

SAMPLE = "data/primus_sample/package_aa"
MAX_EPOCHS = int(os.environ.get("CANARY_EPOCHS", "2500"))


def token_ser(model, device, samples, vocab) -> float:
    model.eval()
    ops = length = 0
    with torch.no_grad():
        for image, gold in samples:
            x = preprocess(image).unsqueeze(0).to(device)
            lp = model(x)
            ids = ctc_greedy_decode(lp.cpu(), torch.tensor([lp.shape[0]]))[0]
            pred = vocab.decode(ids)
            ops += levenshtein(gold, pred).distance
            length += len(gold)
    model.train()
    return ops / length if length else 0.0


def dump_diffs(model, device, samples, vocab) -> None:
    """Print per-example gold-vs-greedy-pred diffs so we can see *which* tokens greedy
    misses on a fully-trained model (adjacent-dup collapse vs a real error pattern)."""
    model.eval()
    print("\n=== per-example diffs (- gold-only, + pred-only) ===")
    with torch.no_grad():
        for idx, (image, gold) in enumerate(samples):
            x = preprocess(image).unsqueeze(0).to(device)
            lp = model(x)
            ids = ctc_greedy_decode(lp.cpu(), torch.tensor([lp.shape[0]]))[0]
            pred = vocab.decode(ids)
            d = levenshtein(gold, pred).distance
            print(f"\n[{idx}] L={len(gold)} pred_len={len(pred)} ops={d}")
            if d:
                sm = difflib.SequenceMatcher(a=gold, b=pred, autojunk=False)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        continue
                    print(f"   {tag:8} gold[{i1}:{i2}]={gold[i1:i2]}  pred[{j1}:{j2}]={pred[j1:j2]}")
    model.train()


def main() -> None:
    torch.manual_seed(0)
    ids = list_incipit_ids(SAMPLE)[:8]
    samples = list(iter_samples(SAMPLE, ids))
    vocab = Vocabulary.build(t for _, t in samples)
    loader = DataLoader(PrimusDataset(SAMPLE, ids, vocab), batch_size=8, collate_fn=collate_fn)
    device = select_device(os.environ.get("CANARY_DEVICE"))
    print("device:", device)

    model = CRNN(vocab_size=vocab.size).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for images, targets, il, tl in loader:
            images, targets, il, tl = (
                images.to(device), targets.to(device), il.to(device), tl.to(device),
            )
            opt.zero_grad()
            out = model(images)
            loss = loss_fn(out, targets, il, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            opt.step()
        train_loss = float(loss.detach())

        if epoch % 100 == 0 or epoch == 1:
            # eval-mode loss on the SAME batch (BN running stats instead of batch stats)
            model.eval()
            with torch.no_grad():
                for images, targets, il, tl in loader:
                    images, targets, il, tl = (
                        images.to(device), targets.to(device), il.to(device), tl.to(device),
                    )
                    eval_loss = float(loss_fn(model(images), targets, il, tl).detach())
            model.train()
            ser = token_ser(model, device, samples, vocab)
            print(f"epoch {epoch:5d}  train_loss {train_loss:8.4f}  "
                  f"eval_loss {eval_loss:8.4f}  eval_token_SER {ser:.4f}")
            if ser <= 0.02:
                print("REACHED target SER<=0.02")
                break

    dump_diffs(model, device, samples, vocab)


if __name__ == "__main__":
    main()
