from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from omrt.models.crnn import CRNN
from omrt.models.dataset import PrimusDataset, collate_fn, iter_samples
from omrt.models.metrics import dataset_token_ser
from omrt.models.predict import CRNNModel
from omrt.models.vocab import Vocabulary

# Max global grad-norm. RNN+CTC needs clipping; without it Adadelta(lr=1.0) diverges to inf.
_GRAD_CLIP_NORM = 5.0


@dataclass
class TrainConfig:
    root: str
    out_dir: str
    device: str | None = None
    max_steps: int = 100_000
    batch_size: int = 16
    val_every: int = 1000
    patience: int = 10
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)


def select_device(override: str | None) -> torch.device:
    """cuda -> cpu, unless ``override`` is given, in which case it always wins.

    mps is deliberately never auto-selected: aten::_ctc_loss has no MPS kernel, so
    training would crash with NotImplementedError. Pass --device mps explicitly if
    you want to force it anyway.
    """
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save(
    path: str,
    model: CRNN,
    opt: torch.optim.Optimizer,
    vocab: Vocabulary,
    step: int,
    best: float,
) -> None:
    """Checkpoint contract: CRNNModel.load reads exactly the "vocab" and "model" keys.
    Do not rename or drop keys here without updating that load path."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "vocab": vocab.to_dict(),
            "step": step,
            "best_val_ser": best,
            "vocab_size": vocab.size,
        },
        path,
    )


def _run_loop(
    cfg: TrainConfig,
    device: torch.device,
    vocab: Vocabulary,
    model: CRNN,
    opt: torch.optim.Optimizer,
    start_step: int = 0,
    start_best: float = float("inf"),
) -> dict[str, object]:
    ds = PrimusDataset(cfg.root, cfg.train_ids, vocab)
    if len(ds) == 0:
        raise ValueError("no training incipits; check cfg.train_ids / --root")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    best = start_best
    since_best = 0
    step = start_step
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "val_ser"])
        done = False
        while not done:
            for images, targets, input_lengths, target_lengths in loader:
                model.train()
                images = images.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                opt.zero_grad()
                out = model(images)
                loss = loss_fn(out, targets, input_lengths, target_lengths)
                loss.backward()
                # RNN+CTC gradients explode without this; unclipped, an Adadelta(lr=1.0)
                # step can blow the LSTM up to inf, which CTCLoss(zero_infinity=True) then
                # silently masks to 0 — a "converged" loss over a dead model. See ADR 0012.
                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
                opt.step()
                step += 1

                val_ser = ""
                if step % cfg.val_every == 0 or step >= cfg.max_steps:
                    evalr = CRNNModel(model, vocab, device)
                    ser = dataset_token_ser(evalr, cfg.root, cfg.val_ids)
                    val_ser = f"{ser:.6f}"
                    if ser < best:
                        best, since_best = ser, 0
                        _save(os.path.join(cfg.out_dir, "best.pt"), model, opt, vocab, step, best)
                    else:
                        since_best += 1
                writer.writerow([step, f"{loss.item():.6f}", val_ser])
                fh.flush()
                if step >= cfg.max_steps or since_best >= cfg.patience:
                    done = True
                    break
    if not os.path.exists(os.path.join(cfg.out_dir, "best.pt")):
        _save(os.path.join(cfg.out_dir, "best.pt"), model, opt, vocab, step, best)
    return {"vocab_size": vocab.size, "best_val_ser": best, "step": step}


def train(cfg: TrainConfig) -> dict[str, object]:
    device = select_device(cfg.device)
    vocab = Vocabulary.build(t for _, t in iter_samples(cfg.root, cfg.train_ids))
    model = CRNN(vocab_size=vocab.size).to(device)
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)
    return _run_loop(cfg, device, vocab, model, opt)


def _resume(
    path: str, device: torch.device
) -> tuple[CRNN, torch.optim.Optimizer, Vocabulary, int, float]:
    """Restore model/optimizer/step/best_val_ser from a checkpoint written by ``_save``.
    Caller must reuse the ``train_ids`` the checkpoint's vocab was built from — a resumed
    run does not re-derive the vocab, so a different train_ids would silently desync it."""
    ckpt: dict[str, Any] = torch.load(path, map_location=device)
    vocab = Vocabulary.from_dict(ckpt["vocab"])
    model = CRNN(vocab_size=vocab.size).to(device)
    model.load_state_dict(ckpt["model"])
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)
    opt.load_state_dict(ckpt["optimizer"])
    return model, opt, vocab, int(ckpt["step"]), float(ckpt["best_val_ser"])


def load_split(path: str) -> tuple[list[str], list[str]]:
    """Read a ``split.json`` of the form ``{"train": [...], "val": [...], "test": [...]}``
    and return ``(train_ids, val_ids)``. The ``test`` key is deliberately never returned —
    it is the held-out set and must never feed training or validation."""
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    return split["train"], split["val"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the seam-2 CRNN+CTC model on PrIMuS.")
    parser.add_argument("--root", required=True, help="PrIMuS incipit tree root")
    parser.add_argument("--out-dir", required=True, help="checkpoint/log output directory")
    parser.add_argument("--device", default=None, help="override device (else cuda->cpu)")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    parser.add_argument(
        "--split",
        default="data/primus/split.json",
        help="path to split.json ({train, val, test} incipit id lists); test is never used here",
    )
    args = parser.parse_args()

    train_ids, val_ids = load_split(args.split)
    cfg = TrainConfig(
        root=args.root,
        out_dir=args.out_dir,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        val_every=args.val_every,
        patience=args.patience,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    if args.resume:
        device = select_device(args.device)
        model, opt, vocab, step, best = _resume(args.resume, device)
        result = _run_loop(cfg, device, vocab, model, opt, start_step=step, start_best=best)
    else:
        result = train(cfg)
    print(result)


if __name__ == "__main__":
    main()
