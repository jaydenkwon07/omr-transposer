from __future__ import annotations

import glob
import os
from typing import Iterator

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from omrt.datagen.engravers.base import read_gray_over_white
from omrt.datagen.types import Image
from omrt.decode import parse_semantic
from omrt.models.vocab import Vocabulary

_TARGET_HEIGHT = 128


def preprocess(image: Image) -> Tensor:
    """Grayscale staff -> [1, 128, W] float tensor, aspect ratio preserved, values in [0,1]
    (paper≈1.0, ink≈0.0). Height is fixed at 128 (paper Table 2).

    OMRT_TRAIL_PAD (experimental): append N white columns on the right. PrIMuS incipits are
    cropped flush to the last symbol (0-1 trailing px), leaving CTC greedy no blank frame to
    commit the final label -> systematic tail-token deletion. Applied in BOTH train and eval
    (this one function feeds the dataset and CRNNModel.predict), so the frame budget stays
    consistent. Default 0 = no change."""
    h, w = image.shape[:2]
    new_w = max(1, round(w * (_TARGET_HEIGHT / h)))
    resized = cv2.resize(image, (new_w, _TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    trail = int(os.environ.get("OMRT_TRAIL_PAD", "0"))
    if trail > 0:
        arr = np.concatenate([arr, np.ones((_TARGET_HEIGHT, trail), dtype=np.float32)], axis=1)
    return torch.from_numpy(arr).unsqueeze(0)


def list_incipit_ids(root: str) -> list[str]:
    """Sorted incipit ids under a PrIMuS tree (one dir per incipit holding <id>.png/.semantic).

    Called at test-collection time by ``skipif`` before torch may be importable, so this
    function must stay stdlib-only (``glob``, ``os``) even though the rest of this module
    imports torch/cv2 at the top.
    """
    ids = []
    for semantic in glob.glob(os.path.join(root, "**", "*.semantic"), recursive=True):
        stem = os.path.basename(semantic)[: -len(".semantic")]
        if os.path.exists(os.path.join(os.path.dirname(semantic), stem + ".png")):
            ids.append(stem)
    return sorted(ids)


def _paths(root: str, incipit_id: str) -> tuple[str, str]:
    d = os.path.join(root, incipit_id)
    return os.path.join(d, incipit_id + ".png"), os.path.join(d, incipit_id + ".semantic")


def iter_samples(root: str, ids: list[str]) -> Iterator[tuple[Image, list[str]]]:
    for incipit_id in ids:
        png, semantic = _paths(root, incipit_id)
        image = read_gray_over_white(png)
        with open(semantic, encoding="utf-8") as fh:
            tokens = parse_semantic(fh.read())
        yield image, tokens


def _pad_width_to(t: Tensor, min_w: int) -> Tensor:
    if t.shape[2] >= min_w:
        return t
    pad = torch.ones((t.shape[0], t.shape[1], min_w - t.shape[2]), dtype=t.dtype)  # paper=1.0
    return torch.cat([t, pad], dim=2)


def collate_fn(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batch (image, target) pairs into padded tensors + CTC length tensors.

    Args:
        batch: list of (image[1, 128, W_i], target[L_i]) tuples

    Returns:
        (images[N, 1, 128, Wmax], targets[ΣL_i] int64, input_lengths[N] int64, target_lengths[N] int64)
        where input_lengths[i] = W_i // 4, target_lengths[i] = L_i.

    Asserts CTC width constraint: input_lengths[i] >= target_lengths[i] for all i.
    Images right-padded to batch-max width with paper=1.0.
    Targets concatenated into 1-D tensor.
    """
    input_lengths = torch.tensor(
        [img.shape[2] // 4 for img, _ in batch], dtype=torch.int64
    )
    target_lengths = torch.tensor([tgt.shape[0] for _, tgt in batch], dtype=torch.int64)
    assert torch.all(
        input_lengths >= target_lengths
    ), "CTC width constraint violated: W/4 < L"
    max_w = max(img.shape[2] for img, _ in batch)
    images = torch.ones(len(batch), 1, 128, max_w)  # paper=1.0
    for i, (img, _) in enumerate(batch):
        images[i, :, :, : img.shape[2]] = img
    targets = torch.cat([tgt for _, tgt in batch])
    return images, targets, input_lengths, target_lengths


class PrimusDataset(Dataset[tuple[Tensor, Tensor]]):
    """PrIMuS incipits as ``(image[1,128,W], target_ids[L])`` pairs, with ``W >= 4*L``
    guaranteed by right-padding with paper so CTC always has enough frames to emit L symbols."""

    def __init__(self, root: str, ids: list[str], vocab: Vocabulary) -> None:
        self.root = root
        self.ids = ids
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        png, semantic = _paths(self.root, self.ids[index])
        image = read_gray_over_white(png)
        with open(semantic, encoding="utf-8") as fh:
            tokens = parse_semantic(fh.read())
        target = torch.tensor(self.vocab.encode(tokens), dtype=torch.int64)
        t = _pad_width_to(preprocess(image), 4 * max(1, int(target.shape[0])))
        return t, target
