import pytest

pytest.importorskip("torch")

import torch

from omrt.models.ctc import ctc_greedy_decode


def _one_hot(seq, num_classes):
    # seq: list of class ids per frame -> log-prob-ish tensor [T,1,C]
    t = torch.full((len(seq), 1, num_classes), -20.0)
    for i, c in enumerate(seq):
        t[i, 0, c] = 0.0
    return t


def test_collapses_repeats_and_drops_blank():
    # frames: 2,2,0(blank),2,3,3  -> 2,2,3
    logits = _one_hot([2, 2, 0, 2, 3, 3], num_classes=4)
    out = ctc_greedy_decode(logits, torch.tensor([6]))
    assert out == [[2, 2, 3]]


def test_respects_input_lengths():
    logits = _one_hot([2, 3, 0, 0], num_classes=4)
    out = ctc_greedy_decode(logits, torch.tensor([2]))  # only first 2 frames
    assert out == [[2, 3]]
